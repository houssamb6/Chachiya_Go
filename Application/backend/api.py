"""
chouchane_api.py
================
Chouchane — Tunisia Tourism AI Workflow

Phases:
  Phase 1 — YASMINE        : RAG + Gemini tourism recommendation agent
  Phase 2 — QA             : Tunisia expert Q&A once Yasmine ends

State flags:
  yasmine_done : False → True once place chosen & partners shown
  QA is always available once yasmine_done = True

Endpoints:
  GET  /health               — health check
  GET  /places               — all places from RAG db (for frontend)
  POST /session/start        — start session, get WELCOMING + Yasmine greeting
  POST /session/reset        — reset everything back to Phase 1
  GET  /session/{session_id} — inspect session state (debug)
  POST /yasmine              — chat with Yasmine (while yasmine_done = False)
  POST /qa                   — ask anything about Tunisia (open once yasmine_done)
"""

import json
import uuid
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

from agent.conversation import TunisiaTourismAgent
from config.settings import API_KEY, GEMINI_MODEL
from config.content import WELCOMING
from data.places_db import PLACES_DB

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

SESSIONS_FILE = Path("chouchane_sessions.json")

QA_SYSTEM_PROMPT = """
You are a knowledgeable and enthusiastic Tunisia travel expert working within Chouchane,
a Tunisia tourism AI experience.
You have deep knowledge of Tunisia's history, culture, food, geography,
tourism destinations, traditions, language, and people.
Answer every question in a warm, informative, and engaging way.
Keep answers concise but rich — 2 to 4 sentences unless more detail is needed.
If the question is not related to Tunisia, politely redirect:
"That's a bit outside my expertise — I'm your Tunisia specialist inside Chouchane! Ask me anything about this amazing country."
Do not use markdown, asterisks, or emojis.
""".strip()

# ─────────────────────────────────────────────────────────────────
# SESSION STORAGE  (JSON file)
# ─────────────────────────────────────────────────────────────────

def _load_sessions() -> dict:
    if SESSIONS_FILE.exists():
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    return {}

def _save_sessions(sessions: dict):
    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2, ensure_ascii=False), encoding="utf-8")

def _get_session(session_id: str) -> dict:
    sessions = _load_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /session/start first.")
    return sessions[session_id]

def _update_session(session_id: str, data: dict):
    sessions = _load_sessions()
    sessions[session_id] = data
    _save_sessions(sessions)

def _delete_session(session_id: str):
    sessions = _load_sessions()
    sessions.pop(session_id, None)
    _save_sessions(sessions)

def _new_session(session_id: str) -> dict:
    data = {
        "session_id": session_id,
        "workflow":   "chouchane",

        # ── Phase flags ───────────────────────────────────────────
        "yasmine_done":  False,  # True once place chosen & partners shown

        # ── Phase 1: Yasmine ──────────────────────────────────────
        "yasmine_history":               [],
        "yasmine_prefs":                 {},
        "yasmine_recommendations_given": False,
        "yasmine_recommended_places":    [],
        "yasmine_chosen_place":          None,
        "yasmine_partners_shown":        False,

        # ── Phase 2: Q&A ──────────────────────────────────────────
        "qa_history":     [],
    }
    _update_session(session_id, data)
    return data

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return text.strip().lower().replace("'", "").replace("-", " ")

def _clean_text(text: str) -> str:
    text = text.replace("*", "")
    text = re.sub(r"[\U0001F000-\U0001FFFF\U00002700-\U000027BF\U0001F900-\U0001F9FF\u2600-\u26FF]", "", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def _history_to_contents(history: list) -> list:
    return [
        types.Content(role=h["role"], parts=[types.Part(text=h["text"])])
        for h in history
    ]

def _active_phases(session: dict) -> list[str]:
    """Compute which phases are open — used in every response."""
    if not session["yasmine_done"]:
        return ["yasmine"]
    return ["qa"]


def _primary_phase(session: dict) -> str:
    """
    Map internal flags to the single phase string expected by the frontend.

    - "yasmine" while the preference-gathering agent is running
    - "qa"      after a place is chosen (Q&A stays open)
    """
    if not session["yasmine_done"]:
        return "yasmine"
    return "qa"

# ─────────────────────────────────────────────────────────────────
# YASMINE — rebuild agent from session / persist back
# ─────────────────────────────────────────────────────────────────

def _build_yasmine_agent(session: dict) -> TunisiaTourismAgent:
    agent = TunisiaTourismAgent(api_key=API_KEY)
    agent.history               = _history_to_contents(session["yasmine_history"])
    agent.preferences           = session["yasmine_prefs"]
    agent.recommendations_given = session["yasmine_recommendations_given"]
    agent.recommended_places    = session["yasmine_recommended_places"]
    agent.chosen_place          = session["yasmine_chosen_place"]
    agent._partners_shown       = session["yasmine_partners_shown"]
    return agent

def _persist_yasmine_agent(agent: TunisiaTourismAgent, session: dict) -> dict:
    session["yasmine_history"] = [
        {"role": c.role, "text": c.parts[0].text}
        for c in agent.history
    ]
    session["yasmine_prefs"]                  = agent.preferences
    session["yasmine_recommendations_given"]  = agent.recommendations_given
    session["yasmine_recommended_places"]     = agent.recommended_places
    session["yasmine_chosen_place"]           = agent.chosen_place
    session["yasmine_partners_shown"]         = agent._partners_shown
    return session

# ─────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Chouchane API",
    description=(
        "Chouchane — Tunisia Tourism AI Workflow\n\n"
        "Phase 1: Yasmine recommends destinations based on your preferences.\n"
        "Phase 2: QA — Ask anything about Tunisia once Yasmine ends."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini_client = genai.Client(api_key=API_KEY)

# ─────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    session_id: str
    message:    str

class ChouchaneResponse(BaseModel):
    session_id:      str
    workflow:        str        = "chouchane"
    phase:           str
    active_phases:   list[str]              # ["yasmine"] | ["qa"]
    reply:           str
    # Yasmine extras
    partners:        Optional[str]  = None
    chosen_place:    Optional[str]  = None

# ─────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "workflow": "chouchane"}


@app.get("/places")
def get_places():
    """
    Return all places from the RAG database for the frontend.
    Each place includes name, region, description, and metadata.
    """
    places = []
    for name, data in PLACES_DB.items():
        place = {
            "name":           name,
            "region":         data.get("region", ""),
            "vibe":           data.get("vibe", ""),
            "description":    data.get("description", ""),
            "top_activities": data.get("top_activities", []),
            "insider_tip":    data.get("insider_tip", ""),
            "season":         data.get("season", ""),
        }
        places.append(place)
    return {"places": places}


@app.post("/session/start", response_model=ChouchaneResponse)
def session_start():
    """
    Start a new Chouchane session.
    Returns the WELCOMING message + Yasmine's opening greeting.
    active_phases = ["yasmine"]
    """
    session_id = str(uuid.uuid4())
    session    = _new_session(session_id)

    agent = TunisiaTourismAgent(api_key=API_KEY)
    yasmine_greeting, _ = agent.start()
    reply = f"{WELCOMING}\n\n{_clean_text(yasmine_greeting)}"

    session = _persist_yasmine_agent(agent, session)
    _update_session(session_id, session)

    return ChouchaneResponse(
        session_id=session_id,
        phase="yasmine",
        active_phases=_active_phases(session),
        reply=reply,
    )


@app.post("/session/reset", response_model=ChouchaneResponse)
def session_reset(body: MessageRequest):
    """
    Reset a Chouchane session back to Phase 1 (Yasmine).
    active_phases = ["yasmine"]
    """
    _delete_session(body.session_id)
    session = _new_session(body.session_id)

    agent = TunisiaTourismAgent(api_key=API_KEY)
    yasmine_greeting, _ = agent.start()
    reply = f"{WELCOMING}\n\n{_clean_text(yasmine_greeting)}"

    session = _persist_yasmine_agent(agent, session)
    _update_session(body.session_id, session)

    return ChouchaneResponse(
        session_id=body.session_id,
        phase="yasmine",
        active_phases=_active_phases(session),
        reply=reply,
    )


@app.get("/session/{session_id}")
def session_inspect(session_id: str):
    """Inspect the full state of a Chouchane session (debug)."""
    session = _get_session(session_id)
    # Mirror the shape expected by the frontend (adds a synthetic `phase` field).
    return {
        **session,
        "phase": _primary_phase(session),
    }


# ── Phase 1: Yasmine ─────────────────────────────────────────────

@app.post("/yasmine", response_model=ChouchaneResponse)
def yasmine_chat(body: MessageRequest):
    """
    Phase 1 — Chat with Yasmine.
    When a place is chosen, yasmine_done flips to True and
    active_phases becomes ["qa"].
    """
    session = _get_session(body.session_id)

    if session["yasmine_done"]:
        raise HTTPException(
            status_code=400,
            detail="Yasmine phase is complete. Use /qa for questions about Tunisia."
        )

    agent           = _build_yasmine_agent(session)
    reply, partners = agent.chat(body.message)
    reply           = _clean_text(reply)
    session         = _persist_yasmine_agent(agent, session)

    # Place chosen → unlock QA
    if partners:
        session["yasmine_done"] = True

    _update_session(body.session_id, session)

    return ChouchaneResponse(
        session_id=body.session_id,
        phase=_primary_phase(session),
        active_phases=_active_phases(session),
        reply=reply,
        partners=partners,
        chosen_place=session["yasmine_chosen_place"],
    )


# ── Phase 2: Tunisia Q&A ─────────────────────────────────────────

@app.post("/qa", response_model=ChouchaneResponse)
def qa_chat(body: MessageRequest):
    """
    Tunisia Q&A — open as soon as Yasmine ends.
    Ask anything about Tunisia. Multi-turn context is preserved.
    """
    session = _get_session(body.session_id)

    if not session["yasmine_done"]:
        raise HTTPException(
            status_code=400,
            detail="Complete the Yasmine phase first — Q&A opens once you pick a destination."
        )

    qa_history = session.get("qa_history", [])
    contents   = _history_to_contents(qa_history)
    contents.append(types.Content(role="user", parts=[types.Part(text=body.message)]))

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=QA_SYSTEM_PROMPT),
    )
    reply = _clean_text(response.text)

    session["qa_history"] = qa_history + [
        {"role": "user",  "text": body.message},
        {"role": "model", "text": reply},
    ]
    _update_session(body.session_id, session)

    return ChouchaneResponse(
        session_id=body.session_id,
        phase=_primary_phase(session),
        active_phases=_active_phases(session),
        reply=reply,
    )