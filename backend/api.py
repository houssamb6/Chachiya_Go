"""
chouchane_api.py
================
Chouchane — Tunisia Tourism AI Workflow

Phases:
  Phase 1 — YASMINE   : RAG + Gemini tourism recommendation agent
  Phase 2 — QUIZ      : 1 question about the chosen destination (auto-hints on wrong answer)
  Phase 3 — TUNISIA Q&A : Gemini-powered assistant for anything about Tunisia

Endpoints:
  POST /session/start        — start a Chouchane session, get Yasmine's greeting
  POST /yasmine              — chat with Yasmine (phase: yasmine)
  POST /quiz/start           — load the quiz question for the chosen place (phase: quiz)
  POST /quiz/answer          — submit an answer or 'hint' (phase: quiz)
  POST /qa                   — ask anything about Tunisia (phase: qa)
  POST /session/reset        — reset session back to phase 1
  GET  /session/{session_id} — inspect session state (debug)
  GET  /health               — health check
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
from data.quiz_db import QUIZ_DATA
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
        "phase":      "yasmine",        # yasmine | quiz | qa
        # ── Phase 1: Yasmine ──────────────────────────────────────
        "yasmine_history":                [],
        "yasmine_prefs":                  {},
        "yasmine_recommendations_given":  False,
        "yasmine_recommended_places":     [],
        "yasmine_chosen_place":           None,
        "yasmine_partners_shown":         False,
        # ── Phase 2: Quiz ─────────────────────────────────────────
        "destination":    "",
        "question":       "",
        "hints":          [],
        "correct_answer": "",
        "hints_used":     0,
        "quiz_score":     0,
        "quiz_done":      False,
        # ── Phase 3: Q&A ──────────────────────────────────────────
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

def _get_quiz_for_place(place_name: str) -> Optional[dict]:
    place_lower = place_name.lower()
    for q in QUIZ_DATA:
        if q["destination"].lower() in place_lower or place_lower in q["destination"].lower():
            return q
    first_word = place_lower.split()[0]
    for q in QUIZ_DATA:
        if first_word in q["destination"].lower():
            return q
    return None

def _history_to_contents(history: list) -> list:
    return [
        types.Content(role=h["role"], parts=[types.Part(text=h["text"])])
        for h in history
    ]

# ─────────────────────────────────────────────────────────────────
# YASMINE — rebuild agent from session / persist back
# ─────────────────────────────────────────────────────────────────

def _build_yasmine_agent(session: dict) -> TunisiaTourismAgent:
    agent = TunisiaTourismAgent(api_key=API_KEY)
    agent.history                  = _history_to_contents(session["yasmine_history"])
    agent.preferences              = session["yasmine_prefs"]
    agent.recommendations_given    = session["yasmine_recommendations_given"]
    agent.recommended_places       = session["yasmine_recommended_places"]
    agent.chosen_place             = session["yasmine_chosen_place"]
    agent._partners_shown          = session["yasmine_partners_shown"]
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
        "Phase 2: Quiz about your chosen destination.\n"
        "Phase 3: Tunisia Q&A — ask anything about Tunisia."
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
    message: str

class ChouchaneResponse(BaseModel):
    session_id:      str
    workflow:        str = "chouchane"
    phase:           str           # yasmine | quiz | qa
    reply:           str
    # Yasmine extras
    partners:        Optional[str]  = None
    chosen_place:    Optional[str]  = None
    # Quiz extras
    quiz_question:   Optional[str]  = None
    hints_used:      Optional[int]  = None
    hints_remaining: Optional[int]  = None
    quiz_done:       Optional[bool] = None
    quiz_score:      Optional[int]  = None

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
    Each place includes name, region, description, and optional quiz info.
    """
    places = []
    for name, data in PLACES_DB.items():
        q = _get_quiz_for_place(name)
        place = {
            "name": name,
            "region": data.get("region", ""),
            "vibe": data.get("vibe", ""),
            "description": data.get("description", ""),
            "top_activities": data.get("top_activities", []),
            "insider_tip": data.get("insider_tip", ""),
            "season": data.get("season", ""),
        }
        if q:
            place["quiz"] = {
                "question": q["question"],
                "hints": q["hints"],
                "answer": q["answer"],
            }
        places.append(place)
    return {"places": places}


@app.post("/session/start", response_model=ChouchaneResponse)
def session_start():
    """
    Start a new Chouchane session.
    Returns the WELCOMING message from content.py followed by Yasmine's opening greeting.
    """
    session_id = str(uuid.uuid4())
    session    = _new_session(session_id)

    agent        = TunisiaTourismAgent(api_key=API_KEY)
    yasmine_greeting, _ = agent.start()
    yasmine_greeting    = _clean_text(yasmine_greeting)

    # Combine WELCOMING banner + Yasmine's greeting as one first message
    reply = f"{WELCOMING}\n\n{yasmine_greeting}"

    session = _persist_yasmine_agent(agent, session)
    _update_session(session_id, session)

    return ChouchaneResponse(
        session_id=session_id,
        phase="yasmine",
        reply=reply,
    )


@app.post("/session/reset", response_model=ChouchaneResponse)
def session_reset(body: MessageRequest):
    """
    Reset a Chouchane session back to Phase 1 (Yasmine).
    Returns the WELCOMING message + Yasmine's opening greeting again.
    """
    _delete_session(body.session_id)
    session = _new_session(body.session_id)

    agent    = TunisiaTourismAgent(api_key=API_KEY)
    yasmine_greeting, _ = agent.start()
    yasmine_greeting    = _clean_text(yasmine_greeting)

    reply = f"{WELCOMING}\n\n{yasmine_greeting}"

    session = _persist_yasmine_agent(agent, session)
    _update_session(body.session_id, session)

    return ChouchaneResponse(
        session_id=body.session_id,
        phase="yasmine",
        reply=reply,
    )


@app.get("/session/{session_id}")
def session_inspect(session_id: str):
    """Inspect the full state of a Chouchane session (debug)."""
    return _get_session(session_id)


# ── Phase 1: Yasmine ─────────────────────────────────────────────

@app.post("/yasmine", response_model=ChouchaneResponse)
def yasmine_chat(body: MessageRequest):
    """
    Phase 1 — Chat with Yasmine.
    Yasmine asks about your travel preferences and recommends 2 destinations.
    When a place is chosen, partners (hotels & restaurants) are returned
    and the phase automatically transitions to 'quiz'.
    """
    session = _get_session(body.session_id)

    if session["phase"] != "yasmine":
        raise HTTPException(
            status_code=400,
            detail=f"Chouchane is in phase '{session['phase']}'. Use the correct endpoint."
        )

    agent        = _build_yasmine_agent(session)
    reply, partners = agent.chat(body.message)
    reply        = _clean_text(reply)
    session      = _persist_yasmine_agent(agent, session)

    if partners:
        session["phase"] = "quiz"

    _update_session(body.session_id, session)

    return ChouchaneResponse(
        session_id=body.session_id,
        phase=session["phase"],
        reply=reply,
        partners=partners,
        chosen_place=session["yasmine_chosen_place"],
    )


# ── Phase 2: Quiz ────────────────────────────────────────────────

@app.post("/quiz/start", response_model=ChouchaneResponse)
def quiz_start(body: MessageRequest):
    """
    Phase 2 — Load the quiz question for the chosen destination.
    Call once after /yasmine returns phase='quiz'.
    """
    session = _get_session(body.session_id)

    if session["phase"] != "quiz":
        raise HTTPException(
            status_code=400,
            detail=f"Chouchane is in phase '{session['phase']}'. Complete Phase 1 first."
        )

    chosen = session.get("yasmine_chosen_place", "")
    q      = _get_quiz_for_place(chosen)

    if not q:
        raise HTTPException(status_code=404, detail=f"No quiz question found for '{chosen}'.")

    session["destination"]    = q["destination"]
    session["question"]       = q["question"]
    session["hints"]          = q["hints"]
    session["correct_answer"] = q["answer"]
    session["hints_used"]     = 0
    session["quiz_score"]     = 0
    session["quiz_done"]      = False
    _update_session(body.session_id, session)

    reply = (
        f"Quiz time! Here is your question about {q['destination']}:\n\n"
        f"{q['question']}\n\n"
        f"Type your answer, or send 'hint' for a clue. "
        f"You have {len(q['hints'])} hint(s) available."
    )

    return ChouchaneResponse(
        session_id=body.session_id,
        phase="quiz",
        reply=reply,
        quiz_question=q["question"],
        hints_used=0,
        hints_remaining=len(q["hints"]),
        quiz_done=False,
    )


@app.post("/quiz/answer", response_model=ChouchaneResponse)
def quiz_answer(body: MessageRequest):
    """
    Phase 2 — Submit a quiz answer or 'hint'.
    Wrong answer → next hint auto-revealed.
    All hints exhausted + wrong → phase moves to 'qa'.
    Correct answer → phase moves to 'qa'.
    """
    session = _get_session(body.session_id)

    if session["phase"] != "quiz":
        raise HTTPException(
            status_code=400,
            detail=f"Chouchane is in phase '{session['phase']}'."
        )
    if session["quiz_done"]:
        raise HTTPException(status_code=400, detail="Quiz already completed. Use /qa.")

    hints      = session["hints"]
    hints_used = session["hints_used"]
    user_input = body.message.strip()

    # ── Manual hint request ───────────────────────────────────────
    if _normalize(user_input) == "hint":
        if hints_used < len(hints):
            new_used  = hints_used + 1
            hint_text = hints[hints_used]
            remaining = len(hints) - new_used
            session["hints_used"] = new_used
            _update_session(body.session_id, session)

            reply = f"Hint {new_used}: {hint_text}"
            reply += f"\n({remaining} hint{'s' if remaining > 1 else ''} left)" if remaining > 0 else "\n(Last hint! Give it your best shot)"

            return ChouchaneResponse(
                session_id=body.session_id,
                phase="quiz",
                reply=reply,
                hints_used=new_used,
                hints_remaining=remaining,
                quiz_done=False,
            )
        return ChouchaneResponse(
            session_id=body.session_id,
            phase="quiz",
            reply="No more hints available! Give your best answer.",
            hints_used=hints_used,
            hints_remaining=0,
            quiz_done=False,
        )

    # ── Evaluate answer ───────────────────────────────────────────
    is_correct = _normalize(user_input) == _normalize(session["correct_answer"])

    # ✅ Correct
    if is_correct:
        stars = "★" * max(1, 3 - hints_used)
        reply = (
            f"Correct! {stars}\n"
            f"{'No hints used — perfect!' if hints_used == 0 else f'{hints_used} hint(s) used'}\n\n"
            f"You are now entering the Chouchane Q&A — ask me anything about Tunisia!"
        )
        session["quiz_score"] = 1
        session["quiz_done"]  = True
        session["phase"]      = "qa"
        _update_session(body.session_id, session)

        return ChouchaneResponse(
            session_id=body.session_id,
            phase="qa",
            reply=reply,
            hints_used=hints_used,
            hints_remaining=0,
            quiz_done=True,
            quiz_score=1,
        )

    # ❌ Wrong — auto-reveal next hint
    if hints_used < len(hints):
        new_used  = hints_used + 1
        hint_text = hints[hints_used]
        remaining = len(hints) - new_used
        session["hints_used"] = new_used
        _update_session(body.session_id, session)

        reply = f"Not quite! Here's hint {new_used}: {hint_text}"
        reply += f"\n({remaining} hint{'s' if remaining > 1 else ''} left — try again)" if remaining > 0 else "\n(Last hint! One more chance)"

        return ChouchaneResponse(
            session_id=body.session_id,
            phase="quiz",
            reply=reply,
            hints_used=new_used,
            hints_remaining=remaining,
            quiz_done=False,
            quiz_score=0,
        )

    # ❌ All hints exhausted + still wrong
    reply = (
        f"The correct answer was: {session['correct_answer']}\n"
        f"Better luck next time!\n\n"
        f"You are now entering the Chouchane Q&A — ask me anything about Tunisia!"
    )
    session["quiz_score"] = 0
    session["quiz_done"]  = True
    session["phase"]      = "qa"
    _update_session(body.session_id, session)

    return ChouchaneResponse(
        session_id=body.session_id,
        phase="qa",
        reply=reply,
        hints_used=hints_used,
        hints_remaining=0,
        quiz_done=True,
        quiz_score=0,
    )


# ── Phase 3: Tunisia Q&A ─────────────────────────────────────────

@app.post("/qa", response_model=ChouchaneResponse)
def qa_chat(body: MessageRequest):
    """
    Phase 3 — Chouchane Q&A.
    Ask anything about Tunisia. Multi-turn — context is preserved across calls.
    Available after the quiz ends.
    """
    session = _get_session(body.session_id)

    if session["phase"] != "qa":
        raise HTTPException(
            status_code=400,
            detail=f"Chouchane is in phase '{session['phase']}'. Complete the quiz first."
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
        phase="qa",
        reply=reply,
    )