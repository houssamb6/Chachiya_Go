"""
main_with_quiz.py
=================
Full LangGraph workflow:

  Phase 1 — YASMINE      : RAG + Gemini tourism recommendation agent
  Phase 2 — QUIZ         : 1 question about the chosen destination (auto-hints on wrong answer)
  Phase 3 — TUNISIA Q&A  : Gemini-powered assistant for anything about Tunisia

Flow:
  Yasmine chats
    → user picks a place → partners shown
    → Quiz starts (1 question, auto-hints, ends after 3 wrong or correct)
    → Q&A mode: "Ask me anything about Tunisia!"
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from google import genai
from google.genai import types
import re

from agent.conversation import TunisiaTourismAgent
from config.settings import API_KEY, GEMINI_MODEL
from config.content import WELCOMING
from data.quiz_db import QUIZ_DATA

data = QUIZ_DATA

# ─────────────────────────────────────────────────────────────────
# TUNISIA Q&A SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """
You are a knowledgeable and enthusiastic Tunisia travel expert.
You have deep knowledge of Tunisia's history, culture, food, geography,
tourism destinations, traditions, language, and people.

Answer every question in a warm, informative, and engaging way.
Keep answers concise but rich — 2 to 4 sentences unless more detail is needed.
If the question is not related to Tunisia, politely redirect:
"That's a bit outside my expertise — I'm your Tunisia specialist! Ask me anything about this amazing country."

Do not use markdown, asterisks, or emojis.
""".strip()

# ─────────────────────────────────────────────────────────────────
# HELPER — find quiz question for a chosen place
# ─────────────────────────────────────────────────────────────────

def get_quiz_for_place(place_name: str) -> Optional[dict]:
    place_lower = place_name.lower()
    for q in data:
        if q["destination"].lower() in place_lower or place_lower in q["destination"].lower():
            return q
    first_word = place_lower.split()[0]
    for q in data:
        if first_word in q["destination"].lower():
            return q
    return None

# ─────────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────────

class AppState(TypedDict):
    # Phase control
    phase: str                  # "yasmine" | "quiz" | "qa" | "done"

    # Yasmine
    yasmine_reply: str
    partners_block: Optional[str]
    chosen_place: Optional[str]

    # Quiz
    destination: str
    question: str
    hints: List[str]
    correct_answer: str
    hints_used: int
    user_input: str
    waiting_for: str
    score: int
    quiz_message: str
    game_over: bool

    # Q&A
    qa_history: List[dict]      # [{role, text}] for Gemini multi-turn
    qa_reply: str

# ─────────────────────────────────────────────────────────────────
# NODE 1 — YASMINE
# ─────────────────────────────────────────────────────────────────

_yasmine_agent: Optional[TunisiaTourismAgent] = None

def yasmine_node(state: AppState) -> AppState:
    global _yasmine_agent
    reply, partners = _yasmine_agent.chat(state["user_input"])
    chosen = _yasmine_agent.chosen_place

    return {
        **state,
        "yasmine_reply":  reply,
        "partners_block": partners,
        "chosen_place":   chosen,
        "phase": "quiz" if partners else "yasmine",
    }

# ─────────────────────────────────────────────────────────────────
# NODE 2 — RAG LOADER
# ─────────────────────────────────────────────────────────────────

def rag_loader_node(state: AppState) -> AppState:
    chosen_place = state.get("chosen_place", "")
    q = get_quiz_for_place(chosen_place)

    if not q:
        return {
            **state,
            "game_over":    True,
            "quiz_message": f"\n  Sorry, no quiz question found for '{chosen_place}'.\n",
        }

    msg = (
        f"\n{'═' * 55}\n"
        f"  QUIZ TIME!  |  {q['destination']}\n"
        f"{'═' * 55}\n\n"
        f"  {q['question']}\n\n"
        f"  Type your answer, or type 'hint' for a clue.\n"
        f"  (You have {len(q['hints'])} hint(s) available)\n"
    )

    return {
        **state,
        "destination":    q["destination"],
        "question":       q["question"],
        "hints":          q["hints"],
        "correct_answer": q["answer"],
        "hints_used":     0,
        "waiting_for":    "hint_or_answer",
        "quiz_message":   msg,
        "game_over":      False,
    }

# ─────────────────────────────────────────────────────────────────
# NODE 3 — QUIZ AGENT
# ─────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return text.strip().lower().replace("'", "").replace("-", " ")

def quiz_agent_node(state: AppState) -> AppState:
    user_input = state["user_input"].strip()
    hints      = state["hints"]
    hints_used = state["hints_used"]

    # ── Manual hint request ───────────────────────────────────────
    if _normalize(user_input) == "hint":
        if hints_used < len(hints):
            new_used  = hints_used + 1
            hint_text = hints[hints_used]
            remaining = len(hints) - new_used
            msg = f"\n  Hint {new_used}: {hint_text}\n"
            if remaining > 0:
                msg += f"  ({remaining} hint{'s' if remaining > 1 else ''} left — try again or type 'hint')\n"
            else:
                msg += "  (Last hint! Give it your best shot)\n"
            return {**state, "hints_used": new_used, "waiting_for": "hint_or_answer", "quiz_message": msg}
        else:
            return {**state, "quiz_message": "\n  No more hints available! Give your best answer.\n"}

    # ── Evaluate answer ───────────────────────────────────────────
    is_correct = _normalize(user_input) == _normalize(state["correct_answer"])

    # ✅ Correct
    if is_correct:
        stars = "★" * max(1, 3 - hints_used)
        msg = (
            f"\n  Correct!  {stars}\n"
            f"  {'No hints used — perfect!' if hints_used == 0 else f'{hints_used} hint(s) used'}\n"
        )
        return {
            **state,
            "score":        1,
            "quiz_message": msg,
            "waiting_for":  "done",
            "game_over":    True,
        }

    # ❌ Wrong — auto-reveal next hint if available
    if hints_used < len(hints):
        new_used  = hints_used + 1
        hint_text = hints[hints_used]
        remaining = len(hints) - new_used
        msg = f"\n  Not quite! Here's hint {new_used}: {hint_text}\n"
        if remaining > 0:
            msg += f"  ({remaining} hint{'s' if remaining > 1 else ''} left — try again or type 'hint')\n"
        else:
            msg += "  (Last hint! One more chance)\n"
        return {
            **state,
            "hints_used":   new_used,
            "waiting_for":  "hint_or_answer",
            "quiz_message": msg,
            "game_over":    False,
        }

    # ❌ All hints exhausted and still wrong → end quiz
    msg = (
        f"\n  The correct answer was: {state['correct_answer']}\n"
        f"  Better luck next time!\n"
    )
    return {
        **state,
        "score":        0,
        "quiz_message": msg,
        "waiting_for":  "done",
        "game_over":    True,
    }

# ─────────────────────────────────────────────────────────────────
# NODE 4 — QUIZ SUMMARY
# ─────────────────────────────────────────────────────────────────

def summary_node(state: AppState) -> AppState:
    score      = state["score"]
    hints_used = state["hints_used"]

    if score == 1:
        if hints_used == 0:
            verdict = "Impressive! You knew it without any help — you're a true Tunisia expert!"
        elif hints_used == 1:
            verdict = "Well done! Just one hint needed — not bad at all!"
        else:
            verdict = f"You got it with {hints_used} hints. Practice makes perfect!"
    else:
        verdict = "Better luck next time! A visit to Tunisia will teach you everything."

    msg = (
        f"\n{'═' * 55}\n"
        f"  QUIZ RESULT  |  {state['destination']}\n"
        f"{'═' * 55}\n\n"
        f"  {verdict}\n\n"
        f"  Enjoy your trip to {state['chosen_place']}!\n"
        f"{'═' * 55}\n"
    )

    return {**state, "quiz_message": msg, "phase": "qa"}

# ─────────────────────────────────────────────────────────────────
# NODE 5 — TUNISIA Q&A  (Gemini-powered, multi-turn)
# ─────────────────────────────────────────────────────────────────

_gemini_client: Optional[genai.Client] = None

def _clean(text: str) -> str:
    text = text.replace("*", "")
    text = re.sub(r"[\U0001F000-\U0001FFFF\U00002700-\U000027BF\U0001F900-\U0001F9FF\u2600-\u26FF]", "", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def qa_node(state: AppState) -> AppState:
    global _gemini_client

    user_input  = state["user_input"].strip()
    qa_history  = state.get("qa_history", [])

    # Build Gemini history from our simple list
    contents = []
    for turn in qa_history:
        contents.append(
            types.Content(role=turn["role"], parts=[types.Part(text=turn["text"])])
        )
    # Add current user message
    contents.append(
        types.Content(role="user", parts=[types.Part(text=user_input)])
    )

    response = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=QA_SYSTEM_PROMPT),
    )
    reply = _clean(response.text)

    # Append both turns to history for next round
    new_history = qa_history + [
        {"role": "user",  "text": user_input},
        {"role": "model", "text": reply},
    ]

    return {
        **state,
        "qa_reply":   reply,
        "qa_history": new_history,
        "phase":      "qa",
    }

# ─────────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────────

def route_after_yasmine(state: AppState) -> str:
    return "rag_loader" if state["phase"] == "quiz" else END

def route_after_rag(state: AppState) -> str:
    return "summary" if state["game_over"] else END

def route_after_quiz(state: AppState) -> str:
    return "summary" if state["game_over"] else END

# ─────────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AppState)

    builder.add_node("yasmine",    yasmine_node)
    builder.add_node("rag_loader", rag_loader_node)
    builder.add_node("quiz_agent", quiz_agent_node)
    builder.add_node("summary",    summary_node)
    builder.add_node("qa",         qa_node)

    builder.set_entry_point("yasmine")

    builder.add_conditional_edges("yasmine", route_after_yasmine, {
        "rag_loader": "rag_loader",
        END: END,
    })
    builder.add_conditional_edges("rag_loader", route_after_rag, {
        "summary": "summary",
        END: END,
    })
    builder.add_conditional_edges("quiz_agent", route_after_quiz, {
        "summary": "summary",
        END: END,
    })

    # Summary always transitions to Q&A
    builder.add_edge("summary", "qa")
    builder.add_edge("qa", END)

    return builder.compile()

# ─────────────────────────────────────────────────────────────────
# INITIAL STATE
# ─────────────────────────────────────────────────────────────────

def create_initial_state(user_message: str) -> AppState:
    return AppState(
        phase="yasmine",
        yasmine_reply="",
        partners_block=None,
        chosen_place=None,
        destination="",
        question="",
        hints=[],
        correct_answer="",
        hints_used=0,
        user_input=user_message,
        waiting_for="hint_or_answer",
        score=0,
        quiz_message="",
        game_over=False,
        qa_history=[],
        qa_reply="",
    )

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    global _yasmine_agent, _gemini_client

    print(WELCOMING)
    print()
    input("[ Press Enter to start your journey ] ")
    print()

    _yasmine_agent = TunisiaTourismAgent(api_key=API_KEY)
    _gemini_client = genai.Client(api_key=API_KEY)
    graph = build_graph()

    reply, _ = _yasmine_agent.start()
    print(f"\nYasmine: {reply}\n")

    state   = None
    phase   = "yasmine"     # local tracker: "yasmine" | "quiz" | "qa"

    while True:
        try:
            if phase == "yasmine":
                prompt = "You: "
            elif phase == "quiz":
                prompt = "Your answer: "
            else:
                prompt = "Your question: "

            user_input = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            _yasmine_agent.reset()
            state = None
            phase = "yasmine"
            print(f"\n[Reset]\n\n{WELCOMING}\n")
            input("[ Press Enter to start your journey ] ")
            print()
            reply, _ = _yasmine_agent.start()
            print(f"\nYasmine: {reply}\n")
            continue

        # ── PHASE: Yasmine ────────────────────────────────────────
        if phase == "yasmine":
            state = create_initial_state(user_input)
            state = graph.invoke(state)

            print(f"\nYasmine: {state['yasmine_reply']}\n")

            if state["partners_block"]:
                print(state["partners_block"])

            if state["phase"] == "quiz":
                phase = "quiz"
                chosen = state["chosen_place"]
                print(f"\nGreat choice! Before you go, here's a quick quiz about {chosen}.")
                print("Type your answer, or 'hint' for a clue!\n")

                state = rag_loader_node(state)
                print(state["quiz_message"])

                # No question found — skip straight to Q&A
                if state["game_over"]:
                    state = summary_node(state)
                    print(state["quiz_message"])
                    print("\nTunisia Expert: I'm now your personal Tunisia guide — ask me anything!\n")
                    phase = "qa"

        # ── PHASE: Quiz ───────────────────────────────────────────
        elif phase == "quiz":
            state = {**state, "user_input": user_input}
            state = quiz_agent_node(state)
            print(state["quiz_message"])

            if state["game_over"]:
                # Show summary then boot Q&A
                state = summary_node(state)
                print(state["quiz_message"])
                print("\nTunisia Expert: Quiz done! I'm now your personal Tunisia guide.")
                print("Ask me anything about Tunisia — history, food, places, culture!\n")
                phase = "qa"

        # ── PHASE: Tunisia Q&A ────────────────────────────────────
        elif phase == "qa":
            state = {**state, "user_input": user_input}
            state = qa_node(state)
            print(f"\nTunisia Expert: {state['qa_reply']}\n")


if __name__ == "__main__":
    main()