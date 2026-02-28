from google import genai
from google.genai import types
import re

from config.settings import GEMINI_MODEL, SYSTEM_PROMPT_PATH
from agent.extractor import extract_preferences
from agent.retriever import build_rag_context, get_recommended_place_names
from data.mock_partners import PARTNERS_DB

EXTRACTION_TURN_THRESHOLD = 5

RESUGGESTION_KEYWORDS = [
    "other", "another", "different", "else", "instead",
    "change", "suggest", "more", "option", "alternative",
    "not this", "don't like", "prefer", "something else"
]


def format_partners(place_name: str) -> str:
    """Build a clean partners block for a single chosen place."""
    partners = PARTNERS_DB.get(place_name)
    if not partners:
        return ""

    lines = ["\n" + "═" * 50]
    lines.append("WHERE TO STAY & WHERE TO EAT")
    lines.append("═" * 50)
    lines.append(f"\n{place_name}")
    lines.append("─" * 40)

    lines.append("  Hotels:")
    for h in partners["hotels"]:
        lines.append(f"     {h['name']} ({h['type']} · {h['price_range']})")
        lines.append(f"     {h['highlight']}")

    lines.append("\n  Restaurants:")
    for r in partners["restaurants"]:
        lines.append(f"     {r['name']} ({r['cuisine']} · {r['price_range']})")
        lines.append(f"     {r['highlight']}")
        lines.append(f"     Must try: {r['must_try']}")

    lines.append("\n" + "═" * 50 + "\n")
    return "\n".join(lines)


def _detect_chosen_place(user_message: str, recommended_places: list) -> str | None:
    """Check if the user's message mentions one of the recommended places."""
    msg = user_message.lower().strip()
    for place in recommended_places:
        # Match full name or first word of place name
        if place.lower() in msg or place.split()[0].lower() in msg:
            return place
    return None


def _detect_first_or_second(user_message: str) -> int | None:
    """
    If the user says "first", "second", "1", "2", "the first one", "sounds good", etc.,
    return 0 or 1 so we can set chosen_place to recommended_places[i]. Returns None otherwise.
    """
    msg = user_message.lower().strip()
    first_patterns = ("first", "1", "one", "option 1", "option one", "the first", "first one")
    second_patterns = ("second", "2", "two", "option 2", "option two", "the second", "second one")
    if any(p in msg for p in first_patterns) and not any(p in msg for p in second_patterns):
        return 0
    if any(p in msg for p in second_patterns):
        return 1
    # Generic positive confirmation → treat as "first" so we end the phase and show partners
    confirm = ("sounds good", "perfect", "great", "that's great", "lets do it", "let's do it",
               "im happy", "i'm happy", "that works", "ok", "okay", "merci", "thanks", "thank you",
               "yes", "yeah", "good", "nice", "that's all", "thats all")
    if any(c in msg for c in confirm) and len(msg) < 80:
        return 0
    return None


def _is_resuggestion_request(message: str) -> bool:
    message_lower = message.lower()
    return any(kw in message_lower for kw in RESUGGESTION_KEYWORDS)


class TunisiaTourismAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.history: list = []
        self.preferences: dict = {}
        self.recommendations_given: bool = False
        self.recommended_places: list = []
        self.chosen_place: str | None = None
        self._partners_shown: bool = False
        self._system_prompt: str = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    # ── Public interface ─────────────────────────────────────────────────────

    def start(self) -> tuple:
        return self.chat("The user just said they're ready to start. Greet them warmly as Yasmine and ask your first question.")

    def chat(self, user_message: str) -> tuple[str, str | None]:
        """
        Returns (yasmine_reply, partners_block or None).
        Partners shown only when user picks their final place.
        """
        self._push_user(user_message)

        # Check if user is picking one of the recommended places (by name or "first"/"second"/confirmation)
        if self.recommendations_given and not self.chosen_place and self.recommended_places:
            picked = _detect_chosen_place(user_message, self.recommended_places)
            if picked:
                self.chosen_place = picked
            else:
                idx = _detect_first_or_second(user_message)
                if idx is not None and 0 <= idx < len(self.recommended_places):
                    self.chosen_place = self.recommended_places[idx]

        system_prompt = self._build_system_prompt(user_message)
        reply = self._generate(system_prompt)
        self._push_model(reply)

        # Show partners only when a place has been chosen
        partners_block = None
        if self.chosen_place and not self._partners_shown:
            partners_block = format_partners(self.chosen_place)
            self._partners_shown = True

        return reply, partners_block

    def reset(self):
        self.history = []
        self.preferences = {}
        self.recommendations_given = False
        self.recommended_places = []
        self.chosen_place = None
        self._partners_shown = False

    def get_history(self) -> list:
        return self.history

    # ── Private helpers ──────────────────────────────────────────────────────

    def __init_subclass__(cls):
        pass

    def _build_system_prompt(self, user_message: str) -> str:
        user_turns = sum(1 for m in self.history if m.role == "user")

        # First time reaching threshold — extract + inject RAG
        if not self.recommendations_given and user_turns >= EXTRACTION_TURN_THRESHOLD:
            self.preferences = extract_preferences(self.client, GEMINI_MODEL, self.history)
            if self.preferences:
                rag_context = build_rag_context(self.preferences)
                self.recommended_places = get_recommended_place_names(self.preferences)
                self.recommendations_given = True
                return self._system_prompt + "\n\n" + rag_context

        # User wants different suggestions — re-extract and re-inject
        if self.recommendations_given and not self.chosen_place and _is_resuggestion_request(user_message):
            self.preferences = extract_preferences(self.client, GEMINI_MODEL, self.history)
            if self.preferences:
                rag_context = build_rag_context(self.preferences)
                self.recommended_places = get_recommended_place_names(self.preferences)
                return self._system_prompt + "\n\n" + rag_context

        # Keep RAG context active after recommendations given
        if self.recommendations_given and self.preferences:
            return self._system_prompt + "\n\n" + build_rag_context(self.preferences)

        return self._system_prompt

    def _generate(self, system_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=self.history,
            config=types.GenerateContentConfig(system_instruction=system_prompt)
        )
        return self._clean(response.text)

    def _clean(self, text: str) -> str:
        text = text.replace("*", "")
        text = re.sub(r"[\U0001F000-\U0001FFFF\U00002700-\U000027BF\U0001F900-\U0001F9FF\u2600-\u26FF]", "", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _push_user(self, text: str):
        self.history.append(types.Content(role="user", parts=[types.Part(text=text)]))

    def _push_model(self, text: str):
        self.history.append(types.Content(role="model", parts=[types.Part(text=text)]))