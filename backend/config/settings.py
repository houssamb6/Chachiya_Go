import os

# ── API ──────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAd1jCklb9OwJUr2gFA7d3BXnqcodMIMKI")

# ── Model ────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

# ── Paths ────────────────────────────────────────────────────────────────────
import pathlib
ROOT_DIR = pathlib.Path(__file__).parent.parent
SYSTEM_PROMPT_PATH = ROOT_DIR / "data" / "system_prompt.txt"