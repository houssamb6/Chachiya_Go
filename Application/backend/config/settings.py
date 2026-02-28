import os

# ── API ──────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAkIzekIGH9R1cCAjuEJakE_frrjYTzgts")

# ── Model ────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

# ── Paths ────────────────────────────────────────────────────────────────────
import pathlib
ROOT_DIR = pathlib.Path(__file__).parent.parent
SYSTEM_PROMPT_PATH = ROOT_DIR / "data" / "system_prompt.txt"