import json
from google.genai import types


EXTRACTION_PROMPT = """
Based on this conversation, extract the user's travel preferences as JSON.
Return ONLY valid JSON — no explanation, no markdown, no backticks.

Use exactly these keys:
{
  "style": "adventure|beach|culture|history|nature|mix",
  "companions": "solo|couple|family|friends",
  "budget": "budget|mid-range|luxury",
  "duration_days": <integer>,
  "interests": ["list", "of", "specific", "interests"]
}

If a value is unknown, use "" for strings, 7 for duration_days, and [] for interests.
"""


def extract_preferences(client, model: str, history: list) -> dict:
    """
    Ask Gemini to extract structured preferences from conversation history.
    Returns a dict or empty dict on failure.
    """
    try:
        response = client.models.generate_content(
            model=model,
            contents=history,
            config=types.GenerateContentConfig(system_instruction=EXTRACTION_PROMPT)
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return {}