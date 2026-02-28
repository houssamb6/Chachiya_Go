from data.places_db import PLACES_DB


def retrieve_top_places(preferences: dict, top_n: int = 2) -> list:
    """
    Score every place in PLACES_DB against user preferences.
    Returns the top_n best-matching place dicts.
    """
    style      = preferences.get("style", "").lower()
    companions = preferences.get("companions", "").lower()
    budget     = preferences.get("budget", "").lower()
    interests  = [i.lower() for i in preferences.get("interests", [])]
    duration   = preferences.get("duration_days", 7)

    scores = {}

    for name, place in PLACES_DB.items():
        score = 0

        for s in place["styles"]:
            if style and s in style:
                score += 3

        for b in place["best_for"]:
            if companions and b in companions:
                score += 2

        for b in place["budget"]:
            if budget and b in budget:
                score += 2

        for interest in interests:
            if interest in place["interests"]:
                score += 4

        if duration <= 3 and place["duration_days"] <= 1:
            score += 2
        elif duration >= 7:
            score += 1

        scores[name] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [PLACES_DB[name] | {"name": name} for name, _ in ranked[:top_n]]


def build_rag_context(preferences: dict) -> str:
    """Build the RAG context string — place profiles only."""
    places = retrieve_top_places(preferences)
    lines = ["RETRIEVED PLACE PROFILES (recommend ONLY these 2 places):\n"]

    for p in places:
        lines.append(f"""
PLACE: {p['name']} — {p['region']}
Vibe: {p['vibe']}
Description: {p['description']}
Top Activities: {', '.join(p['top_activities'])}
Insider Tip: {p['insider_tip']}
Best Season: {p['season']}
Tunisian Word: {p['tunisian_word']}
{"─" * 60}""")

    return "\n".join(lines)


def get_recommended_place_names(preferences: dict) -> list[str]:
    """Return just the names of the top 2 recommended places."""
    return [p["name"] for p in retrieve_top_places(preferences)]