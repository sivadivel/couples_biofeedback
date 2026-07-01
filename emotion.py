import json
import os

from anthropic import AsyncAnthropic

TONES = ("aggressive", "neutral", "kind")

SYSTEM = (
    "You are a couples therapist's assistant. "
    "Classify the emotional tone of a single therapy session utterance into one of three tones:\n"
    "- aggressive: hostility, criticism, contempt, sarcasm, passive aggression, dismissiveness, rudeness, stonewalling\n"
    "- kind: warmth, affection, empathy, vulnerability, repair attempts, genuine support, appreciation\n"
    "- neutral: factual statements, questions, or observations with no strong emotional charge\n"
    "Output ONLY valid JSON, no markdown:\n"
    '{"tone":"<aggressive|neutral|kind>","confidence":<0.0-1.0>,"note":"<3-8 word description>"}\n'
    'Default to "neutral" when the utterance is ambiguous or short.'
)

_FALLBACK = {"tone": "neutral", "confidence": 0.0, "note": ""}

_client: "AsyncAnthropic | None" = None


def _get_client() -> "AsyncAnthropic | None":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=api_key)
    return _client


async def classify_emotion(text: str) -> dict:
    """Return {"tone": aggressive|neutral|kind, "confidence": float, "note": str}."""
    client = _get_client()
    if not client or not text.strip():
        return {**_FALLBACK, "confidence": 1.0}
    try:
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            system=SYSTEM,
            messages=[{"role": "user", "content": f'Utterance: "{text}"'}],
        )
        raw = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), "")
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        if not raw:
            return {**_FALLBACK}
        result = json.loads(raw)
        if result.get("tone") not in TONES:
            result["tone"] = "neutral"
        result.setdefault("confidence", 0.5)
        result.setdefault("note", "")
        return result
    except Exception as exc:
        print(f"[emotion] error: {exc}")
        return {**_FALLBACK}
