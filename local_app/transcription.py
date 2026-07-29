"""
transcription.py
Local Whisper speech-to-text for the couples biofeedback session log.
A single-worker thread pool serialises calls — Whisper is not thread-safe.
"""

from concurrent.futures import ThreadPoolExecutor
import asyncio
import functools
import os
import tempfile

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
_model = None


def _load_model(name: str):
    global _model
    if _model is None:
        import whisper  # imported lazily so the module loads fast at startup
        _model = whisper.load_model(name)
    return _model


def _transcribe_bytes(audio_bytes: bytes, model_name: str, suffix: str = ".webm") -> dict:
    model = _load_model(model_name)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        # fp16=False works on both CPU and MPS without configuration
        result = model.transcribe(tmp, fp16=False)
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language"),
        }
    finally:
        os.unlink(tmp)


async def transcribe_audio(
    audio_bytes: bytes,
    model_name: str = "base",
    content_type: str = "audio/webm",
) -> dict:
    """Transcribe raw audio bytes, returns {"text": ..., "language": ...}."""
    # Map MIME type to a file extension ffmpeg will accept
    if "mp4" in content_type or "m4a" in content_type or "aac" in content_type:
        suffix = ".mp4"
    elif "ogg" in content_type:
        suffix = ".ogg"
    else:
        suffix = ".webm"
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor,
        functools.partial(_transcribe_bytes, audio_bytes, model_name, suffix),
    )


# ── faster-whisper drop-in (optional, ~4× faster) ────────────────────────────
#
# To use faster-whisper instead of openai-whisper:
#   pip install faster-whisper
#
# Replace _load_model and _transcribe_bytes with:
#
#   from faster_whisper import WhisperModel
#
#   def _load_model(name):
#       global _model
#       if _model is None:
#           _model = WhisperModel(name, device="cpu", compute_type="int8")
#       return _model
#
#   def _transcribe_bytes(audio_bytes, model_name):
#       model = _load_model(model_name)
#       with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
#           f.write(audio_bytes); tmp = f.name
#       try:
#           segments, info = model.transcribe(tmp)
#           return {"text": " ".join(s.text.strip() for s in segments),
#                   "language": info.language}
#       finally:
#           os.unlink(tmp)
