"""
voice_id.py
Speaker enrollment and identification using resemblyzer GE2E embeddings.
A single-worker ThreadPoolExecutor serialises calls — resemblyzer is not thread-safe.
"""

from __future__ import annotations
import asyncio
import functools
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_enc_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="resemblyzer")
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder("cpu")
    return _encoder


def _embed(audio_bytes: bytes, suffix: str) -> np.ndarray:
    import subprocess
    from resemblyzer import preprocess_wav
    enc = _get_encoder()
    # Write browser audio (WebM/Opus or MP4/AAC) to a temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        src = f.name
    # Decode to 16kHz mono PCM WAV so soundfile can read it directly,
    # avoiding the PySoundFile-failed / audioread-fallback warning.
    wav_tmp = src + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", wav_tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        wav = preprocess_wav(wav_tmp)
        return enc.embed_utterance(wav)
    finally:
        os.unlink(src)
        if os.path.exists(wav_tmp):
            os.unlink(wav_tmp)


async def get_embedding(audio_bytes: bytes, suffix: str = ".webm") -> np.ndarray:
    """Extract 256-dim GE2E speaker embedding from raw audio bytes."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _enc_executor,
        functools.partial(_embed, audio_bytes, suffix),
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def identify(
    embedding: np.ndarray,
    enrollments: dict[str, np.ndarray],
    threshold: float = 0.70,
) -> tuple[str | None, dict]:
    """Return (speaker_key_or_None, similarity_scores_dict)."""
    if not enrollments:
        return None, {}
    sims = {k: cosine_sim(embedding, v) for k, v in enrollments.items()}
    best = max(sims, key=sims.get)
    return (best if sims[best] >= threshold else None), sims
