"""
Audio: ElevenLabs Andy voice-change (speech-to-speech) + Andy TTS (closing voiceover),
Gemini TTS (scene-7 toy line), and keep-SFX vocal removal (Demucs, mute fallback).
"""
import os
import json
import shutil
import subprocess
from pathlib import Path

import requests

import config
import gemini

EL = "https://api.elevenlabs.io/v1"


def _el_key():
    k = os.environ.get("ELEVENLABS_API_KEY")
    if not k:
        for line in (Path(__file__).parent / ".env").read_text().splitlines():
            if line.startswith("ELEVENLABS_API_KEY="):
                k = line.split("=", 1)[1].strip()
    return k


def extract_audio(video: str, out_wav: str) -> str:
    subprocess.run(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "44100",
                    out_wav], check=True, capture_output=True)
    return out_wav


def andy_voice_change(in_audio: str, out_mp3: str) -> str:
    """ElevenLabs speech-to-speech: transform a clip's spoken line into the Andy voice."""
    a = config.ELEVEN_ANDY
    settings = {"stability": a["stability"], "similarity_boost": a["similarity_boost"],
                "style": a["style"]}
    with open(in_audio, "rb") as f:
        r = requests.post(
            f"{EL}/speech-to-speech/{a['voice_id']}",
            headers={"xi-api-key": _el_key()},
            data={"model_id": a["sts_model"], "voice_settings": json.dumps(settings),
                  "output_format": "mp3_44100_128", "remove_background_noise": "false"},
            files={"audio": f}, timeout=180)
    r.raise_for_status()
    Path(out_mp3).write_bytes(r.content)
    return out_mp3


def andy_tts(text: str, out_mp3: str, settings: dict | None = None) -> str:
    """ElevenLabs text-to-speech in the Andy voice. `settings` overrides voice_settings
    (stability/similarity_boost/style/use_speaker_boost/speed) for this one call."""
    a = config.ELEVEN_ANDY
    vs = settings or {"stability": a["stability"], "similarity_boost": a["similarity_boost"],
                      "style": a["style"]}
    r = requests.post(
        f"{EL}/text-to-speech/{a['voice_id']}",
        headers={"xi-api-key": _el_key(), "accept": "audio/mpeg", "content-type": "application/json"},
        json={"text": text, "model_id": a["model"], "voice_settings": vs},
        timeout=180)
    r.raise_for_status()
    Path(out_mp3).write_bytes(r.content)
    return out_mp3


def toy_tts(text: str, style: str, out_wav: str) -> str:
    """Gemini TTS for the scene-7 pull-string toy line."""
    return gemini.tts(config.GEMINI_TTS_MODEL, text, style, out_wav)


def bonnie_tts(text: str, out_wav: str) -> str:
    """Gemini TTS for Bonnie's lines (scenes 4 & 6) — a young, bright little-girl voice.
    Gemini TTS comes out hotter than the ElevenLabs Andy voice, so pull it down ~7 dB."""
    raw = gemini.tts(config.GEMINI_TTS_MODEL, text,
                     "Say this like an excited, cute little girl, bright and high-pitched.",
                     str(out_wav) + ".raw.wav", voice="Leda")
    subprocess.run(["ffmpeg", "-y", "-i", raw, "-af", "volume=0.45", out_wav],
                   check=True, capture_output=True)
    return out_wav


def remove_vocals(in_audio: str, out_wav: str) -> str | None:
    """Keep-SFX vocal removal via Demucs. Returns the SFX bed, or None if Demucs unavailable
    (caller then mutes, to avoid doubling the generated voice under the Andy track)."""
    if shutil.which("demucs") is None and subprocess.run(
            ["python3", "-c", "import demucs"], capture_output=True).returncode != 0:
        return None
    work = Path(out_wav).parent / "_demucs"
    env = dict(os.environ)
    try:
        import certifi
        env["SSL_CERT_FILE"] = env["REQUESTS_CA_BUNDLE"] = certifi.where()  # torch.hub model download
    except Exception:
        pass
    r = subprocess.run(["python3", "-m", "demucs", "--two-stems=vocals", "-o", str(work), in_audio],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return None  # fall back to muted SFX rather than crashing the render

    nv = next(work.rglob("no_vocals.wav"), None)
    if nv:
        shutil.copy(nv, out_wav)
        return out_wav
    return None
