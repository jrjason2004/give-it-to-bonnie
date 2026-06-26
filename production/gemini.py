"""Thin Gemini REST helpers (stdlib only) — text, image (Nano Banana 2), TTS."""
import os
import ssl
import json
import time
import socket
import base64
import urllib.request
from pathlib import Path

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl.create_default_context()

API = "https://generativelanguage.googleapis.com/v1beta"


def _key():
    k = os.environ.get("GEMINI_API_KEY")
    if not k:
        env = Path(__file__).parent / ".env"
        for line in env.read_text().splitlines() if env.exists() else []:
            if line.startswith("GEMINI_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise RuntimeError("GEMINI_API_KEY not set")
    return k


def _post(model: str, body: dict, method: str = "generateContent", timeout=300) -> dict:
    url = f"{API}/models/{model}:{method}?key={_key()}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Gemini {e.code}: {e.read().decode()[:600]}") from None
        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            last = e
            time.sleep(2 * (attempt + 1))  # transient network/timeout — retry with backoff
    raise RuntimeError(f"Gemini request failed after retries: {last}")


def generate_json(model: str, prompt: str, schema: dict, system: str = "", thinking=True) -> dict:
    """Text generation constrained to a JSON schema (the script brain). thinking=False sets a low
    thinking level for a much faster response (used for short, simple generations like the letter)."""
    gen = {"responseMimeType": "application/json", "responseSchema": schema}
    if not thinking:
        gen["thinkingConfig"] = {"thinkingLevel": "low"}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    resp = _post(model, body)
    parts = resp["candidates"][0]["content"]["parts"]
    text = "".join(p.get("text", "") for p in parts)
    return json.loads(text)


def generate_image(model: str, prompt: str, input_paths: list[str], out_path: str,
                   grounding=True, thinking_high=True, image_size: str | None = None,
                   aspect: str = "16:9") -> str:
    """Nano Banana 2 image edit/gen. Feeds reference images + prompt, saves the result.
    image_size (e.g. "512", "1K", "2K") lowers resolution for a faster render when set.
    aspect sets the output aspect ratio (e.g. "16:9", "1:1")."""
    parts = []
    for p in input_paths:
        data = base64.standard_b64encode(Path(p).read_bytes()).decode()
        mime = "image/png" if str(p).lower().endswith(".png") else "image/jpeg"
        parts.append({"inlineData": {"mimeType": mime, "data": data}})
    parts.append({"text": prompt})
    img_cfg = {"aspectRatio": aspect}
    if image_size:
        img_cfg["imageSize"] = image_size
    gc = {"responseModalities": ["IMAGE"], "imageConfig": img_cfg}
    if thinking_high:
        gc["thinkingConfig"] = {"thinkingLevel": "high"}
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gc}
    if grounding:
        body["tools"] = [{"googleSearch": {}}]  # image-search grounding is built into Nano Banana 2
    last = ""
    for _ in range(3):
        resp = _post(model, body)
        cands = resp.get("candidates") or []
        content = (cands[0].get("content") if cands else None) or {}
        for part in content.get("parts") or []:
            if "inlineData" in part:
                Path(out_path).write_bytes(base64.standard_b64decode(part["inlineData"]["data"]))
                return out_path
        last = (cands[0].get("finishReason") if cands else None) or json.dumps(resp)[:200]
        time.sleep(3)  # empty candidate — retry
    raise RuntimeError(f"no image after 3 tries (last: {last})")


def tts(model: str, text: str, style: str, out_wav: str, voice="Charon") -> str:
    """Gemini TTS -> wav. `style` is natural-language delivery instruction; [tags] also work in text."""
    prompt = (style + "\n\n" + text) if style else text
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    resp = _post(model, body)
    for part in resp["candidates"][0]["content"]["parts"]:
        if "inlineData" in part:
            pcm = base64.standard_b64decode(part["inlineData"]["data"])
            _write_wav(out_wav, pcm)
            return out_wav
    raise RuntimeError(f"no audio in response: {json.dumps(resp)[:400]}")


def _write_wav(path: str, pcm: bytes, rate=24000):
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
