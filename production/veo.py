"""Google Veo 3.1 video generation (image-to-video with native audio/dialogue) via the
Gemini API predictLongRunning op. Used as a quality benchmark vs the self-hosted Wan pipeline."""
import json
import time
import base64
import urllib.request
import urllib.error
from pathlib import Path

import gemini  # reuse _key() + _SSL

API = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "veo-3.1-lite-generate-preview"


def _find_video(obj):
    """Recursively locate a {'uri':...} or {'bytesBase64Encoded':...} video payload."""
    if isinstance(obj, dict):
        if "uri" in obj and ("video" in str(obj).lower() or obj.get("uri", "").startswith("http")):
            return obj
        if "bytesBase64Encoded" in obj:
            return obj
        for v in obj.values():
            r = _find_video(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_video(v)
            if r:
                return r
    return None


def generate_video(prompt: str, image_path: str | None, out_path: str,
                   model: str = MODEL, aspect: str = "16:9", timeout_s: int = 900,
                   dur: float | None = None) -> str:
    key = gemini._key()
    inst = {"prompt": prompt}
    if image_path:
        b = base64.standard_b64encode(Path(image_path).read_bytes()).decode()
        mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
        inst["image"] = {"bytesBase64Encoded": b, "mimeType": mime}
    params = {"aspectRatio": aspect}
    if dur:
        # Veo 3.1 Lite only accepts discrete values: 4, 6, or 8 seconds
        params["durationSeconds"] = next((v for v in (4, 6, 8) if v >= dur), 8)
    body = {"instances": [inst], "parameters": params}
    req = urllib.request.Request(f"{API}/models/{model}:predictLongRunning?key={key}",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        op = json.loads(urllib.request.urlopen(req, context=gemini._SSL, timeout=120).read())["name"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"veo start {e.code}: {e.read().decode()[:500]}") from None
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = json.loads(urllib.request.urlopen(f"{API}/{op}?key={key}", context=gemini._SSL, timeout=60).read())
        if r.get("done"):
            if r.get("error"):
                raise RuntimeError(f"veo op error: {json.dumps(r['error'])[:400]}")
            vid = _find_video(r.get("response", {}))
            if not vid:
                raise RuntimeError("veo: no video in response: " + json.dumps(r)[:400])
            if vid.get("uri"):
                u = vid["uri"] + ("&" if "?" in vid["uri"] else "?") + f"key={key}"
                data = urllib.request.urlopen(u, context=gemini._SSL, timeout=600).read()
            else:
                data = base64.standard_b64decode(vid["bytesBase64Encoded"])
            Path(out_path).write_bytes(data)
            return out_path
        time.sleep(4)   # poll faster so a finished clip is detected sooner (was 10s)
    raise TimeoutError(f"veo op timed out: {op}")
