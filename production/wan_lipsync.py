"""Wav2Lip lip-sync over the worker tunnels: POST a (silent Wan) clip + TTS audio to a
box-side wav2lip_server, get back the lip-synced clip (audio padded to clip length).
Round-robins across BONNIE_WAV2LIP_ENDPOINTS so lip-sync parallelizes across the fleet."""
import os
import uuid
import itertools
import threading
import urllib.request
import urllib.error
from pathlib import Path

import config

# Both wav2lip_server and latentsync_server expose POST /lipsync — same client, different
# endpoint set per mode. Each box's service: wav2lip on :8189, latentsync on :8190.
_ENV = {"wav2lip": "BONNIE_WAV2LIP_ENDPOINTS", "latentsync": "BONNIE_LATENTSYNC_ENDPOINTS"}
_DEFAULT = {"wav2lip": "http://localhost:9011", "latentsync": "http://localhost:9015"}
_mode = config.LIPSYNC if config.LIPSYNC in _ENV else "wav2lip"
ENDPOINTS = [e.strip() for e in os.environ.get(_ENV[_mode], _DEFAULT[_mode]).split(",") if e.strip()]
_rr = itertools.cycle(ENDPOINTS)
_lock = threading.Lock()


def _next() -> str:
    with _lock:
        return next(_rr)


def lipsync(clip: str, audio: str, out: str, endpoint: str | None = None) -> str:
    endpoint = endpoint or _next()
    boundary = "----w2l" + uuid.uuid4().hex
    body = b""
    for name, path, ctype in [("video", clip, "video/mp4"), ("audio", audio, "application/octet-stream")]:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                 f"filename=\"{Path(path).name}\"\r\nContent-Type: {ctype}\r\n\r\n").encode()
        body += Path(path).read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(endpoint + "/lipsync", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=400) as r:
            Path(out).write_bytes(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{endpoint} lipsync {e.code}: {e.read().decode()[-300:]}") from None
    return out
