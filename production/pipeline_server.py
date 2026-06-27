"""
Pipeline server — runs on EC2, accepts generation jobs from Render (landing.py),
executes pipeline.py as isolated subprocesses, uploads result to S3.

    uvicorn pipeline_server:app --host 0.0.0.0 --port 8000
"""
import json
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).parent
LOGS = Path("/opt/bonnie/logs")
LOGS.mkdir(parents=True, exist_ok=True)

SECRET = os.environ.get("BONNIE_PIPELINE_SECRET", "")
if not SECRET:
    raise RuntimeError("BONNIE_PIPELINE_SECRET must be set — refusing to start without auth")
BUCKET = os.environ.get("BONNIE_S3_BUCKET", "bonnie-video-output")
MAX_WORKERS = int(os.environ.get("BONNIE_PIPELINE_WORKERS", "2"))

app = FastAPI()
JOBS: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


class GenerateRequest(BaseModel):
    jid: str
    topic: str
    secret: str = ""


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate")
def generate(req: GenerateRequest):
    if SECRET and req.secret != SECRET:
        raise HTTPException(status_code=403, detail="forbidden")
    if req.jid in JOBS and JOBS[req.jid].get("status") in ("queued", "running", "done"):
        return {"queued": True, "existing": True}
    JOBS[req.jid] = {"status": "queued", "topic": req.topic}
    _executor.submit(_run_job, req.jid, req.topic)
    return {"queued": True}


@app.get("/status/{jid}")
def status(jid: str):
    return JOBS.get(jid, {"status": "not_found"})


@app.get("/logs/{jid}")
def logs(jid: str, lines: int = 50):
    return {"log": _tail(LOGS / f"{jid}.log", lines)}


def _run_job(jid: str, topic: str):
    out_dir = Path(f"/opt/bonnie/output/{jid}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{jid}.log"
    JOBS[jid]["status"] = "running"
    JOBS[jid]["started"] = time.time()

    env = {**os.environ, "BONNIE_OUTPUT": str(out_dir), "BONNIE_S3_BUCKET": BUCKET}

    try:
        with open(log_path, "w") as log_f:
            result = subprocess.run(
                ["python3", "pipeline.py", topic],
                cwd=str(ROOT),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
        if result.returncode != 0:
            tail = _tail(log_path, 20)
            raise RuntimeError(f"exit {result.returncode}: {tail}")

        # pipeline.py prints the S3 URL (or local path) as its last output line
        output = log_path.read_text().strip().splitlines()
        # find the URL/path returned by pipeline.run() — logged as "=== DONE -> <path> ==="
        video_url = None
        for line in reversed(output):
            if "=== DONE ->" in line:
                video_url = line.split("=== DONE ->")[-1].strip().rstrip("=").strip()
                break
        if not video_url:
            raise RuntimeError("could not find output URL in pipeline log")

        JOBS[jid] = {"status": "done", "video_url": video_url, "topic": topic}
    except Exception as e:
        JOBS[jid] = {"status": "error", "error": str(e)[-500:], "topic": topic}


def _tail(path: Path, n: int) -> str:
    try:
        lines = path.read_text().splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""
