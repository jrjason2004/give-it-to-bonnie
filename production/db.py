"""SQLite log of every prompt + output the pipeline produces.

output/script.json (and the generated images/clips) get overwritten by each new run,
so the only durable record of what prompt produced what is this database. Lives in
output/runs.db, which persists across runs even though individual scene files don't.
"""
import sqlite3
import time

import config

DB_PATH = config.OUTPUT / "runs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    final_path TEXT,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    scene_id TEXT NOT NULL,
    kind TEXT NOT NULL,        -- image | video | audio
    model TEXT,
    prompt TEXT NOT NULL,
    output_path TEXT,
    created_at TEXT NOT NULL,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_prompts_run ON prompts(run_id);
"""


def _conn():
    config.OUTPUT.mkdir(exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.executescript(_SCHEMA)
    return c


def start_run(topic: str) -> int:
    with _conn() as c:
        cur = c.execute("INSERT INTO runs (topic, started_at) VALUES (?, ?)",
                         (topic, time.strftime("%Y-%m-%d %H:%M:%S")))
        return cur.lastrowid


def finish_run(run_id: int, final_path: str = None, status: str = "done"):
    with _conn() as c:
        c.execute("UPDATE runs SET finished_at=?, final_path=?, status=? WHERE id=?",
                  (time.strftime("%Y-%m-%d %H:%M:%S"), final_path, status, run_id))


def log_prompt(run_id: int, scene_id: str, kind: str, prompt: str,
               model: str = None, output_path: str = None, error: str = None):
    if run_id is None:
        return  # called outside a tracked run (e.g. ad-hoc script) — nothing to attach to
    with _conn() as c:
        c.execute("""INSERT INTO prompts (run_id, scene_id, kind, model, prompt, output_path, created_at, error)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                  (run_id, scene_id, kind, model, prompt, output_path,
                   time.strftime("%Y-%m-%d %H:%M:%S"), error))


def recent_runs(limit: int = 20):
    with _conn() as c:
        return c.execute("SELECT id, topic, started_at, finished_at, status, final_path "
                         "FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def prompts_for_run(run_id: int):
    with _conn() as c:
        return c.execute("SELECT scene_id, kind, model, prompt, output_path, error, created_at "
                         "FROM prompts WHERE run_id=? ORDER BY id", (run_id,)).fetchall()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        rid = int(sys.argv[1])
        for scene_id, kind, model, prompt, out, err, created in prompts_for_run(rid):
            tag = f"✗ {err[:60]}" if err else (out or "")
            print(f"[{created}] {scene_id:>10} {kind:6} {model or '':18} {prompt[:80]!r:84} {tag}")
    else:
        for rid, topic, started, finished, status, final in recent_runs():
            print(f"{rid:4}  {started}  {status:8} {topic:30} {final or ''}")
