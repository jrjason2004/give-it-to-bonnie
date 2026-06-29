"""
Minimal Supabase REST + Storage helpers for the live community wall.

Stdlib + requests only (no SDK). Reads SUPABASE_URL and SUPABASE_SERVICE_KEY from env or .env.
If those aren't set, enabled() is False and landing.py falls back to the local JSON store.

Server-side writes use the service_role key (bypasses RLS), so the table only needs a public
SELECT policy. Setup SQL + bucket are created by `python3 supa.py setup`.
"""
import os
import sys
import mimetypes
from pathlib import Path
from datetime import datetime

import requests

import config

BUCKET = "wall"
TABLE = "wall_entries"
VIDEO_BUCKET = "videos"
VIDEO_TABLE = "videos"


def _creds():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    env = config.ROOT / ".env"
    if (not url or not key) and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("SUPABASE_URL=") and not url:
                url = line.split("=", 1)[1].strip()
            elif line.startswith("SUPABASE_SERVICE_KEY=") and not key:
                key = line.split("=", 1)[1].strip()
    return (url.rstrip("/") if url else None), key


def enabled():
    u, k = _creds()
    return bool(u and k)


def _h(key, extra=None):
    h = {"apikey": key, "Authorization": f"Bearer {key}"}
    if extra:
        h.update(extra)
    return h


def ensure_bucket():
    """Create the public 'wall' storage bucket if it doesn't exist (idempotent)."""
    url, key = _creds()
    try:
        requests.post(f"{url}/storage/v1/bucket", headers=_h(key, {"Content-Type": "application/json"}),
                      json={"id": BUCKET, "name": BUCKET, "public": True}, timeout=20)
    except Exception:
        pass


def ensure_video_bucket():
    """Create the public 'videos' storage bucket if it doesn't exist (idempotent)."""
    url, key = _creds()
    try:
        requests.post(f"{url}/storage/v1/bucket", headers=_h(key, {"Content-Type": "application/json"}),
                      json={"id": VIDEO_BUCKET, "name": VIDEO_BUCKET, "public": True}, timeout=20)
    except Exception:
        pass


def upload_image(local_path):
    """Upload a local image to the public bucket, return its public URL."""
    url, key = _creds()
    p = Path(local_path)
    dest = p.name
    ctype = mimetypes.guess_type(dest)[0] or "image/jpeg"
    r = requests.post(f"{url}/storage/v1/object/{BUCKET}/{dest}",
                      headers=_h(key, {"Content-Type": ctype, "x-upsert": "true"}),
                      data=p.read_bytes(), timeout=60)
    r.raise_for_status()
    return f"{url}/storage/v1/object/public/{BUCKET}/{dest}"


def insert(name, item, image_url):
    url, key = _creds()
    r = requests.post(f"{url}/rest/v1/{TABLE}",
                      headers=_h(key, {"Content-Type": "application/json", "Prefer": "return=minimal"}),
                      json={"name": name, "item": item, "image_url": image_url}, timeout=20)
    r.raise_for_status()


def fetch(limit=16):
    """Recent wall entries, newest first, as [{name, item, img, ts}]."""
    url, key = _creds()
    r = requests.get(f"{url}/rest/v1/{TABLE}"
                     f"?select=name,item,image_url,created_at&order=created_at.desc&limit={limit}",
                     headers=_h(key), timeout=20)
    r.raise_for_status()
    out = []
    for row in r.json():
        ts = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")).timestamp()
        out.append({"name": row["name"], "item": row["item"], "img": row["image_url"], "ts": ts})
    return out


def upload_video(local_path, dest_name=None):
    """Upload a local mp4 to the public 'videos' bucket, return its public URL."""
    url, key = _creds()
    p = Path(local_path)
    dest = dest_name or p.name
    r = requests.post(f"{url}/storage/v1/object/{VIDEO_BUCKET}/{dest}",
                      headers=_h(key, {"Content-Type": "video/mp4", "x-upsert": "true"}),
                      data=p.read_bytes(), timeout=300)
    r.raise_for_status()
    return f"{url}/storage/v1/object/public/{VIDEO_BUCKET}/{dest}"


def insert_video(topic, video_url, run_id=None):
    """Record a finished render in the 'videos' table so it's browsable later."""
    url, key = _creds()
    r = requests.post(f"{url}/rest/v1/{VIDEO_TABLE}",
                      headers=_h(key, {"Content-Type": "application/json", "Prefer": "return=minimal"}),
                      json={"topic": topic, "video_url": video_url, "run_id": run_id}, timeout=20)
    r.raise_for_status()


def fetch_videos(limit=50):
    """Recent rendered videos, newest first, as [{topic, video_url, run_id, created_at}]."""
    url, key = _creds()
    r = requests.get(f"{url}/rest/v1/{VIDEO_TABLE}"
                     f"?select=topic,video_url,run_id,created_at&order=created_at.desc&limit={limit}",
                     headers=_h(key), timeout=20)
    r.raise_for_status()
    return r.json()


SETUP_SQL = """\
create table if not exists wall_entries (
  id bigint generated always as identity primary key,
  name text not null,
  item text not null,
  image_url text not null,
  created_at timestamptz not null default now()
);
alter table wall_entries enable row level security;
drop policy if exists "wall public read" on wall_entries;
create policy "wall public read" on wall_entries for select using (true);

create table if not exists videos (
  id bigint generated always as identity primary key,
  topic text not null,
  video_url text not null,
  run_id bigint,
  created_at timestamptz not null default now()
);
alter table videos enable row level security;
drop policy if exists "videos public read" on videos;
create policy "videos public read" on videos for select using (true);
"""

if __name__ == "__main__":
    if not enabled():
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY not set in .env — add them first.")
        sys.exit(1)
    print("Creating storage buckets 'wall' + 'videos'…")
    ensure_bucket()
    ensure_video_bucket()
    print("Buckets ready.\n\nRun this SQL once in the Supabase SQL editor:\n")
    print(SETUP_SQL)
