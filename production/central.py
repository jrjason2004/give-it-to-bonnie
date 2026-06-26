"""
Report view/generation counts to the CENTRAL portfolio Supabase project.

This is a DIFFERENT Supabase project from this app's own (supa.py) — do not reuse this app's
client/keys here. Uses a plain requests RPC call, fire-and-forget (never blocks or raises).
The anon key is publishable (safe to embed).
"""
import threading

import requests

CENTRAL_URL = "https://kvpkljeammzmzzraikzx.supabase.co"
CENTRAL_ANON = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt2cGtsa"
                "mVhbW16bXp6cmFpa3p4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODE3MTc2NjIsImV4cCI6MjA5NzI5"
                "MzY2Mn0.XjQYqClk0_aVaBm7jKhj7tzVnJ4NOKCpWtrDN0xpMRE")
CENTRAL_PROJECT_ID = "fc82f550-5d09-4a4d-a7c0-b6c156e1b34d"


def report_stat(fn):
    """Fire-and-forget RPC ('increment_project_views' / 'increment_project_generations') to the
    central portfolio project. Runs in a thread so it never blocks; swallows all errors."""
    def _go():
        try:
            requests.post(
                f"{CENTRAL_URL}/rest/v1/rpc/{fn}",
                headers={"apikey": CENTRAL_ANON, "Authorization": f"Bearer {CENTRAL_ANON}",
                         "Content-Type": "application/json"},
                json={"project_id": CENTRAL_PROJECT_ID}, timeout=5)
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()
