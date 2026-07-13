# Give It To Bonnie

A Toy Story parody: type something you're ready to let go of, and Bonnie sends back a photo of
herself holding it plus a handwritten letter. Pay $5 to watch Andy drop it off.

The product is the web app in [`production/`](production/) (`landing.py`).

## Run locally

```bash
cd production
pip install -r requirements.txt   # also needs ffmpeg on PATH
python landing.py                 # http://localhost:8095
```

## Deploy (Render, Docker)

- Runtime: **Docker** (installs ffmpeg), Root Directory: `production`.
- Start command: handled by the Dockerfile (`python landing.py`); the app reads `$PORT` and binds `0.0.0.0`.
- Set these environment variables in the host dashboard (not committed — see `.env`):
  `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`,
  `STRIPE_SECRET_KEY`, `STRIPE_PUBLISHABLE_KEY`, `BASE_URL` (the public https URL),
  `FLEET_AWS_ACCESS_KEY_ID` / `FLEET_AWS_SECRET_ACCESS_KEY` (IAM user `bonnie-fleet`:
  discovers + starts the on-demand Wan GPU fleet by EC2 tag `Fleet=wan`).
- Optional: `BONNIE_WAN_TOKEN` routes Wan workers through their :8443 token proxy
  (header `X-Wan-Token`) instead of open :8188. Do NOT set `BONNIE_WAN_ENDPOINTS` in
  prod — it pins worker IPs (which change on every fleet stop/start) and disables
  discovery; it exists only for local SSM tunnels in dev.

Community wall persists to Supabase (`supa.py`); payments via Stripe Checkout + inline
Apple Pay / Google Pay (`stripe_pay.py`).
