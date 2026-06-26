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
  `STRIPE_SECRET_KEY`, `STRIPE_PUBLISHABLE_KEY`, `BASE_URL` (the public https URL).

Community wall persists to Supabase (`supa.py`); payments via Stripe Checkout + inline
Apple Pay / Google Pay (`stripe_pay.py`).
