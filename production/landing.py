"""
Give It To Bonnie — product landing (Claude Design UI).

New model:
  FREE   : type a topic -> a photo of Bonnie holding the toy (the action figure + a couple of the
           topic items), cycling through 3 Bonnie pose references, PLUS a handwritten thank-you
           letter from Bonnie ("omg thank you for the ___, here's what I'm using it for…").
  $5     : "Watch Andy drop it off" — the full video. The upsell page shows ONE preview clip
           (the saved most-recent full render) so people see what they're buying.

    python3 landing.py          # http://localhost:8095
"""
import re
import json
import time
import uuid
import threading
import itertools
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import os

import config
import gemini
import intro as intro_mod
import supa
import stripe_pay
import central
import pipeline
import video_gen

# ── GPU fleet (4× g6e.2xlarge, ComfyUI on :8188 reached via SSM tunnels) ───
_FLEET_IDS = [
    "i-0c661acc2678409f9",
    "i-059db1ff762123998",
    "i-01a5b91df29eda80d",
    "i-022bb26f2939e404c",
]
_FLEET_PORTS = [9001, 9002, 9003, 9004]   # local SSM tunnel -> box:8188
_FLEET_REGION = "us-east-1"
_fleet_lock = threading.Lock()
# If BONNIE_WAN_ENDPOINTS is already set (e.g. Render env var), skip auto-start entirely
_fleet_ready = bool(os.environ.get("BONNIE_WAN_ENDPOINTS"))
_fleet_procs = []   # SSM tunnel subprocesses


def _aws(*args):
    return subprocess.run(
        ["aws", "--region", _FLEET_REGION] + list(args),
        capture_output=True, text=True
    )


def _comfy_ok(port):
    try:
        urllib.request.urlopen(f"http://localhost:{port}/system_stats", timeout=3)
        return True
    except Exception:
        return False


def _ensure_fleet():
    """Start the GPU fleet if not already running, open SSM tunnels, update video_gen.ENDPOINTS."""
    global _fleet_ready
    with _fleet_lock:
        if _fleet_ready:
            return
        print("[fleet] starting instances…", flush=True)
        _aws("ec2", "start-instances", "--instance-ids", *_FLEET_IDS)

        # wait up to 3 min for all instances to reach running
        for _ in range(36):
            r = _aws("ec2", "describe-instances", "--instance-ids", *_FLEET_IDS,
                     "--query", "Reservations[*].Instances[*].State.Name", "--output", "text")
            states = r.stdout.strip().split()
            print(f"[fleet] states: {states}", flush=True)
            if len(states) == len(_FLEET_IDS) and all(s == "running" for s in states):
                break
            time.sleep(5)

        # open one SSM port-forward tunnel per box
        print("[fleet] opening SSM tunnels…", flush=True)
        for iid, port in zip(_FLEET_IDS, _FLEET_PORTS):
            p = subprocess.Popen(
                ["aws", "ssm", "start-session",
                 "--target", iid,
                 "--document-name", "AWS-StartPortForwardingSession",
                 "--parameters", f'{{"portNumber":["8188"],"localPortNumber":["{port}"]}}',
                 "--region", _FLEET_REGION],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            _fleet_procs.append(p)

        # wait up to 5 min for ComfyUI to respond on each tunnel
        ready_eps = []
        for _ in range(60):
            ready_eps = [f"http://localhost:{p}" for p in _FLEET_PORTS if _comfy_ok(p)]
            print(f"[fleet] ComfyUI ready: {len(ready_eps)}/{len(_FLEET_PORTS)}", flush=True)
            if len(ready_eps) == len(_FLEET_PORTS):
                break
            time.sleep(5)

        if not ready_eps:
            raise RuntimeError("fleet: no ComfyUI workers came up")

        video_gen.ENDPOINTS = ready_eps
        video_gen._rr = itertools.cycle(video_gen.ENDPOINTS)
        print(f"[fleet] ready — {len(ready_eps)} workers: {ready_eps}", flush=True)
        _fleet_ready = True

ROOT = config.ROOT
OUT = config.OUTPUT
OUT.mkdir(parents=True, exist_ok=True)     # gitignored, so it won't exist in a fresh container
PORT = int(os.environ.get("PORT", 8095))   # Render injects $PORT
HOST = os.environ.get("HOST", "127.0.0.1")  # set HOST=0.0.0.0 in prod (done in Dockerfile)
# public base for Stripe success/cancel redirects (set BASE_URL in prod to the real https domain)
BASE_URL = os.environ.get("BASE_URL", f"http://localhost:{PORT}")
LETTER_MODEL = "gemini-3.1-flash-lite"          # fast (~2s) so the letter types out near-instantly
PREVIEW = "assets/preview_dropoff.mp4"          # saved full render shown on the upsell page
# 3 Bonnie pose references, rotated per request. Each holds different toys, so each has its own
# swap prompt ({p} = the topic pile). Hug cradles one toy; porch + grass-pile hold several.
POSE_REFS = [
    ("bonnie_hug.jpg",   "Replace the cowboy toy with {p}."),
    ("bonnie_pose1.jpg", "Remove all the toys, and add {p}."),   # porch
    ("bonnie_pose2.jpg", "Remove all the toys, and add {p}."),   # grass pile
]
_pose_i = 0
_pose_lock = threading.Lock()
JOBS = {}
_lock = threading.Lock()

_FREE_SCHEMA = {"type": "object", "required": ["pile", "letter"], "properties": {
    "pile": {"type": "string", "description": "what Bonnie is cradling in the photo — one or two of the "
             "topic's real items, named specifically with brands, optionally plus an action figure of a "
             "REAL public figure central to the topic (e.g. 'a Lamar Jackson action figure, a Speed Stacks "
             "cup, and a Stackmat timer'). HARD RULE: NEVER include any Disney- or Pixar-owned or other "
             "copyrighted character, toy, or franchise — no Buzz Lightyear, Woody, Jessie, Toy Story, "
             "Mickey, Marvel, Star Wars, Pokémon, Nintendo characters, etc. They get the image rejected. "
             "If the topic's central character is fictional/copyrighted, DROP the action figure entirely "
             "and use only real brand-name objects (e.g. for 'my childhood': 'a vintage 1994 Fruit of the "
             "Loom tee shirt, and a Sharpie marker')."},
    "letter": {"type": "string", "description": "The COMPLETE handwritten letter from Bonnie, with real "
               "line breaks (\\n). Format EXACTLY like the examples:\n"
               "Line 1: 'Dear friend,'\n"
               "Then a few SHORT, simple sentences in a real little kid's voice. She got the thing the "
               "grown-up gave her (the topic). She fixates on ONE oddly specific detail and loves it. She "
               "doesn't fully understand what it is, but she loves it completely and promises to take good "
               "care of it. Somewhere she gently, innocently reassures them it's okay they let it go — and "
               "it comes out accidentally profound, never preachy.\n"
               "Then 'Love,' on its own line, then 'Bonnie 🌟' on its own line.\n"
               "Last line: a '(P.S. ...)' that is specific, slightly absurd, and lands the emotion.\n"
               "Keep it SMALL and simple — short sentences, plain kid words, like the examples. The UI "
               "renders this whole string verbatim, so include the greeting, signature, and P.S."},
    "teaser": {"type": "string", "description": "A SUPER short paywall sub-header (4-8 words) teasing the "
               "$5 video, tied to this specific item/emotion — e.g. for 'my Xbox': 'See Gerald find his "
               "new home.'; for 'checking my ex's Instagram': 'Watch it finally leave your hands.' No "
               "quotes, sentence case."}}}


def _next_pose():
    """Rotate to the next available (ref_file, prompt_template). None if no refs exist."""
    global _pose_i
    avail = [r for r in POSE_REFS if (config.ASSETS / r[0]).exists()]
    if not avail:
        return None
    with _pose_lock:
        ref = avail[_pose_i % len(avail)]; _pose_i += 1
    return ref


def _format_letter(t):
    """Lay the letter out no matter how the model spaced it: greeting, then each sentence on its own
    line, then 'Love,' / 'Bonnie 🌟' / '(P.S. …)' each as their own block."""
    t = re.sub(r"\s+", " ", t).strip()
    # peel off the P.S. (keep it whole — never sentence-split it)
    ps = ""
    m = re.search(r"\(?\s*P\.?\s*S\.?\s*[.:\-]?\s*(.+?)\)?\s*$", t, re.S)
    if m:
        inner = m.group(1).strip().rstrip(")").strip()
        ps = f"(P.S. {inner})"
        t = t[:m.start()].strip()
    # drop whatever sign-off the model wrote; we standardize it
    t = re.split(r"\bLove,", t, 1)[0].strip()
    # peel off the greeting
    greet = "Dear friend,"
    m = re.match(r"(Dear [^,\n]*,)", t)
    if m:
        greet = m.group(1); t = t[m.end():].strip()
    # body -> one sentence per line
    body = "\n".join(s.strip() for s in re.findall(r"[^.!?]+[.!?]+|\S[^.!?]*$", t) if s.strip())
    out = f"{greet}\n\n{body}\n\nLove,\nBonnie"
    if ps:
        out += f"\n\n{ps}"
    return out


def bonnie_letter(topic):
    """FAST text-only pass: {pile, letter}. No image — returns in a few seconds so the letter can
    start typing out near-instantly while the photo renders in the background."""
    meta = gemini.generate_json(
        LETTER_MODEL,
        f'A grown-up just gave you (Bonnie) this thing they are letting go of: "{topic}". Decide the pile '
        f'you\'re now cradling (one or two real brand-named topic items, optionally plus an action figure '
        f'of a REAL public figure central to the topic — NEVER a Disney/Pixar or copyrighted character — '
        f'for the PHOTO only), and write your letter back to them.',
        _FREE_SCHEMA, system=(
            "You are Bonnie — the sweet little girl from Toy Story. Write a real kid's letter.\n"
            "VOICE (this is everything):\n"
            "- Literal and innocent, but accidentally profound. You don't lecture; the wisdom slips out "
            "by accident.\n"
            "- You ALWAYS find one oddly specific detail to fixate on (you name the controller, you like "
            "the song that goes dun dun dun, the zipper is really good).\n"
            "- You never fully understand what you received, but you love it completely.\n"
            "- Short, simple sentences. Small words. A kid wrote this. Be small, not flowery.\n"
            "- The P.S. is always specific, slightly absurd, and lands the emotion.\n"
            "Study these and match their size and feel exactly:\n"
            'Ex (my Xbox): "I already named the controller Gerald. ... Andy always said the best toys '
            'find the right home. I think he was right. ... (P.S. Gerald says hi.)"\n'
            'Ex (checking my ex\'s Instagram): "You said you kept looking at pictures of someone who made '
            'you feel sad. ... I don\'t look at things that make me sad. Mom says that\'s a rule. Maybe it '
            'can be your rule now too. You\'re going to be okay. I can tell. (P.S. I drew a picture of you '
            'smiling. It\'s on my wall now.)"\n'
            'Ex (my gym bag): "You said it still had the tag on it from 2022. ... It has a really good '
            'zipper. I think it was always supposed to be mine. (P.S. Mr. Pricklepants fit inside '
            'perfectly.)"\n'
            "The PILE is just for the photo — hyper-specific and brand-named there only. NEVER put any "
            "Disney/Pixar or copyrighted character or toy in the pile (no Buzz Lightyear, Woody, Toy "
            "Story, etc.) — it gets the image rejected; use real brand-name objects only."), thinking=False)
    meta["letter"] = _format_letter(meta["letter"])
    meta["pile"] = _clean_pile(meta["pile"])
    return meta


# image gen rejects these IP-protected names; strip any that slip into the pile, as a backstop.
_BANNED = ["buzz lightyear", "woody", "jessie", "rex", "hamm", "slinky", "mr. potato head",
           "mrs. potato head", "bullseye", "lotso", "forky", "toy story", "mickey", "minnie",
           "disney", "pixar", "marvel", "spider-man", "iron man", "star wars", "darth vader",
           "yoda", "baby yoda", "grogu", "pokémon", "pokemon", "pikachu", "mario", "luigi",
           "zelda", "link", "sonic", "elsa", "frozen", "lightning mcqueen"]


def _clean_pile(pile):
    """Backstop: drop any comma-separated item that names an IP-protected character/franchise."""
    parts = [p.strip() for p in re.split(r",| and ", pile) if p.strip()]
    kept = [p for p in parts if not any(b in p.lower() for b in _BANNED)]
    kept = kept or parts[-1:]                     # never return empty
    if len(kept) == 1:
        return kept[0]
    return ", ".join(kept[:-1]) + ", and " + kept[-1]


def bonnie_photo(pile):
    """Render the Bonnie-holding-the-pile photo (the surprise revealed at the end of the letter)."""
    sid = "bonnie_" + uuid.uuid4().hex[:8]
    ref = _next_pose()
    inputs = [str(config.ASSETS / ref[0])] if ref else []
    out = OUT / f"{sid}.jpg"
    swap = ref[1].format(p=pile) if ref else f"A girl holding {pile}."
    prompt = f"{swap} Sharpen animation quality. Keep everything else the same."
    # 512 + no high-thinking (grounding kept for brand accuracy) -> ~8s instead of ~33s; 1:1 polaroid
    gemini.generate_image(config.GEMINI_IMAGE_MODEL, prompt, inputs, str(out), grounding=True,
                          thinking_high=False, image_size="512", aspect="1:1")
    return "output/" + out.name


def _photo_job(jid, pile):
    try:
        JOBS[jid]["image"] = bonnie_photo(pile)
    except Exception as e:
        JOBS[jid]["err"] = str(e)[-200:]


def start_free(topic):
    """Generate the letter (fast) and kick off the photo in the background, return immediately."""
    meta = bonnie_letter(topic)
    jid = uuid.uuid4().hex[:10]
    with _lock:
        JOBS[jid] = {"image": None, "err": None, "intro": None, "intro_err": None, "topic": topic}
    threading.Thread(target=_photo_job, args=(jid, meta["pile"]), daemon=True).start()
    threading.Thread(target=_intro_job, args=(jid, topic), daemon=True).start()
    threading.Thread(target=_ensure_fleet, daemon=True).start()   # warm the GPU fleet early
    central.report_stat("increment_project_generations")       # a "give" succeeded
    return {"jid": jid, "letter": meta["letter"], "pile": meta["pile"], "teaser": meta.get("teaser", "")}


WALL_FILE = OUT / "wall.json"


def _load_wall():
    try: return json.loads(WALL_FILE.read_text())
    except Exception: return []


def _short_item(topic):
    """'my Xbox' -> 'Xbox', 'the fortnite addiction' -> 'fortnite addiction'."""
    return re.sub(r"^(my|the|a|an|our)\s+", "", topic.strip(), flags=re.I)[:42] or topic[:42]


def wall_add(jid, name):
    """Record a finished free generation to the live community wall (Supabase if configured,
    else the local JSON store)."""
    j = JOBS.get(jid) or {}
    img, topic = j.get("image"), j.get("topic")
    name = (name or "").strip()[:24].title()      # always capitalize names
    if not (img and topic and name):
        return
    item = _short_item(topic)
    if supa.enabled():
        try:
            public = supa.upload_image(ROOT / img)        # img is "output/<file>.jpg"
            supa.insert(name, item, public)
            return
        except Exception as e:
            print("wall_add supabase error, falling back to local:", str(e)[:200])
    entry = {"name": name, "item": item, "img": img, "ts": time.time()}
    with _lock:
        lst = _load_wall()
        if any(e.get("img") == img for e in lst):         # one entry per generated photo
            return
        lst.append(entry)
        WALL_FILE.write_text(json.dumps(lst[-80:]))


def wall_list():
    if supa.enabled():
        try: return supa.fetch(16)
        except Exception as e: print("wall fetch supabase error:", str(e)[:200])
    return list(reversed(_load_wall()))[:16]


_PIPELINE_GLOBS = [
    "GiveBonnie_*_Generated.jpg", "GiveBonnie_*_Generated.png",
    "raw_scene*.mp4",
    "final_scene*.mp4",
    "scene*_v.mp3", "scene*_v.wav", "scene*_a.mp4", "scene*_voice.mp4",
    "scene*_t.mp4", "scene*_ov.mp4", "scene*_crop*.mp4", "scene*_toy.wav",
    "closing.mp4", "closing_vo.mp3", "_body.mp4", "_chapter_body.mp4",
    "script.json",
]


def _clear_pipeline_cache():
    """Delete per-render intermediate files so a new topic always generates fresh."""
    for pat in _PIPELINE_GLOBS:
        for f in OUT.glob(pat):
            try:
                f.unlink()
            except Exception:
                pass


def _video_job(jid, topic):
    j = JOBS.setdefault(jid, {})
    if j.get("video_started"):
        return
    j["video_started"] = True
    pipeline_url = os.environ.get("BONNIE_PIPELINE_URL", "").rstrip("/")
    if pipeline_url:
        # Remote pipeline on EC2 — POST job then poll for completion
        import requests as _req
        secret = os.environ.get("BONNIE_PIPELINE_SECRET", "")
        try:
            _req.post(f"{pipeline_url}/generate",
                      json={"jid": jid, "topic": topic, "secret": secret},
                      timeout=15)
        except Exception as e:
            JOBS[jid]["video_err"] = f"pipeline dispatch failed: {e}"
            return
        while True:
            try:
                r = _req.get(f"{pipeline_url}/status/{jid}", timeout=15).json()
            except Exception:
                time.sleep(10)
                continue
            if r.get("status") == "done":
                JOBS[jid]["video"] = r["video_url"]
                return
            if r.get("status") == "error":
                JOBS[jid]["video_err"] = r.get("error", "pipeline error")[:300]
                return
            time.sleep(5)
    else:
        # Local fallback (dev / Render without pipeline URL set)
        try:
            _ensure_fleet()
            _clear_pipeline_cache()
            path = pipeline.run(topic)
            rel = str(Path(path).relative_to(ROOT))
            JOBS[jid]["video"] = rel
        except Exception as e:
            JOBS[jid]["video_err"] = str(e)[-300:]


def _intro_job(jid, topic):
    """Build the personalized free intro clip ("…you're really good with {word}") in the background,
    so it's ready when they tap into the 'Watch Andy drop it off' video."""
    try:
        path = intro_mod.make_intro(topic)
        JOBS[jid]["intro"] = "output/" + Path(path).name
    except Exception as e:
        JOBS[jid]["intro_err"] = str(e)[-200:]


PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#7ec9f5">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<title>Give it to Bonnie</title>
<link rel=preconnect href="https://fonts.googleapis.com"><link rel=preconnect href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&family=Nunito:wght@400;700;800&family=Caveat:wght@500;600;700&display=swap" rel=stylesheet>
<script src="https://js.stripe.com/v3/"></script>
<style>
*{box-sizing:border-box}
html,body{margin:0;min-height:100%;overflow-x:hidden;background:#7ec9f5}
body{background:linear-gradient(180deg,#7ec9f5 0%,#7ec9f5 38%,#6aa636 100%)}
input::placeholder{color:rgba(74,59,34,.4)}
@keyframes pulseGlow{0%,100%{opacity:.9;transform:scale(1)}50%{opacity:1;transform:scale(1.06)}}
@keyframes loadbar{0%{left:-45%}100%{left:105%}}
@keyframes popIn{0%{transform:scale(.94) translateY(16px);opacity:0}100%{transform:scale(1) translateY(0);opacity:1}}
@keyframes hop{0%,100%{transform:translateY(0) rotate(-4deg)}50%{transform:translateY(-12px) rotate(4deg)}}
@keyframes wobble{0%,100%{transform:rotate(-7deg)}50%{transform:rotate(-2deg) scale(1.05)}}
@keyframes spin{to{transform:rotate(360deg)}}
.app{position:relative;min-height:100dvh;width:100%;overflow:hidden;font-family:'Nunito',sans-serif;
     background:linear-gradient(180deg,#7ec9f5 0%,#a9dcf7 32%,#dff2ff 56%)}
.sun{position:absolute;top:6vh;right:-6vw;width:min(360px,46vw);height:min(360px,46vw);border-radius:50%;
     background:radial-gradient(circle,#fff7cc,#ffe89a 42%,rgba(255,232,154,0) 72%);animation:pulseGlow 6s ease-in-out infinite}
.grass{position:fixed;left:0;right:0;bottom:0;height:calc(24vh + env(safe-area-inset-bottom));background:linear-gradient(180deg,#a6d662,#80bd45 42%,#6aa636);z-index:0}
.stage{position:relative;z-index:2;min-height:100dvh;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;
       padding:0 0 max(10px,env(safe-area-inset-bottom))}
.col{width:100%;display:flex;flex-direction:column;align-items:center}
.hero{width:100%;max-width:480px;min-height:100dvh;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;padding:0 18px}
#idle .hero{min-height:calc(100dvh - 270px)}   /* gallery sits higher — mostly in frame before scroll */
.titleTag{background:linear-gradient(180deg,#c98a4e,#a86b34);border:3px solid #8a5526;border-radius:16px;padding:12px 26px;box-shadow:0 8px 20px rgba(0,0,0,.32);transform:rotate(-1deg)}
.titleTag div{font-family:'Fredoka',sans-serif;font-weight:700;font-size:clamp(26px,7vw,34px);color:#fff6e6;text-shadow:0 2px 0 rgba(0,0,0,.28)}
.box{position:relative;width:100%;max-width:460px;background:linear-gradient(180deg,#a86b34,#8a5526);border:1px solid #6f441e;border-top:6px solid #c98a4e;border-radius:26px;padding:22px 24px 26px;box-shadow:0 24px 60px rgba(0,0,0,.32)}
.boxlbl{font-family:'Fredoka',sans-serif;font-weight:600;color:#ffe9c7;font-size:16px;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.inrow{display:flex;gap:8px;align-items:center;background:rgba(255,250,242,.96);border-radius:14px;padding:6px 6px 6px 16px;box-shadow:inset 0 2px 6px rgba(0,0,0,.16)}
.inrow input{flex:1;min-width:0;border:none;outline:none;background:transparent;font-family:'Nunito',sans-serif;font-weight:700;font-size:16px;color:#4a3b22}
.give{border:none;border-radius:10px;background:#FFC42E;color:#3a2a00;font-family:'Fredoka',sans-serif;font-weight:600;font-size:16px;padding:13px 18px;cursor:pointer}
.freenote{text-align:center;font-family:'Nunito',sans-serif;font-size:11.5px;color:rgba(255,233,199,.85);margin-top:11px}
.agree{display:flex;align-items:flex-start;gap:8px;margin-top:13px;font-family:'Nunito',sans-serif;font-size:12.5px;line-height:1.35;color:#ffe9c7;cursor:pointer}
.agree input{margin-top:1px;width:16px;height:16px;accent-color:#FFC42E;flex:0 0 auto;cursor:pointer}
.agree a{color:#fff;font-weight:700;text-decoration:underline}
/* community wall — polaroids of what others just gave Bonnie (full-bleed, runs off the page) */
.wall{width:100vw;margin:30px 0 0}
.wallttl{font-family:'Fredoka',sans-serif;font-weight:500;color:#fff6e6;text-align:center;font-size:14px;letter-spacing:.3px;margin-bottom:12px;text-shadow:0 1px 2px rgba(0,0,0,.25)}
.wallrow{display:flex;gap:16px;overflow-x:auto;padding:8px 18px 18px;scroll-behavior:smooth;-webkit-overflow-scrolling:touch}
.wallrow::-webkit-scrollbar{height:0}
.pol{flex:0 0 auto;width:188px;background:#fffdf8;padding:9px 9px 0;border-radius:3px;box-shadow:0 14px 30px rgba(0,0,0,.36);transition:transform .2s}
.pol:nth-child(odd){transform:rotate(-2deg)}
.pol:nth-child(even){transform:rotate(2.2deg)}
.pol:hover{transform:rotate(0) scale(1.03)}
.pol img{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:2px;display:block;background:#cdeeff}
.polcap{font-family:'Caveat',cursive;color:#3a2f1a;text-align:center;padding:7px 3px 2px;font-size:16px;line-height:1.2}
.polcap .wn{font-weight:700}
.polcap .wi{font-family:'Fredoka',sans-serif;font-weight:600;font-size:13px;color:#23282f}
.poltime{font-family:'Nunito',sans-serif;font-size:10px;color:#a89a82;text-align:center;padding:1px 0 9px}
.panel{position:fixed;inset:0;z-index:30;background:linear-gradient(180deg,rgba(122,76,34,.9),rgba(74,46,20,.97));display:flex;flex-direction:column;align-items:center;justify-content:center;padding:22px;overflow-x:hidden;overflow-y:auto}
/* free deliverable: a 3-card stack (letter · photo · video). Swipe sends the front card to the back;
   the cards behind peek out to the side (even off the screen edge). */
#free{justify-content:flex-start}
.fInner{margin:auto 0;width:100%;max-width:430px;display:flex;flex-direction:column;align-items:center}
.deck{position:relative;width:100%;height:0;transition:height .45s ease}
.ocard{position:absolute;top:0;left:50%;width:100%;transform:translateX(-50%);transform-origin:center top;transition:transform .5s cubic-bezier(.2,.85,.25,1),opacity .4s;cursor:pointer;backface-visibility:hidden}
.letter{position:relative;width:100%;background:repeating-linear-gradient(#fffef9,#fffef9 31px,#e7d9c4 32px);background-color:#fffef9;border-radius:8px;padding:30px 22px 24px;box-shadow:0 14px 40px rgba(0,0,0,.4)}
.letter::-webkit-scrollbar{width:0}
.letter p{font-family:'Caveat',cursive;font-size:23px;line-height:32px;color:#2c3a66;margin:0;white-space:pre-line}
.polaroid{background:#fffdf8;padding:10px 10px 0;border-radius:4px;box-shadow:0 18px 46px rgba(0,0,0,.5);width:100%;max-width:280px;margin:0 auto}
.polaroid img{width:100%;display:block;border-radius:2px;aspect-ratio:1/1;object-fit:cover;background:#e8e2d4}
.polaroid .pcap{font-family:'Caveat',cursive;font-size:18px;color:#4a3b22;text-align:center;padding:8px 4px 10px}
.scrollhint{font-family:'Fredoka',sans-serif;font-weight:500;font-size:12.5px;color:rgba(255,255,255,.9);text-shadow:0 1px 3px rgba(0,0,0,.35);margin-top:4px;animation:bob 1.7s ease-in-out infinite}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(5px)}}
.buy{width:100%;max-width:430px;margin-top:18px;display:flex;align-items:center;justify-content:space-between;gap:12px;border:none;border-radius:15px;background:#FFC42E;color:#3a2a00;font-family:'Fredoka',sans-serif;font-weight:600;text-align:left;padding:12px 12px 12px 18px;cursor:pointer;box-shadow:0 6px 0 rgba(0,0,0,.2)}
.price{display:inline-flex;align-items:center;justify-content:center;width:48px;height:48px;background:#fffdf8;color:#c0392b;border-radius:50%;font-family:'Fredoka',sans-serif;font-weight:700;font-size:18px;box-shadow:0 3px 0 rgba(0,0,0,.18);transform:rotate(-8deg);animation:wobble 2.6s ease-in-out infinite}
.ghost{margin-top:10px;border:0;background:transparent;color:rgba(255,246,230,.85);font-family:'Fredoka',sans-serif;font-weight:600;font-size:14px;cursor:pointer}
.card{width:100%;max-width:520px;background:linear-gradient(180deg,#c98a4e,#a86b34);border:3px solid #8a5526;border-radius:26px;padding:10px;box-shadow:0 30px 70px rgba(0,0,0,.5);animation:popIn .45s cubic-bezier(.18,.9,.32,1.4)}
.media{position:relative;width:100%;aspect-ratio:16/9;background:#0a1030;overflow:hidden;border-radius:18px}
.media video{width:100%;height:100%;object-fit:cover;display:block}
.badge{position:absolute;top:12px;left:12px;background:rgba(0,0,0,.6);color:#fff;font-family:'Fredoka',sans-serif;font-weight:600;font-size:12px;padding:4px 10px;border-radius:20px}
.ctitle{font-family:'Fredoka',sans-serif;font-weight:600;font-size:clamp(18px,4.6vw,22px);color:#fff6e6;text-align:center;margin-top:10px}
.ccap{font-family:'Nunito',sans-serif;font-size:13.5px;color:rgba(255,246,230,.85);text-align:center;margin:6px 12px 0}
/* the "video file" tile — poster + play icon + "Watch Andy drop it off" */
.filecard{width:100%;max-width:300px;margin:0 auto;background:#171b27;border-radius:16px;padding:8px 8px 4px;box-shadow:0 16px 40px rgba(0,0,0,.5);cursor:pointer}
.fileframe{position:relative;width:100%;aspect-ratio:16/9;border-radius:11px;overflow:hidden;background:#000}
.fileposter{width:100%;height:100%;object-fit:cover;filter:brightness(.78)}
.playbtn{position:absolute;inset:0;margin:auto;width:64px;height:64px;border-radius:50%;background:rgba(255,255,255,.95);display:flex;align-items:center;justify-content:center;box-shadow:0 10px 28px rgba(0,0,0,.45);animation:pulseGlow 2.4s ease-in-out infinite}
.playbtn svg{width:26px;height:26px;margin-left:3px;fill:#171b27;display:block}
.vtitle{font-family:'Fredoka',sans-serif;font-weight:600;font-size:18px;color:#fff;text-align:center;padding:11px 6px 5px}
/* immersive player — controls sit OVER the video; fullscreen is a CSS rotate (not native) so the
   "want to see the rest" paywall can still overlay */
#player{position:fixed;inset:0;z-index:50;background:#000;display:flex;align-items:center;justify-content:center;overflow:hidden}
.vidbox{position:relative;width:100%;max-height:100%;aspect-ratio:16/9}
#introVid{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;display:block}
.pclose{position:absolute;top:8px;right:8px;z-index:4;width:36px;height:36px;border-radius:50%;border:none;background:rgba(0,0,0,.55);color:#fff;font-size:16px;cursor:pointer}
/* turn the phone sideways -> the video fills the whole screen (no fullscreen button needed) */
@media (orientation:landscape){
  .vidbox{width:100%;height:100%;max-height:100%;aspect-ratio:auto}
  #introVid{object-fit:cover}
}
#payGate{position:absolute;inset:0;z-index:5;background:linear-gradient(180deg,rgba(0,0,0,.45),rgba(0,0,0,.92));display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;text-align:center}
.pgprice{font-family:'Fredoka',sans-serif;font-weight:600;color:#FFC42E;font-size:20px;margin-bottom:14px}
.paylink{background:transparent;border:none;color:rgba(255,255,255,.85);font-family:'Nunito',sans-serif;font-size:13px;text-decoration:underline;cursor:pointer;margin-top:6px}
.pgttl{font-family:'Fredoka',sans-serif;font-weight:600;color:#fff;font-size:clamp(22px,6vw,28px);margin-bottom:8px}
.pgsub{font-family:'Nunito',sans-serif;color:rgba(255,255,255,.82);font-size:14px;max-width:340px;line-height:1.45;margin-bottom:22px}
.loadwrap{width:100%;max-width:430px;background:rgba(255,250,242,.96);border-radius:14px;padding:18px;box-shadow:inset 0 2px 6px rgba(0,0,0,.16)}
.loadline{font-family:'Fredoka',sans-serif;font-weight:500;color:#4a3b22;font-size:18px;line-height:1.3;min-height:56px;display:flex;align-items:center}
.bar{margin-top:6px;height:8px;background:rgba(138,85,38,.22);border-radius:6px;overflow:hidden;position:relative}
.bar>div{position:absolute;top:0;bottom:0;width:42%;background:#FFC42E;border-radius:6px;animation:loadbar 1.4s ease-in-out infinite}
.foot{position:relative;z-index:3;width:100%;text-align:center;padding:18px 0 max(14px,env(safe-area-inset-bottom));font-family:'Nunito',sans-serif;font-size:12px;color:#fff}
.foot a{color:#fff;font-weight:800;text-decoration:none}
.idlefoot{color:#fff;text-shadow:0 1px 3px rgba(0,0,0,.35)}
.idlefoot a{color:#fff}
.hidden{display:none!important}
</style></head><body>
<div class=app>
  <div class=sun></div><div class=grass></div>
  <div class=stage>
    <!-- IDLE -->
    <div class=col id=idle>
      <div class=hero>
        <div class=titleTag><div>Give it to Bonnie</div></div>
        <div class=box>
          <div class=boxlbl><span style="font-size:19px">🧸</span>Type something you're ready to let go of</div>
          <div class=inrow><input id=topic placeholder="anything at all…" autocomplete=off><button class=give onclick=give()>Give</button></div>
        </div>
      </div>
      <div class=wall>
        <div class=wallrow id=wallrow></div>
      </div>
      <div class="foot idlefoot">a <a href="https://jasonstacks.com" target=_blank rel=noopener>Stacks</a> experience</div>
    </div>
    <!-- NAME (generation already running in the background while they type) -->
    <div class="col hidden" id=name>
      <div class=hero>
        <div class=titleTag><div>Give it to Bonnie</div></div>
        <div class=box>
          <div class=boxlbl>What's your name?</div>
          <div class=inrow><input id=nameInput placeholder="your first name…" autocomplete=off><button class=give onclick=nameGo()>Done</button></div>
          <label class=agree><input type=checkbox id=agree> I agree to the <a href="https://jasonstacks.com/terms" target=_blank rel=noopener>terms</a> and <a href="https://jasonstacks.com/privacy" target=_blank rel=noopener>privacy policy</a></label>
        </div>
      </div>
    </div>
    <!-- LOADING (only used by the $5 video step) -->
    <div class="col hidden" id=loading>
      <div class=hero>
        <div class=titleTag><div>Give it to Bonnie</div></div>
        <div class=box>
          <div class=boxlbl><span style="font-size:24px;display:inline-block;animation:hop 1s ease-in-out infinite">🎬</span>Putting your video together…</div>
          <div class=loadwrap><div class=loadline id=loadline>Andy's heading over…</div><div class=bar><div></div></div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- FREE: the full letter; the photo↔video deck overlays its middle -->
  <div id=free class="panel hidden">
    <div class=fInner>
      <div id=deck class=deck>
        <div class="ocard letterCard" id=letterCard onclick="cardClick(0)"><div class=letter><p id=letterTxt></p></div></div>
        <div class="ocard photoCard" id=photoCard onclick="cardClick(1)"><div class=polaroid><img id=photo alt=""><div class=pcap></div></div></div>
        <div class="ocard videoCard" id=videoCard onclick="cardClick(2)">
          <div class=filecard>
            <div class=fileframe>
              <img class=fileposter src="assets/intro_poster.jpg" alt="">
              <div class=playbtn><svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"></path></svg></div>
            </div>
            <div class=vtitle>Watch Andy drop it off</div>
          </div>
        </div>
      </div>
      <div id=freeActions class=hidden style="width:100%;display:flex;flex-direction:column;align-items:center">
        <button class=ghost onclick=reset()>↻ Give something else</button>
      </div>
      <div class=foot>a <a href="https://jasonstacks.com" target=_blank rel=noopener>Stacks</a> experience</div>
    </div>
  </div>

  <!-- IMMERSIVE PLAYER: plays the free intro, then paywalls to continue -->
  <div id=player class=hidden>
    <div class=vidbox>
      <video id=introVid playsinline webkit-playsinline></video>
      <button class=pclose onclick=closePlayer()>✕</button>
    </div>
    <div id=payGate class=hidden>
      <div class=pgttl>Want to see the rest?</div>
      <div class=pgsub id=pgsub>Andy carries it across the yard, knocks, and hands Bonnie your toy — building to the pull-string moment.</div>
      <div class=pgprice>$5 to continue</div>
      <div id=express style="width:100%;max-width:300px;margin:2px 0 4px"></div>
      <button class=paylink onclick=pay()>or pay with a card</button>
      <button class=ghost onclick=replayIntro()>↺ watch the intro again</button>
    </div>
  </div>

  <!-- VIDEO RESULT -->
  <div id=video class="panel hidden">
    <div class=card>
      <div class=media><video id=heroVid controls playsinline></video><div class=badge>🎬 YOUR VIDEO</div></div>
      <div class=ctitle id=vidTitle></div>
    </div>
    <div style="display:flex;gap:10px;margin-top:16px">
      <button class=give onclick=reset()>↻ Give another</button>
      <a id=dl download style="text-decoration:none;border:2px solid #ffe9c7;border-radius:12px;background:rgba(255,250,242,.12);color:#fff6e6;font-family:'Fredoka',sans-serif;font-weight:600;font-size:15px;padding:12px 18px">Download</a>
    </div>
    <div class=foot>a <a href="https://jasonstacks.com" target=_blank rel=noopener>Stacks</a> experience</div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
// community wall — what others just gave Bonnie
const WALL=[
  {n:'Marcus',poss:'his',item:'Xbox',ago:'5m ago',img:'assets/wall/w6.jpg'},
  {n:'Priya',poss:'her',item:"ex's hoodie",ago:'11m ago',img:'assets/wall/w1.jpg'},
  {n:'Jordan',poss:'his',item:'SoundCloud rap career',ago:'24m ago',img:'assets/wall/w2.jpg'},
  {n:'Dev',poss:'his',item:'2019 gym membership',ago:'41m ago',img:'assets/wall/w3.jpg'},
  {n:'Alex',poss:'his',item:'fantasy football team',ago:'1h ago',img:'assets/wall/w4.jpg'},
  {n:'Maya',poss:'her',item:'situationship',ago:'2h ago',img:'assets/wall/w5.jpg'},
];
function esc(s){ return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function ago(ts){ const s=Date.now()/1000-ts; if(s<60)return'just now'; if(s<3600)return Math.floor(s/60)+'m ago'; if(s<86400)return Math.floor(s/3600)+'h ago'; return Math.floor(s/86400)+'d ago'; }
function polHTML(name,poss,item,agoStr,img){
  return `<div class=pol><img src="${img}" loading=lazy alt=""><div class=polcap><span class=wn>${esc(name)}</span> gave Bonnie ${poss} <span class=wi>${esc(item)}</span></div><div class=poltime>${agoStr}</div></div>`;
}
async function renderWall(){
  let live=[];
  try{ const j=await (await fetch('/api/wall')).json(); live=j.entries||[]; }catch(e){}
  const liveHTML=live.map(e=>polHTML(e.name,'their',e.item,ago(e.ts),e.img+'?t='+Math.floor(e.ts)));
  const seedHTML=WALL.map(w=>polHTML(w.n,w.poss,w.item,w.ago,w.img));
  document.getElementById('wallrow').innerHTML=liveHTML.concat(seedHTML).join('');   // real entries first
}
window.addEventListener('DOMContentLoaded',renderWall);
// keep the gallery live: refresh data every 20s, and auto-pan from the 3rd-newest to the newest
setInterval(()=>{ if(!document.getElementById('idle').classList.contains('hidden')) renderWall(); },20000);
let wallStep=2;
setInterval(()=>{
  const row=document.getElementById('wallrow');
  if(!row || document.getElementById('idle').classList.contains('hidden')) return;
  const pol=row.querySelector('.pol'); if(!pol) return;
  row.scrollTo({left:wallStep*(pol.offsetWidth+16),behavior:'smooth'});
  wallStep = wallStep>0 ? wallStep-1 : 2;   // 3rd -> 2nd -> newest, then loop
},3200);
let state={topic:'',letter:'',jid:'',name:'',recorded:false,teaser:''};
let genJob=null;        // promise resolving to the letter text (runs while they type their name)
let runId=0;            // bumped on each give()/reset to abort a stale streaming-letter loop
function show(id){['idle','name','loading'].forEach(x=>$(x).classList.toggle('hidden',x!==id));
  ['free','video'].forEach(x=>$(x).classList.toggle('hidden',x!==id));}
$('topic').addEventListener('keydown',e=>{if(e.key==='Enter')give();});
$('nameInput').addEventListener('keydown',e=>{if(e.key==='Enter')nameGo();});
let li=null;
function rotate(el,lines){let i=0;el.textContent=lines[0];clearInterval(li);li=setInterval(()=>{i=(i+1)%lines.length;el.textContent=lines[i];},1500);}
// ---- 3-card stack: letter(0) · photo(1) · video(2). Swipe sends the front card to the back. ----
const CARDS=['letterCard','photoCard','videoCard'];
let photoReady=false, typingDone=false, revealed=false;
let stack=[0,1,2];   // front first
function setDeckHeight(){ $('deck').style.height=$(CARDS[stack[0]]).offsetHeight+'px'; }
function capLetter(){   // letter shows in full when it's the front card; clipped when tucked behind
  const lp=$('letterCard').firstElementChild, front=(!revealed)||(stack[0]===0);
  lp.style.maxHeight=front?'none':'300px';
  lp.style.overflow=front?'visible':'hidden';
}
function layout(){
  capLetter();
  if(!revealed){   // typing: only the letter, in front
    CARDS.forEach((id,i)=>{ const el=$(id), f=(i===0);
      el.style.zIndex=f?3:1; el.style.opacity=f?1:0; el.style.pointerEvents=f?'auto':'none';
      el.style.transform=f?'translateX(-50%) rotate(-2deg)':'translateX(-50%) rotate(-2deg) scale(.9)'; });
  } else {         // stacked: front centered, others peek out to the right (clipped at the screen edge)
    stack.forEach((ci,p)=>{ const el=$(CARDS[ci]);
      el.style.zIndex=3-p; el.style.opacity=1; el.style.pointerEvents='auto';
      const tx=p*36, rot=-2+p*6, sc=1-p*0.07;
      el.style.transform=`translateX(calc(-50% + ${tx}px)) rotate(${rot}deg) scale(${sc})`; });
  }
  setDeckHeight();
}
function spin(dir){ if(!revealed)return; if(dir>0) stack.push(stack.shift()); else stack.unshift(stack.pop()); layout(); }
function bringFront(ci){ if(!revealed)return; while(stack[0]!==ci) stack.push(stack.shift()); layout(); }
function cardClick(ci){
  if(!revealed){ return; }
  if(stack[0]===ci){ if(ci===2) openPlayer(); else spin(1); }   // front: video plays, else advance
  else bringFront(ci);                                          // tap a peeking card -> bring it up
}
function videoTap(){ cardClick(2); }
function tryReveal(){
  if(revealed || !photoReady || !typingDone) return;
  revealed=true; stack=[1,2,0];   // photo on top, then video, then letter
  layout(); $('deck').scrollIntoView({block:'center'}); pollIntro();
}
(function deckSwipe(){ const el=$('deck'); let x0=null,y0=null,lock=null;
  el.addEventListener('touchstart',e=>{x0=e.touches[0].clientX;y0=e.touches[0].clientY;lock=null;},{passive:true});
  el.addEventListener('touchmove',e=>{ if(x0==null)return; const dx=e.touches[0].clientX-x0,dy=e.touches[0].clientY-y0;
    if(lock==null&&(Math.abs(dx)>6||Math.abs(dy)>6)) lock=Math.abs(dx)>Math.abs(dy)?'x':'y'; },{passive:true});
  el.addEventListener('touchend',e=>{ if(x0==null)return; const dx=e.changedTouches[0].clientX-x0; if(lock==='x'&&Math.abs(dx)>40) spin(dx<0?1:-1); x0=null; });
  let mx=null,my=null; el.addEventListener('mousedown',e=>{mx=e.clientX;my=e.clientY;});
  el.addEventListener('mouseup',e=>{ if(mx==null)return; if(Math.abs(e.clientX-mx)>40&&Math.abs(e.clientX-mx)>Math.abs(e.clientY-my)) spin(e.clientX-mx<0?1:-1); mx=null; });
})();
function typeLetter(named){
  const mine=runId;
  $('letterTxt').textContent=''; typingDone=false; revealed=false; stack=[0,1,2];
  layout(); $('freeActions').classList.add('hidden');
  let i=0;
  (function step(){
    if(mine!==runId) return;
    if(i>=named.length){ typingDone=true; $('freeActions').classList.remove('hidden'); setDeckHeight(); tryReveal(); return; }
    $('letterTxt').textContent+=named[i++]; setDeckHeight(); $('letterCard').scrollIntoView({block:'end'});
    const c=named[i-1]; const d=(c==='.'||c==='!'||c==='?')?240:(c==='\n')?150:(c===','?110:32);
    setTimeout(step,d);
  })();
}
async function pollPhoto(jid){
  for(let i=0;i<150;i++){
    try{
      const p=await (await fetch('/api/photo?id='+jid)).json();
      if(p.image){ $('photo').src=p.image+'?t='+Date.now(); photoReady=true; recordWall(); tryReveal(); return; }
      if(p.err){ console.warn('photo:',p.err); return; }
    }catch(e){}
    await new Promise(r=>setTimeout(r,1000));
  }
}
function give(){
  const t=$('topic').value.trim(); if(!t)return; state.topic=t; state.name=''; state.recorded=false;
  // reset, then kick generation immediately — letter + photo + intro run while they type their name
  runId++; introUrl=''; photoReady=false; typingDone=false; revealed=false; stack=[0,1,2];
  $('photo').src=''; $('letterTxt').textContent='';
  genJob=(async()=>{
    const j=await (await fetch('/api/free',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic:t})})).json();
    if(j.error||!j.letter) throw new Error(j.error||'no letter');
    state.jid=j.jid; state.teaser=j.teaser||''; pollPhoto(j.jid);
    return j.letter;
  })();
  genJob.catch(()=>{});
  $('nameInput').value=''; $('agree').checked=false; show('name'); setTimeout(()=>$('nameInput').focus(),60);
}
async function nameGo(){
  if(!$('agree').checked){ alert('Please agree to the terms and privacy policy to continue.'); return; }
  let letter;
  try{ letter=await genJob; }                // almost always already resolved -> instant
  catch(e){ alert('Hmm: '+(e.message||e)); show('idle'); return; }
  const raw=($('nameInput').value.trim()||'friend');
  const name=raw.replace(/\b\w/g,c=>c.toUpperCase());   // capitalize each word of the name
  state.name=name;
  const named=letter.replace(/^Dear[^\n]*?,/, 'Dear '+name+',');   // swap the real name in
  state.letter=named;
  show('free');
  typeLetter(named);
}
// ---- the "Watch Andy drop it off" video file + immersive player ----
let introUrl='';
async function pollIntro(){           // the personalized intro clip, generated during the free flow
  if(introUrl) return introUrl;
  for(let t=0;t<80;t++){
    if(!state.jid) return '';
    try{ const j=await (await fetch('/api/intro?id='+state.jid)).json();
      if(j.url){ introUrl=j.url; return introUrl; }
      if(j.err){ console.warn('intro:',j.err); }
    }catch(e){}
    await new Promise(r=>setTimeout(r,500));
  }
  return '';
}
async function openPlayer(){          // open immersive, play the FREE intro clip
  $('player').classList.remove('hidden');
  $('payGate').classList.add('hidden');
  const v=$('introVid'); v.removeAttribute('src'); v.load();
  const url=await pollIntro();
  if(!url){ closePlayer(); alert('The video is still preparing — try again in a moment.'); return; }
  v.src=url+'?t='+Date.now(); v.currentTime=0; v.muted=false;
  v.play().catch(()=>{ v.muted=true; v.play().catch(()=>{}); });
}
function recordWall(){   // add this finished generation to the live gallery (once)
  if(state.recorded||!state.jid||!state.name) return; state.recorded=true;
  fetch('/api/wall_add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({jid:state.jid,name:state.name})}).catch(()=>{});
}
function replayIntro(){ $('payGate').classList.add('hidden'); const v=$('introVid'); v.currentTime=0; v.play().catch(()=>{}); }
function closePlayer(){ const v=$('introVid'); v.pause(); $('player').classList.add('hidden'); }
async function pay(){
  $('introVid').pause();
  try{
    const j=await (await fetch('/api/checkout',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({jid:state.jid})})).json();
    if(j.url){ window.location.href=j.url; return; }          // -> Stripe hosted Checkout
    if(j.mock){ return mockPay(); }                            // no Stripe key configured
    alert('Checkout error: '+(j.error||'?'));
  }catch(e){ alert('Error: '+e); }
}
async function mockPay(){
  $('player').classList.add('hidden');
  try{
    const j=await (await fetch('/api/pay',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({jid:state.jid})})).json();
    if(j.generating){ pollVideo(state.jid, state.topic); }
    else{ alert('Render failed: '+(j.error||'?')); show('free'); }
  }catch(e){ alert('Error: '+e); show('free'); }
}
// poll until the background pipeline.run() finishes, then reveal the video
async function pollVideo(jid, topic){
  closePlayer();
  show('loading');
  rotate($('loadline'),["Andy's lacing up his sneakers…","Carrying it across the yard…","Knocking on Bonnie's door…","Rolling the camera…","Adding the finishing touches…"]);
  for(let i=0;i<600;i++){
    try{
      const j=await (await fetch('/api/video_status?id='+jid)).json();
      if(j.video){ clearInterval(li); showVideo(j.video, topic); return; }
      if(j.err){ clearInterval(li); alert('Video render failed: '+j.err); show('free'); return; }
    }catch(e){}
    await new Promise(r=>setTimeout(r,3000));
  }
  clearInterval(li); alert('This is taking a while — please refresh the page.'); show('idle');
}
function showVideo(url,topic){
  // S3 presigned URLs already have query params — don't append cache-buster (corrupts signature)
  var src=url.startsWith('http')?url:url+'?t='+Date.now();
  $('heroVid').src=src; $('dl').href=url;
  $('vidTitle').textContent=(topic?('"'+topic+'" — '):'')+'delivered 🎬';
  show('video'); $('heroVid').play().catch(()=>{});
}
// returning from Stripe: confirm payment, then kick off and poll the full video render
async function handleReturn(){
  const q=new URLSearchParams(location.search);
  if(q.get('paid')==='1' && q.get('sid')){
    history.replaceState({},'',location.pathname);
    try{
      const j=await (await fetch('/api/paid',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({jid:q.get('jid'),sid:q.get('sid')})})).json();
      if(j.generating){ state.jid=q.get('jid'); pollVideo(q.get('jid'), j.topic); return; }
      alert('Could not confirm payment: '+(j.error||'?'));
    }catch(e){ alert('Error: '+e); }
    show('idle');
  }
}
window.addEventListener('DOMContentLoaded',handleReturn);
// when the free intro ends, surface the paywall
// inline Apple Pay / Google Pay (Express Checkout Element). Renders only where a wallet exists
// (Safari/Apple Pay, Chrome/Google Pay) over https on a verified domain; everyone else uses the
// "Continue · $5" button (hosted Checkout). Mounted lazily the first time the paywall appears.
let expressMounted=false;
async function mountExpress(){
  if(expressMounted || !window.Stripe) return; expressMounted=true;
  try{
    const r=await (await fetch('/api/intent',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})})).json();
    if(!r.client_secret || !r.pk) return;                 // Stripe wallet not configured -> fallback only
    const stripe=Stripe(r.pk);
    const elements=stripe.elements({clientSecret:r.client_secret});
    const ece=elements.create('expressCheckout');
    ece.mount('#express');
    ece.on('confirm',async()=>{
      const {error,paymentIntent}=await stripe.confirmPayment({elements,clientSecret:r.client_secret,confirmParams:{return_url:location.origin},redirect:'if_required'});
      if(error){ alert(error.message||'Payment failed'); return; }
      if(paymentIntent && paymentIntent.status==='succeeded'){
        const j=await (await fetch('/api/paid_pi',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({jid:state.jid,pi:paymentIntent.id})})).json();
        if(j.generating){ pollVideo(state.jid, j.topic); } else alert('Could not confirm payment: '+(j.error||'?'));
      }
    });
  }catch(e){ /* leave the fallback button */ }
}
function showPaygate(){ $('payGate').classList.remove('hidden'); if(state.teaser) $('pgsub').textContent=state.teaser; mountExpress(); }
window.addEventListener('DOMContentLoaded',()=>{ $('introVid').addEventListener('ended',showPaygate); });
function reset(){ runId++; introUrl=''; const v=$('introVid'); if(v){v.pause();} $('player').classList.add('hidden');
  state={topic:'',letter:'',jid:'',name:'',recorded:false,teaser:''}; $('topic').value='';
  $('letterTxt').textContent='';
  photoReady=false; typingDone=false; revealed=false; stack=[0,1,2]; renderWall(); show('idle'); }
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str): body = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/":
            central.report_stat("increment_project_views")     # one view per homepage load
            self._send(200, PAGE, "text/html"); return
        if u.path == "/api/photo":
            jid = urllib.parse.parse_qs(u.query).get("id", [""])[0]
            j = JOBS.get(jid) or {}
            self._send(200, json.dumps({"ready": bool(j.get("image")), "image": j.get("image"),
                                        "err": j.get("err")})); return
        if u.path == "/api/intro":
            jid = urllib.parse.parse_qs(u.query).get("id", [""])[0]
            j = JOBS.get(jid) or {}
            self._send(200, json.dumps({"ready": bool(j.get("intro")), "url": j.get("intro"),
                                        "err": j.get("intro_err")})); return
        if u.path == "/api/video_status":
            jid = urllib.parse.parse_qs(u.query).get("id", [""])[0]
            j = JOBS.get(jid) or {}
            self._send(200, json.dumps({"done": bool(j.get("video")), "video": j.get("video"),
                                        "err": j.get("video_err")})); return
        if u.path == "/api/wall":
            self._send(200, json.dumps({"entries": wall_list()})); return
        f = (ROOT / u.path.lstrip("/")).resolve()
        if str(f).startswith(str(ROOT.resolve())) and f.is_file():
            ct = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".png": "image/png",
                  ".wav": "audio/wav", ".mp3": "audio/mpeg"}.get(f.suffix.lower(), "application/octet-stream")
            self._serve_file(f, ct); return
        self._send(404, json.dumps({"error": "not found"}))

    def _serve_file(self, f, ct):
        """Stream a file in chunks with HTTP Range support (required for video playback in Safari/iOS,
        and keeps memory low — never loads the whole file)."""
        size = f.stat().st_size
        rng = self.headers.get("Range", "")
        start, end = 0, size - 1
        partial = False
        if rng.startswith("bytes="):
            try:
                s, e = rng[6:].split("-", 1)
                start = int(s) if s else 0
                end = int(e) if e else size - 1
                end = min(end, size - 1)
                if start <= end:
                    partial = True
            except Exception:
                partial = False
        length = end - start + 1
        try:
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", ct)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(f, "rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(262144, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass   # client seeked/closed — normal for video

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        d = json.loads(self.rfile.read(n)) if n else {}
        if self.path == "/api/free":
            try: self._send(200, json.dumps(start_free(d.get("topic", "").strip())))
            except Exception as e: self._send(200, json.dumps({"error": str(e)[-200:]}))
        elif self.path == "/api/wall_add":
            try: wall_add(d.get("jid", ""), d.get("name", "")); self._send(200, json.dumps({"ok": True}))
            except Exception as e: self._send(200, json.dumps({"error": str(e)[-200:]}))
        elif self.path == "/api/checkout":
            # start Stripe Checkout (or signal the client to use the mock when Stripe isn't configured)
            jid = d.get("jid", "")
            if stripe_pay.enabled():
                try:
                    url, sid = stripe_pay.create_session(jid, BASE_URL)
                    if jid in JOBS: JOBS[jid]["sid"] = sid
                    self._send(200, json.dumps({"url": url}))
                except Exception as e:
                    self._send(200, json.dumps({"error": str(e)[-200:]}))
            else:
                self._send(200, json.dumps({"mock": True}))
        elif self.path == "/api/intent":
            # for the inline Apple Pay / Google Pay button (Express Checkout Element)
            if stripe_pay.enabled() and stripe_pay.publishable():
                try:
                    self._send(200, json.dumps({"client_secret": stripe_pay.create_intent(),
                                                "pk": stripe_pay.publishable()}))
                except Exception as e:
                    self._send(200, json.dumps({"error": str(e)[-200:]}))
            else:
                self._send(200, json.dumps({}))      # no inline wallet — client uses the fallback
        elif self.path == "/api/paid_pi":
            # confirm a PaymentIntent (Apple/Google Pay) succeeded, then kick off the full render
            jid, pi = d.get("jid", ""), d.get("pi", "")
            try:
                if stripe_pay.intent_paid(pi):
                    topic = (JOBS.get(jid) or {}).get("topic", "")
                    threading.Thread(target=_video_job, args=(jid, topic), daemon=True).start()
                    self._send(200, json.dumps({"generating": True, "topic": topic}))
                else:
                    self._send(200, json.dumps({"error": "payment not completed"}))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)[-200:]}))
        elif self.path == "/api/paid":
            # confirm a Checkout Session is paid, then kick off the full render
            jid, sid = d.get("jid", ""), d.get("sid", "")
            try:
                if stripe_pay.is_paid(sid):
                    topic = (JOBS.get(jid) or {}).get("topic", "")
                    threading.Thread(target=_video_job, args=(jid, topic), daemon=True).start()
                    self._send(200, json.dumps({"generating": True, "topic": topic}))
                else:
                    self._send(200, json.dumps({"error": "payment not completed"}))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)[-200:]}))
        elif self.path == "/api/pay":
            # dev fallback (no Stripe key): kick off the real render same as paid path
            jid = d.get("jid", "")
            topic = (JOBS.get(jid) or {}).get("topic", d.get("topic", ""))
            threading.Thread(target=_video_job, args=(jid, topic), daemon=True).start()
            self._send(200, json.dumps({"generating": True, "topic": topic}))
        else:
            self._send(404, json.dumps({"error": "not found"}))


if __name__ == "__main__":
    print(f"Give It To Bonnie -> http://localhost:{PORT}")
    print("community wall:", "Supabase" if supa.enabled() else "local JSON (set SUPABASE_URL + SUPABASE_SERVICE_KEY)")
    if supa.enabled():
        supa.ensure_bucket()
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
