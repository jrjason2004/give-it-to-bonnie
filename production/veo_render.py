"""
Veo 3.1 Lite full render (benchmark vs Wan). Per scene: Veo image-to-video from the scene's
Nano Banana image + the prompt (quotes included — Veo speaks them) → clip with native audio +
lip-sync. Then ElevenLabs speech-to-speech swaps Veo's voice for the consistent Andy voice on
his lines (s2s preserves timing, so the lip-sync stays matched). Then sequence.

    python veo_render.py "my ex's instagram"
"""
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import veo
import voice
import composite
from pipeline import fill, resolve, OUT, log


def _veo_scene(sc, script):
    sid = sc["id"]
    out = OUT / f"veo_{sid}.mp4"
    if out.exists() and out.stat().st_size > 10000:
        log(f"  ↻ reuse veo {sid}"); return sid, str(out)
    prompt = fill(sc["video"]["prompt"], script)
    img = resolve(sc["video"]["start"])          # Veo i2v first frame = the scene's image
    veo.generate_video(prompt, img, str(out))
    log(f"  ✓ veo {sid}")
    return sid, str(out)


def run(topic):
    script = json.loads((OUT / "script.json").read_text())
    log("VEO 1/3 — generating clips (Veo 3.1 lite, parallel)")
    clips = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        for f in as_completed([ex.submit(_veo_scene, sc, script) for sc in config.SCENES]):
            sid, path = f.result(); clips[sid] = path

    log("VEO 2/3 — Andy voice-change (ElevenLabs s2s) + per-scene normalize")
    norm = []
    for sc in config.SCENES:
        sid = sc["id"]; clip = clips[sid]
        if sc["audio"] == "andy":                # swap Veo's voice -> consistent Andy
            wav = str(OUT / f"veo_{sid}_in.wav"); voice.extract_audio(clip, wav)
            try:
                andy = voice.andy_voice_change(wav, str(OUT / f"veo_{sid}_andy.mp3"))
                clip2 = str(OUT / f"veo_{sid}_vc.mp4"); composite.replace_audio(clip, andy, clip2); clip = clip2
            except Exception as e:
                log(f"    ⚠ {sid} voice-change failed, keeping Veo audio ({str(e)[-80:]})")
        n = str(OUT / f"veo_norm_{sid}.mp4"); composite.normalize(clip, n); norm.append(n)
        log(f"  ✓ {sid}")

    log("VEO 3/3 — sequence")
    out = str(OUT / f"veo_final_{topic.replace(' ', '_')}.mp4")
    composite.concat(norm, out)
    log(f"=== VEO DONE -> {out} ===")
    return out


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or json.loads((OUT / "script.json").read_text()).get("_topic", "render")
    run(topic)
