"""
Give-It-To-Bonnie production pipeline.

    python pipeline.py "celiac disease"

One prompt -> script (Gemini) -> images (Nano Banana 2) -> clips (LTX fleet) ->
Andy voice-change / toy TTS / closing VO (ElevenLabs + Gemini) -> ffmpeg composite
-> output/final_<topic>.mp4
"""
import sys
import json
import os
import time
import traceback
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import subprocess

import config
import script_brain
import gemini
import veo as _veo
import video_gen
import voice
import composite
import wan_lipsync
import db
import supa

OUT = config.OUTPUT
OUT.mkdir(exist_ok=True)

_RUN_ID = None  # set by run()/render_subset(); read by the per-scene generators below


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _audio_rms(path: str) -> float:
    """Mean volume in dB (negative). Returns -99 on error or silence."""
    r = subprocess.run(["ffmpeg", "-i", path, "-af", "volumedetect", "-f", "null", "/dev/null"],
                       capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "mean_volume:" in line:
            try:
                return float(line.split("mean_volume:")[-1].strip().split()[0])
            except ValueError:
                pass
    return -99.0


class _Safe(dict):
    def __missing__(self, k):
        return ""


def fill(template: str, script: dict) -> str:
    return template.format_map(_Safe(script))


def resolve(name: str) -> str:
    """Prefer a generated frame in output/, else the raw asset."""
    g = OUT / name
    return str(g if g.exists() else config.ASSETS / name)


def _gen_image(sid, spec, script):
    inputs = [resolve(n) for n in spec["inputs"]]
    prompt = fill(spec["prompt"], script)
    out = str(OUT / spec["output"])
    log(f"  {sid}: {spec['output']}  <- {prompt[:70]}")
    try:
        gemini.generate_image(config.GEMINI_IMAGE_MODEL, prompt, inputs, out, grounding=True)
    except Exception as e:
        db.log_prompt(_RUN_ID, sid, "image", prompt, model=config.GEMINI_IMAGE_MODEL, error=str(e)[:400])
        raise
    db.log_prompt(_RUN_ID, sid, "image", prompt, model=config.GEMINI_IMAGE_MODEL, output_path=out)


def stage_images(script, scenes=None):
    scenes = scenes or config.SCENES
    log("STAGE 1/4 — images (Nano Banana 2, dependency-parallel)")
    from concurrent.futures import as_completed
    specs = [(sc["id"], spec) for sc in scenes
             for spec in (sc.get("image"), (sc.get("overlay") or {}).get("image")) if spec]
    gen_outputs = {spec["output"] for _, spec in specs}  # files produced by *this* stage
    def done(spec): p = OUT / spec["output"]; return p.exists() and p.stat().st_size > 5000
    def ready(spec):  # all generated-image inputs already exist (raw assets always do)
        return all((OUT / n).exists() for n in spec["inputs"] if n in gen_outputs)
    for sid, spec in specs:
        if done(spec): log(f"  ↻ reuse {spec['output']}")
    pending = [(sid, spec) for sid, spec in specs if not done(spec)]
    while pending:
        wave = [(sid, spec) for sid, spec in pending if ready(spec)]
        if not wave:
            raise RuntimeError(f"image deps unsatisfiable: {[s['output'] for _, s in pending]}")
        with ThreadPoolExecutor(max_workers=min(len(wave), 6)) as ex:
            futs = {ex.submit(_gen_image, sid, spec, script): spec for sid, spec in wave}
            for fut in as_completed(futs):
                fut.result()  # raise on failure
        pending = [(sid, spec) for sid, spec in pending if not done(spec)]


# Scenes that use Wan: the 4.5 reveal overlay (first-last-frame, detected by end!=None)
# and the closing trio (muted, motion-only — Wan is faster/cheaper for these).
_WAN_SCENES = {"scene8", "scene9", "scene10"}


def _gen_clip(job):
    out = job["out"]
    # First-last-frame overlay (scene4_ov has end set) and closing muted scenes → Wan
    if job.get("end") or job["id"] in _WAN_SCENES:
        result = video_gen.generate(
            job["prompt"], job["start"], out,
            end_img=job.get("end"), dur_s=job["dur"],
            overrides=job.get("overrides"))
        db.log_prompt(_RUN_ID, job["id"], "video", job["prompt"], model="wan", output_path=out)
        return job["id"], result
    # All dialogue/action scenes → Veo 3.1 Lite; fall back to Wan if Veo is filtered/fails
    raw = out.replace(".mp4", "_vraw.mp4")
    try:
        _veo.generate_video(job["prompt"], job["start"], raw, dur=job["dur"])
        subprocess.run(["ffmpeg", "-y", "-i", raw, "-an", "-c:v", "copy", out],
                       check=True, capture_output=True)
        # Preserve native audio for ElevenLabs STS in stage 3 (Andy voice-change)
        subprocess.run(["ffmpeg", "-y", "-i", raw, "-vn", "-ac", "1", "-ar", "44100",
                        out.replace(".mp4", "_veo_audio.wav")],
                       check=True, capture_output=True)
        db.log_prompt(_RUN_ID, job["id"], "video", job["prompt"], model="veo-3.1-lite", output_path=out)
    except Exception as e:
        log(f"  ⚠ {job['id']} Veo failed ({str(e)[-120:]}) → Wan fallback")
        db.log_prompt(_RUN_ID, job["id"], "video", job["prompt"], model="veo-3.1-lite", error=str(e)[:400])
        video_gen.generate(job["prompt"], job["start"], out, dur_s=job["dur"])
        db.log_prompt(_RUN_ID, job["id"], "video", job["prompt"], model="wan", output_path=out)
    return job["id"], out


def stage_videos(script, scenes=None):
    scenes = scenes or config.SCENES
    log(f"STAGE 2/4 — clips (Veo 3.1 for scenes 1-7, Wan for 4.5-overlay + scenes 8-10)")
    jobs = []
    for sc in scenes:
        v = sc["video"]
        jobs.append(dict(id=sc["id"], prompt=fill(v["prompt"], script),
                         start=resolve(v["start"]), end=resolve(v["end"]) if v["end"] else None,
                         dur=v["dur"], out=str(OUT / f"raw_{sc['id']}.mp4"),
                         overrides={k: v[k] for k in ("steps", "cfg", "stg", "seed") if k in v}))
        ov = (sc.get("overlay") or {}).get("video")
        if ov:
            jobs.append(dict(id=f"{sc['id']}_ov", prompt=fill(ov["prompt"], script),
                             start=resolve(ov["start"]), end=resolve(ov["end"]) if ov["end"] else None,
                             dur=ov["dur"], out=str(OUT / f"raw_{sc['id']}_ov.mp4")))
    # resume: reuse any clip already rendered
    results, pending = {}, []
    for j in jobs:
        p = Path(j["out"])
        if p.exists() and p.stat().st_size > 10000:
            results[j["id"]] = j["out"]; log(f"  ↻ reuse {j['id']}")
        else:
            pending.append(j)
    from concurrent.futures import as_completed
    if pending:
        with ThreadPoolExecutor(max_workers=min(len(pending), 5)) as ex:
            for fut in as_completed([ex.submit(_gen_clip, j) for j in pending]):
                jid, path = fut.result(); results[jid] = path; log(f"  ✓ {jid}")
    return results


def _audio_one(sc, script, clips):
    """One scene's audio + composite, end to end (independent of other scenes)."""
    sid = sc["id"]
    fin = OUT / f"final_{sid}.mp4"
    if fin.exists() and fin.stat().st_size > 10000:
        return sid, str(fin), True
    work = clips[sid]
    muted = sc.get("muted", False)

    # 1) dialogue — TTS the line, then lip-sync (Wav2Lip) or overlay onto the silent Wan clip.
    a = sc["audio"]
    start_s = sc.get("audio_start_s", 0)  # delay the line to land later in the shot
    if a in ("andy", "bonnie"):
        line = script.get(f"{sid}_line", "")
        work2 = str(OUT / f"{sid}_a.mp4")
        if a == "andy":
            veo_audio = str(OUT / f"raw_{sid}_veo_audio.wav")
            if Path(veo_audio).exists():
                try:
                    aud = voice.andy_voice_change(veo_audio, str(OUT / f"{sid}_v.mp3"))
                    rms = _audio_rms(aud)
                    if rms < -50:
                        raise RuntimeError(f"STS near-silent ({rms:.0f} dB) — Veo had no speech")
                    composite.replace_audio(work, aud, work2)
                    work = work2
                except Exception as e:
                    log(f"  ⚠ {sid} STS failed → TTS fallback ({str(e)[-80:]})")
                    if line:
                        aud = voice.andy_tts(line, str(OUT / f"{sid}_v.mp3"))
                        db.log_prompt(_RUN_ID, sid, "audio", line, model="andy_tts", output_path=aud)
                        composite.overlay_audio_at(work, aud, work2, start_s=start_s)
                        work = work2
            elif line:
                aud = voice.andy_tts(line, str(OUT / f"{sid}_v.mp3"))
                db.log_prompt(_RUN_ID, sid, "audio", line, model="andy_tts", output_path=aud)
                composite.overlay_audio_at(work, aud, work2, start_s=start_s)
                work = work2
        elif line:
            aud = voice.bonnie_tts(line, str(OUT / f"{sid}_v.wav"))
            db.log_prompt(_RUN_ID, sid, "audio", line, model="bonnie_tts", output_path=aud)
            if start_s == 0 and config.LIPSYNC in ("wav2lip", "latentsync"):
                box = sc.get("lipsync_crop")
                try:
                    if box:
                        cr = str(OUT / f"{sid}_crop.mp4"); composite.crop_region(work, cr, *box)
                        crl = str(OUT / f"{sid}_crop_ls.mp4"); wan_lipsync.lipsync(cr, aud, crl)
                        composite.paste_region(work, crl, work2, box[0], box[1])
                    else:
                        wan_lipsync.lipsync(work, aud, work2)
                except Exception as e:
                    log(f"  ⚠ {sid} lip-sync failed → overlay fallback ({str(e)[-90:]})")
                    composite.overlay_audio_at(work, aud, work2, start_s=0)
            else:
                composite.overlay_audio_at(work, aud, work2, start_s=start_s)
            work = work2
    elif a == "tts":
        t = sc["tts"]
        try:
            toy_text = fill("{" + t["text_key"] + "}", script)
            wav = voice.toy_tts(toy_text,
                                fill("{" + t["style_key"] + "}", script), str(OUT / f"{sid}_toy.wav"))
            db.log_prompt(_RUN_ID, sid, "audio", toy_text, model="toy_tts", output_path=wav)
            base = str(OUT / f"{sid}_voice.mp4")
            if t.get("at") == "end":
                composite.overlay_audio_end(work, wav, base)
            else:
                composite.overlay_audio_at(work, wav, base, start_s=t.get("start_s", 0))
        except Exception as e:
            log(f"  ⚠ {sid} toy TTS failed → SFX-only fallback ({str(e)[-80:]})")
            base = work  # skip toy voice, keep raw clip
        sfx = sc.get("sfx")
        if sfx:
            work2 = str(OUT / f"{sid}_a.mp4")
            composite.mix_audio_at(base, str(config.ASSETS / sfx["file"]), work2,
                                   start_s=sfx.get("start_s", 0), vol=sfx.get("vol", 1.0))
            work = work2
        else:
            work = base

    # 2) cutaway overlay (scene1 -> 1.5 asset; scene4 -> 4.5 generated clip)
    ov = sc.get("overlay")
    if ov and ov.get("clip"):
        ins = resolve(ov["clip"]); work2 = str(OUT / f"{sid}_ov.mp4")
        composite.cutaway(work, ins, work2, ov["dur"]); work = work2
    elif ov and ov.get("video"):
        ins = clips[f"{sid}_ov"]; work2 = str(OUT / f"{sid}_ov.mp4")
        composite.cutaway(work, ins, work2, ov["dur"]); work = work2

    # 3) trim tail + normalize
    if sc["video"].get("trim_end"):
        work2 = str(OUT / f"{sid}_t.mp4"); composite.trim_tail(work, work2, sc["video"]["trim_end"]); work = work2
    norm = str(OUT / f"final_{sid}.mp4"); composite.normalize(work, norm, muted=muted)
    return sid, norm, False


def stage_audio_composite(script, clips, scenes=None):
    scenes = scenes or config.SCENES
    mode = {"wav2lip": "Wav2Lip lip-sync", "latentsync": "LatentSync lip-sync"}.get(config.LIPSYNC, "audio overlay")
    log(f"STAGE 3/4 — TTS + {mode} + per-scene composite (parallel)")
    from concurrent.futures import as_completed
    nw = len(wan_lipsync.ENDPOINTS) if config.LIPSYNC == "wav2lip" else 6
    scene_finals = {}
    sc_by_fut = {}
    with ThreadPoolExecutor(max_workers=max(2, min(len(scenes), nw))) as ex:
        for sc in scenes:
            sc_by_fut[ex.submit(_audio_one, sc, script, clips)] = sc
        for fut in as_completed(sc_by_fut):
            sc = sc_by_fut[fut]
            try:
                sid, path, reused = fut.result()
                scene_finals[sid] = path
                log(f"  {'↻ reuse' if reused else '✓'} {sid}")
            except Exception as e:
                sid = sc["id"]
                log(f"  ✗ {sid} audio failed → using raw clip ({str(e)[-120:]})")
                scene_finals[sid] = clips[sid]  # fall back to silent raw clip
    return scene_finals


def stage_stitch(script, scene_finals, topic):
    log("STAGE 4/4 — stitch + closing voiceover + global music bed")
    main = [scene_finals[s["id"]] for s in config.SCENES if s["id"] not in config.CLOSING_SEQUENCE]
    closing_clips = [scene_finals[s] for s in config.CLOSING_SEQUENCE]
    vo = voice.andy_tts(script["closing_voiceover"], str(OUT / "closing_vo.mp3"))
    db.log_prompt(_RUN_ID, "closing", "audio", script["closing_voiceover"], model="andy_tts", output_path=vo)
    closing = str(OUT / "closing.mp4")
    composite.closing_vo(closing_clips, vo, closing)          # VO only — music is global now
    body = str(OUT / "_body.mp4")
    composite.concat(main + [closing], body)
    scored = str(OUT / "_scored.mp4")
    composite.under_music(body, str(config.ASSETS / config.BG_MUSIC), scored)  # one bed, whole video
    final = str(OUT / f"final_{topic.replace(' ', '_')}.mp4")
    composite.watermark(scored, final)
    return final


def _load_or_write_script(topic):
    sj = OUT / "script.json"
    if sj.exists():
        script = json.loads(sj.read_text())
        if script.get("_topic") == topic:
            log("↻ reusing output/script.json"); return script
    script = script_brain.write_script(topic)
    sj.write_text(json.dumps(script, indent=2, ensure_ascii=False))
    log("script written -> output/script.json")
    return script


def render_subset(topic, scene_ids):
    """Render images+videos+audio for just `scene_ids` (reuses anything already on disk),
    returning (script, {sid: final_clip}). Used for the free chapter (scenes 1-2)."""
    global _RUN_ID
    _RUN_ID = db.start_run(f"{topic} (preview)")
    try:
        script = _load_or_write_script(topic)
        scenes = [s for s in config.SCENES if s["id"] in scene_ids]
        stage_images(script, scenes)
        clips = stage_videos(script, scenes)
        if config.LIPSYNC == "latentsync":
            video_gen.free_all()
        finals = stage_audio_composite(script, clips, scenes)
        db.finish_run(_RUN_ID)
        return script, finals
    except Exception as e:
        db.finish_run(_RUN_ID, status=f"error: {str(e)[:200]}")
        raise


def stitch_chapter(finals, scene_ids, out):
    """Stitch a mid-video chapter (no closing VO) with the global music bed, trimmed."""
    clips = [finals[s] for s in scene_ids if s in finals]
    body = str(OUT / "_chapter_body.mp4"); composite.concat(clips, body)
    composite.under_music(body, str(config.ASSETS / config.BG_MUSIC), out)
    return out


_GLOBAL_INTERMEDIATES = [
    "closing_vo.mp3", "closing.mp4", "_body.mp4", "_scored.mp4",
    "_closing_seq.mp4", "_closing_bed.m4a", "_watermark.png",
]


def _cleanup_intermediates(keep: str):
    """Delete this run's per-scene working files (raw clips, voice swaps, crops, per-scene
    finals) and the global stitch scratch files, now that the final video is uploaded to
    S3/Supabase. `keep` (the final_<topic>.mp4 path) is left alone — local/dev delivery
    still serves it straight off disk. Without this, output/ only ever grows (root cause
    of the Render disk filling up)."""
    keep_path = Path(keep).resolve()
    patterns = [f"{sc['id']}_*" for sc in config.SCENES]
    patterns += [f"raw_{sc['id']}*" for sc in config.SCENES]
    patterns += [f"final_{sc['id']}.mp4" for sc in config.SCENES]
    for pat in patterns:
        for f in OUT.glob(pat):
            if f.resolve() != keep_path:
                try:
                    f.unlink()
                except OSError:
                    pass
    for name in _GLOBAL_INTERMEDIATES:
        try:
            (OUT / name).unlink()
        except OSError:
            pass


def run(topic: str):
    global _RUN_ID
    log(f"=== GIVE IT TO BONNIE: '{topic}' ===")
    _RUN_ID = db.start_run(topic)
    try:
        script = _load_or_write_script(topic)
        stage_images(script)
        clips = stage_videos(script)
        if config.LIPSYNC == "latentsync":
            log("  freeing Wan VRAM on workers (two-phase LatentSync)…")
            video_gen.free_all()  # LatentSync needs the GPU to itself
        finals = stage_audio_composite(script, clips)
        out = stage_stitch(script, finals, topic)
        # Mirror to Supabase Storage so finished videos are browsable later (best-effort, never blocks delivery)
        if supa.enabled():
            try:
                dest = f"{topic.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.mp4"
                video_url = supa.upload_video(out, dest)
                supa.insert_video(topic, video_url, run_id=_RUN_ID)
                log(f"  mirrored to Supabase -> {video_url}")
            except Exception as e:
                log(f"  ⚠ Supabase video upload failed: {str(e)[:200]}")
        # Upload to S3 if configured; return presigned URL so remote callers get a playable link
        bucket = os.environ.get("BONNIE_S3_BUCKET")
        if bucket:
            import boto3
            s3 = boto3.client("s3")
            key = f"videos/{topic.replace(' ', '_')}_{uuid.uuid4().hex[:8]}/final.mp4"
            s3.upload_file(out, bucket, key, ExtraArgs={"ContentType": "video/mp4"})
            url = s3.generate_presigned_url("get_object",
                Params={"Bucket": bucket, "Key": key}, ExpiresIn=86400 * 7)
            db.finish_run(_RUN_ID, final_path=url)
            _cleanup_intermediates(out)
            try:
                Path(out).unlink()  # already in S3 — landing.py serves the URL, not this path
            except OSError:
                pass
            log(f"=== DONE -> {url} ===")
            return url
        db.finish_run(_RUN_ID, final_path=out)
        _cleanup_intermediates(out)
        log(f"=== DONE -> {out} ===")
        return out
    except Exception as e:
        db.finish_run(_RUN_ID, status=f"error: {str(e)[:200]}")
        raise


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "celiac disease"
    try:
        run(topic)
    except Exception:
        traceback.print_exc(); sys.exit(1)
