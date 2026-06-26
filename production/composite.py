"""ffmpeg compositing: trims, audio swaps, cutaway overlays, concat, closing bed."""
import json
import subprocess
from pathlib import Path

W, H, FPS, AR = 1280, 704, 24, 44100


def _run(args):
    r = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + r.stderr[-800:])


def dur(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", path], capture_output=True, text=True).stdout
    return float(json.loads(out)["format"]["duration"])


def normalize(inp: str, out: str, muted=False):
    """Standardize one clip to common codec/size/fps (+ silent track if muted/none)."""
    if muted:
        _run(["-i", inp, "-f", "lavfi", "-i", f"anullsrc=r={AR}:cl=mono",
              "-map", "0:v", "-map", "1:a", "-vf", f"scale={W}:{H},fps={FPS}",
              "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", str(AR),
              "-shortest", out])
    else:
        _run(["-i", inp, "-vf", f"scale={W}:{H},fps={FPS}", "-c:v", "libx264",
              "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", str(AR), out])


def trim_tail(inp: str, out: str, secs: float):
    if secs <= 0:
        _run(["-i", inp, "-c", "copy", out]); return
    _run(["-i", inp, "-t", f"{max(0.1, dur(inp) - secs):.3f}", "-c:v", "libx264",
          "-pix_fmt", "yuv420p", "-c:a", "aac", out])


def replace_audio(video: str, audio: str, out: str):
    """Swap the video's audio for `audio` (cut to video length)."""
    _run(["-i", video, "-i", audio, "-map", "0:v", "-map", "1:a",
          "-c:v", "copy", "-c:a", "aac", "-ar", str(AR), "-shortest", out])


def mix(a: str, b: str, out: str, vol_a=1.0, vol_b=1.0):
    """Mix two audio files (longest wins)."""
    _run(["-i", a, "-i", b, "-filter_complex",
          f"[0:a]volume={vol_a}[x];[1:a]volume={vol_b}[y];[x][y]amix=inputs=2:duration=longest:normalize=0[o]",
          "-map", "[o]", "-ar", str(AR), out])


def overlay_audio_at(video: str, audio: str, out: str, start_s: float = 0.0, vol=1.0):
    """Lay `audio` onto the (silent) Wan clip starting at start_s. The clip has no audio
    track, so the delayed TTS becomes the sole audio; output keeps the full video length."""
    ms = int(start_s * 1000)
    _run(["-i", video, "-i", audio, "-filter_complex",
          f"[1:a]adelay={ms}|{ms},volume={vol}[d]",
          "-map", "0:v", "-map", "[d]", "-c:v", "copy", "-c:a", "aac", out])


def mix_audio_at(video: str, audio: str, out: str, start_s: float = 0.0, vol=1.0):
    """Mix `audio` ON TOP of the video's EXISTING audio, starting at start_s (keeps the
    original track — used to drop one spoken word into the intro at the 8s mark)."""
    ms = int(start_s * 1000)
    _run(["-i", video, "-i", audio, "-filter_complex",
          f"[1:a]adelay={ms}|{ms},volume={vol}[w];[0:a][w]amix=inputs=2:duration=first:normalize=0[o]",
          "-map", "0:v", "-map", "[o]", "-c:v", "copy", "-c:a", "aac", out])


def overlay_audio_end(video: str, audio: str, out: str, pad: float = 0.12):
    """Overlay `audio` so it FINISHES ~pad seconds before the video ends (lands at the end,
    regardless of the line's length). Replaces the clip's audio with the delayed line."""
    start = max(0.0, dur(video) - dur(audio) - pad)
    overlay_audio_at(video, audio, out, start_s=start)
    return out


def crop_region(inp: str, out: str, x: float, y: float, w: float, h: float):
    """Crop a normalized region (fractions of frame) — to isolate one speaker's face
    before lip-sync, so the lip-sync engine can't grab the wrong character."""
    _run(["-i", inp, "-vf", f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", out])


def paste_region(base: str, region: str, out: str, x: float, y: float):
    """Overlay the lip-synced region back onto the (silent) base clip at (x,y) fractions,
    and take the region's audio (the TTS that drove the lip-sync)."""
    _run(["-i", base, "-i", region, "-filter_complex",
          f"[0:v][1:v]overlay=W*{x}:H*{y}[v]",
          "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-shortest", out])


def cutaway(base: str, insert: str, out: str, win: float):
    """Replace the middle `win` seconds of base's VIDEO with the first `win` of insert;
    keep base's audio continuous."""
    d = dur(base)
    win = min(win, d)
    start = max(0.0, (d - win) / 2)
    end = start + win
    cond = f"scale={W}:{H},fps={FPS},setsar=1,format=yuv420p,settb=AVTB"
    _run(["-i", base, "-i", insert, "-filter_complex",
          f"[0:v]split=2[b1][b2];"
          f"[b1]trim=0:{start:.3f},setpts=PTS-STARTPTS,{cond}[v0];"
          f"[1:v]trim=0:{win:.3f},setpts=PTS-STARTPTS,{cond}[v1];"
          f"[b2]trim={end:.3f}:{d:.3f},setpts=PTS-STARTPTS,{cond}[v2];"
          f"[v0][v1][v2]concat=n=3:v=1:a=0[v]",
          "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-shortest", out])


def closing_vo(clips: list[str], voiceover: str, out: str, vo_vol=1.0):
    """Concat the muted closing scenes and lay ONLY the Andy voiceover under them
    (background music is now applied globally over the whole video, not just here)."""
    seq = str(Path(out).parent / "_closing_seq.mp4")
    concat(clips, seq)
    total = dur(seq)
    bed = str(Path(out).parent / "_closing_bed.m4a")
    _run(["-i", voiceover, "-filter_complex", f"[0:a]volume={vo_vol},apad[v]",
          "-map", "[v]", "-ar", str(AR), "-t", f"{total:.3f}", bed])
    replace_audio(seq, bed, out)


def under_music(video: str, music: str, out: str, vol=0.28):
    """Lay a background music bed UNDER the video's existing audio (dialogue/VO), trimmed
    to the video length — one continuous track for the whole clip, cut off at the end."""
    total = dur(video)
    _run(["-i", video, "-i", music, "-filter_complex",
          f"[1:a]volume={vol},atrim=0:{total:.3f}[m];"
          f"[0:a][m]amix=inputs=2:duration=first:normalize=0[o]",
          "-map", "0:v", "-map", "[o]", "-c:v", "copy", "-c:a", "aac", "-ar", str(AR), out])


def concat(clips: list[str], out: str):
    """Concat clips, re-conditioning each segment in-graph so SAR/fps/tb all match."""
    n = len(clips)
    args = []
    for c in clips:
        args += ["-i", c]
    pre = ";".join(
        f"[{i}:v]scale={W}:{H},fps={FPS},setsar=1,format=yuv420p,settb=AVTB[v{i}];"
        f"[{i}:a]aresample={AR},asetpts=N/SR/TB[a{i}]" for i in range(n))
    streams = "".join(f"[v{i}][a{i}]" for i in range(n))
    _run([*args, "-filter_complex", f"{pre};{streams}concat=n={n}:v=1:a=1[v][a]",
          "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-ar", str(AR), out])


def closing_bed(clips: list[str], voiceover: str, music: str, out: str,
                music_vol=0.35, vo_vol=1.0):
    """Concat the muted scenes 8-10 and lay (voiceover + bg music) under them."""
    seq = str(Path(out).parent / "_closing_seq.mp4")
    concat(clips, seq)
    total = dur(seq)
    bed = str(Path(out).parent / "_closing_bed.m4a")
    # music looped/cut to length + voiceover from the top
    _run(["-i", music, "-i", voiceover, "-filter_complex",
          f"[0:a]volume={music_vol},atrim=0:{total:.3f}[m];"
          f"[1:a]volume={vo_vol}[v];[m][v]amix=inputs=2:duration=first:normalize=0[o]",
          "-map", "[o]", "-ar", str(AR), "-t", f"{total:.3f}", bed])
    replace_audio(seq, bed, out)
