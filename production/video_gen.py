"""
Image-to-video via the self-hosted Wan 2.2 I2V A14B (FP8 MoE) + LightX2V 4-step Lightning
worker fleet (ComfyUI). Each box runs ComfyUI on :8188, reached over an SSM tunnel.
BONNIE_WAN_ENDPOINTS (comma-separated) lists the worker base URLs; jobs round-robin /
least-busy across them. Wan is start-frame-only (no end frame); end_img is ignored.
"""
import os
import json
import time
import itertools
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

import config

ENDPOINTS = [e.strip() for e in os.environ.get("BONNIE_WAN_ENDPOINTS", "http://localhost:9010").split(",") if e.strip()]
_rr = itertools.cycle(ENDPOINTS)
_LORAS = {}  # base -> (high_lora_name, low_lora_name), resolved from each box's object_info


def _lora_names(base):
    """Resolve this box's Lightning LoRA filenames (setup script names them high_lightning/low_lightning)."""
    if base not in _LORAS:
        enum = _api(base, "/object_info/LoraLoaderModelOnly")["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0]
        hi = next((x for x in enum if "high_lightning" in x), "wan22_i2v_high_lightning.safetensors")
        lo = next((x for x in enum if "low_lightning" in x), "wan22_i2v_low_lightning.safetensors")
        _LORAS[base] = (hi, lo)
    return _LORAS[base]


def free_all():
    """Unload models + free VRAM on every Wan worker (so LatentSync's ~18GB fits). The next
    clip job reloads cold (~10s). Used for the two-phase LatentSync lip-sync pass."""
    body = json.dumps({"unload_models": True, "free_memory": True}).encode()
    for e in ENDPOINTS:
        try:
            req = urllib.request.Request(e + "/free", data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
        except Exception:
            pass


def _frames(dur_s: float) -> int:
    n = max(5, round(dur_s * config.LTX_FPS))
    return 4 * round((n - 1) / 4) + 1  # Wan VAE + Lightning: (4*K)+1


def _api(base, path, data=None, ctype="application/json", timeout=60):
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": ctype} if data else {})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{base}{path} -> {e.code}: {e.read().decode()[:500]}") from None


def _upload(base, img: Path) -> str:
    b = "----bonnie"; body = (
        f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{img.name}\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n").encode() + img.read_bytes() + f"\r\n--{b}--\r\n".encode()
    return _api(base, "/upload/image", body, f"multipart/form-data; boundary={b}")["name"]


def _least_busy() -> str:
    """Pick the endpoint with the fewest queued+running jobs; fall back to round-robin."""
    best, best_load = None, 1e9
    for e in ENDPOINTS:
        try:
            q = _api(e, "/prompt", timeout=8)  # {exec_info:{queue_remaining:N}}
            load = q.get("exec_info", {}).get("queue_remaining", 0)
            if load < best_load:
                best, best_load = e, load
        except Exception:
            continue
    return best or next(_rr)


def _workflow(image_name, prompt, w, h, length, seed, lora_hi, lora_lo, end_name=None):
    p = config.VIDEO_PARAMS
    # start+end frames -> WanFirstLastFrameToVideo (confirmed works with 2.2 i2v + Lightning);
    # start-only -> WanImageToVideo.
    if end_name:
        node10 = {"class_type": "WanFirstLastFrameToVideo", "inputs": {"positive": ["7", 0], "negative": ["8", 0], "vae": ["6", 0], "width": w, "height": h, "length": length, "batch_size": 1, "start_image": ["9", 0], "end_image": ["16", 0]}}
        end_node = {"16": {"class_type": "LoadImage", "inputs": {"image": end_name}}}
    else:
        node10 = {"class_type": "WanImageToVideo", "inputs": {"positive": ["7", 0], "negative": ["8", 0], "vae": ["6", 0], "width": w, "height": h, "length": length, "batch_size": 1, "start_image": ["9", 0]}}
        end_node = {}
    return {**end_node,
      "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "fp8_e4m3fn"}},
      "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "fp8_e4m3fn"}},
      "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora_hi, "strength_model": p["lora_strength"]}},
      "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": lora_lo, "strength_model": p["lora_strength"]}},
      "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
      "6": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
      "7": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["5", 0]}},
      "8": {"class_type": "CLIPTextEncode", "inputs": {"text": p["negative_prompt"], "clip": ["5", 0]}},
      "9": {"class_type": "LoadImage", "inputs": {"image": image_name}},
      "10": node10,
      "11": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["3", 0], "add_noise": "enable", "noise_seed": seed, "steps": p["steps"], "cfg": p["cfg"], "sampler_name": "euler", "scheduler": "simple", "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["10", 2], "start_at_step": 0, "end_at_step": p["split"], "return_with_leftover_noise": "enable"}},
      "12": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["4", 0], "add_noise": "disable", "noise_seed": seed, "steps": p["steps"], "cfg": p["cfg"], "sampler_name": "euler", "scheduler": "simple", "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["11", 0], "start_at_step": p["split"], "end_at_step": p["steps"], "return_with_leftover_noise": "disable"}},
      "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["6", 0]}},
      "14": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["13", 0], "frame_rate": config.LTX_FPS, "loop_count": 0, "filename_prefix": "bonnie/clip", "format": "video/h264-mp4", "pingpong": False, "save_output": True}},
    }


def generate(prompt: str, start_img: str, out_path: str, end_img: str | None = None,
             dur_s: float = 4.0, poll=5, timeout_s=1200, overrides: dict | None = None) -> str:
    """Submit one clip to a Wan worker, wait, download the mp4. Returns out_path.
    end_img (optional) -> first-last-frame conditioning. overrides may carry seed."""
    p = config.VIDEO_PARAMS
    o = overrides or {}
    seed = o.get("seed", p["seed"])
    w, h = o.get("width", p["width"]), o.get("height", p["height"])  # per-call res (e.g. fast teaser)
    base = _least_busy()
    lora_hi, lora_lo = _lora_names(base)
    name = _upload(base, Path(start_img))
    end_name = _upload(base, Path(end_img)) if end_img else None
    wf = _workflow(name, prompt, w, h, _frames(dur_s), seed, lora_hi, lora_lo, end_name=end_name)
    pid = _api(base, "/prompt", json.dumps({"prompt": wf}).encode())["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        h = _api(base, f"/history/{pid}", timeout=30)
        if pid in h:
            if h[pid].get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"wan clip error: {json.dumps(h[pid])[:300]}")
            files = [it for node in h[pid].get("outputs", {}).values() for v in node.values()
                     if isinstance(v, list) for it in v if isinstance(it, dict) and it.get("filename")]
            if files:
                f = files[-1]
                q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
                with urllib.request.urlopen(f"{base}/view?{q}", timeout=120) as r:
                    Path(out_path).write_bytes(r.read())
                return out_path
        time.sleep(poll)
    raise TimeoutError(f"wan clip {pid} timed out after {timeout_s}s")
