"""
Give-It-To-Bonnie — production pipeline configuration.

Encodes the full 10-scene spec as data. One user prompt (the thing being "given to
Bonnie") -> Gemini writes a structured SCRIPT -> per-scene image gen (Nano Banana 2)
-> per-scene video gen (LTX-2.3 Pro) -> Andy voice-change / toy TTS (ElevenLabs +
Gemini TTS) -> ffmpeg compositing (overlays, trims, audio swaps, background music)
-> one stitched final video.

This file is the single source of truth for *what* each scene does. The orchestrator
(pipeline.py) reads it and runs the steps. {customization}/{custom} placeholders are
filled from the Gemini-written SCRIPT (see SCRIPT_SCHEMA below).
"""
import os
from pathlib import Path

# Dialogue handling for talking shots: "overlay" = lay TTS on the silent Wan clip (Wan's
# generic mouth motion); "wav2lip" = drive the mouth to the TTS via the box-side Wav2Lip service.
LIPSYNC = os.environ.get("BONNIE_LIPSYNC", "overlay")

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
OUTPUT = Path(os.environ.get("BONNIE_OUTPUT", str(ROOT / "output")))

# ── Models / providers ──────────────────────────────────────────────────────
GEMINI_TEXT_MODEL = "gemini-3.1-pro-preview"        # script writer (the "brain")
GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"       # "Nano Banana 2" (image-search grounding)
GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"   # toy pull-string voice (scene 7)

IMAGE_PARAMS = dict(aspect_ratio="16:9", resolution="1k", thinking_level="high",
                    grounding=["google_search", "google_image_search"])

# Wan 2.2 I2V A14B (MoE, FP8) + LightX2V 4-step Lightning — self-hosted ComfyUI worker fleet.
# (Migrated off LTX-2.3: Wan is higher fidelity. Start+end frames via the WanFirstLastFrameToVideo
#  node when a scene sets `end`; dialogue is ElevenLabs/Gemini TTS — overlaid or Wav2Lip lip-synced.)
VIDEO_PARAMS = dict(width=1280, height=720, steps=4, split=2, cfg=1.0, seed=0,
                    lora_strength=1.0,
                    negative_prompt="blurry, distorted, low quality, watermark, text, deformed, extra limbs, jpeg artifacts")
LTX_FPS = 16  # Wan A14B runs at 16 fps; num_frames snapped to (4K)+1

# ElevenLabs "Andy" voice change (speech-to-speech on whole clip)
ELEVEN_ANDY = dict(voice_id="UkjpxcmWHdgSfY46JwUs",
                   model="eleven_multilingual_v2",          # text-to-speech (closing voiceover)
                   sts_model="eleven_multilingual_sts_v2",  # speech-to-speech (Andy voice-change)
                   stability=1.0, similarity_boost=1.0, style=0.0)
# All Andy clips: the ElevenLabs Andy line REPLACES the clip's audio outright.
# (Vocal-isolation/SFX-keep was removed — the negative prompt already suppresses
# generated music/singing, so the clip audio is essentially just the spoken line.)
ANDY_AUDIO_SWAP = True

# One continuous background music bed under the ENTIRE video (continues the intro clip's
# music). Mixed under all dialogue/VO and trimmed to the length of the generated clips.
BG_MUSIC = "Bonnie_BGAudio.mp3"

# ── What Gemini must produce from the user's one-line prompt ─────────────────
# (filled per request; field names are referenced by the SCENES table below)
SCRIPT_SCHEMA = {
    "action_figure":     "the final punchline character made into a toy (e.g. for 'celiac' -> the waiter who was told about the celiac order)",
    "action_figure_short": "a ONE-word noun for that toy, used verbatim in image prompts (e.g. 'waiter', 'ninja') — no description, no adjectives",
    "scene1_item":       "first physical item Andy holds up (image)",
    "scene1_item_short": "a 1-2 word short noun for scene1_item, used verbatim in prompts (e.g. 'a neon green Speed Stacks cup' -> 'cup'); no adjectives",
    "scene1_line":       "what Andy says holding it (video)",
    "scene2_line":       "Andy's line as Bonnie runs over and grabs scene1_item",
    "scene3_items":      "the pair of items that are 'madly in love' (image swap)",
    "scene3_line":       "Andy's 'you gotta keep them together, they're madly in love' line",
    "scene4_line":       "Bonnie's line peeking over the box at the pile",
    "scene5_line":       "Andy's confused reaction lifting the action-figure toy",
    "scene6_line":       "Bonnie's excited quote pointing at the toy",
    "scene7_toy_line":   "the catchphrase the pull-string toy says, exactly ONCE (not repeated)",
    "scene7_tts_style":  "style/delivery instructions for the toy voice (Gemini TTS)",
    "scene8_image":      "custom childhood-bedroom still description (image)",
    "scene8_video":      "custom childhood-bedroom motion description (video, muted)",
    "scene9_outfit":     "themed outfit the boy wears (must match the toy)",
    "scene9_image":      "boy running down the street with toy on shoulders (image)",
    "scene9_video":      "boy running motion description (video, muted)",
    "closing_voiceover": "Andy's heartfelt+funny monologue played over scenes 8-10 (ElevenLabs)",
}

# ── The 10 scenes ────────────────────────────────────────────────────────────
# image.inputs: reference images fed to Nano Banana 2 (first = base to edit).
# *_Generated.jpg outputs are produced upstream and reused as inputs downstream.
# video: start_frame / end_frame are conditioning images; dur/trim in seconds.
# audio: andy=ElevenLabs voice-change on whole clip; tts=Gemini toy voice; none=muted/keep.
SCENES = [
    dict(id="scene1", name="Andy's first item",
         image=dict(inputs=["GiveBonnie_Scene1_Raw.jpg"],
                    prompt="make this guy holding {scene1_item}, and sharpen the animation quality",
                    output="GiveBonnie_Scene1_Generated.jpg"),
         video=dict(start="GiveBonnie_Scene1_Generated.jpg", end=None,
                    prompt="Guy says “{scene1_line}” 3D animation. Static shot.",
                    dur=4, trim_end=1),
         audio="andy",
         overlay=dict(clip="GiveBonnie_Scene1.5.mp4", at="middle", dur=1)),

    dict(id="scene2", name="Bonnie grabs it",
         image=dict(inputs=["GiveBonnie_Scene2_Raw.png", "GiveBonnie_Scene1_Generated.jpg"],
                    prompt="replace the cowgirl and horse toy with {scene1_item}, and sharpen the animation quality",
                    output="GiveBonnie_Scene2_Generated.jpg"),
         video=dict(start="GiveBonnie_Scene2_Generated.jpg", end=None,
                    prompt="Guy on the left says “{scene2_line}” as little girl runs over and grabs the {scene1_item}. 3D animation. Static shot.",
                    dur=4, trim_end=1),
         audio="andy"),

    dict(id="scene3", name="Madly in love",
         image=dict(inputs=["GiveBonnie_Scene3_Raw.jpg"],
                    prompt="Swap the rice flour and tapioca starch with {scene3_items}.",
                    output="GiveBonnie_Scene3_Generated.jpg"),
         video=dict(start="GiveBonnie_Scene3_Generated.jpg", end=None,
                    prompt="Guy faces forward and says “{scene3_line}”, holding the two items up and bringing them together. 3D animation. Static shot.",
                    dur=8, trim_end=1),
         audio="andy"),

    dict(id="scene4", name="Bonnie discovers",
         image=dict(inputs=["GiveBonnie_Scene4_Raw.png", "GiveBonnie_Scene1_Generated.jpg", "GiveBonnie_Scene3_Generated.jpg"],
                    prompt="Replace the toys on the ground with the {scene1_item}, and {scene3_items}. Sharpen animation quality.",
                    output="GiveBonnie_Scene4_Generated.jpg"),
         video=dict(start="GiveBonnie_Scene4_Generated.jpg", end=None,
                    prompt="The girl runs up and slowly peers down into the box looking inside, then at the very end cutely says “{scene4_line}”. 3D animation. Static shot.",
                    dur=5, trim_end=0),
         audio="bonnie", audio_start_s=3.6,  # quote lands at the END, after the box-reveal cutaway (silent while she runs up/looks)
         # 4.5 overlay: a separate image+video, cut to first 2s, placed over the middle 2s of scene4
         overlay=dict(
             image=dict(inputs=["GiveBonnie_Scene4.5_Raw.jpg"],
                        # "in same style" was making it keep the waiter outfit — describe the
                        # actual toy and lock POSE (not style); grounding pulls the real look.
                        prompt="Swap the waiter toy out for a plush {action_figure} mascot toy lying in the exact same pose and position in the box. It must clearly look like the real {action_figure} ({action_figure_short}) — a soft plush action figure, NOT a waiter, no apron or waiter outfit. Keep the same box, lighting, and pose.",
                        output="GiveBonnie_Scene4.5_Generated.jpg"),
             video=dict(start="GiveBonnie_Scene4.5_StartFrame.jpg", end="GiveBonnie_Scene4.5_Generated.jpg",
                        prompt="Camera slowly moves forward and down to reveal the {action_figure_short} toy laying in the box.",
                        dur=4, use_first_s=2),
             at="middle", dur=2)),

    dict(id=”scene5”, name=”Andy's confusion”,
         image=dict(inputs=[“GiveBonnie_Scene5_Raw.jpg”, “GiveBonnie_Scene4.5_Generated.jpg”],
                    prompt=”replace the waiter toy with the {action_figure_short} toy in same style with a pull ring on its back.”,
                    output=”GiveBonnie_Scene5_Generated.jpg”),
         video=dict(start=”GiveBonnie_Scene5_Generated.jpg”, end=None,
                    prompt=”Guy lifts up his toy and says “{scene5_line}”. 3D animation. Static shot.”,
                    dur=4, trim_end=0),
         audio=”andy”, audio_start_s=0.8),

    dict(id="scene6", name="Bonnie's quote",
         image=dict(inputs=["GiveBonnie_Scene4_Generated.jpg", "GiveBonnie_Scene5_Generated.jpg"],
                    prompt="make the guy on the left holding the {action_figure_short} toy, and make the girl on the right excited pointing at it, while the guy is straight faced. Sharpen animation quality.",
                    output="GiveBonnie_Scene6_Generated.jpg"),
         video=dict(start="GiveBonnie_Scene6_Generated.jpg", end=None,
                    prompt="The little girl on the right points and cutely says “{scene6_line}”, only her mouth moving as she speaks. The man on the left silently holds the {action_figure_short} toy, his mouth closed and still, not talking. 3D animation. Static shot.",
                    dur=4, trim_end=1),
         audio="bonnie",  # Bonnie quotes the catchphrase ({scene6_line}); Gemini child voice
         # two people in frame — isolate Bonnie (right side) so lip-sync targets her, not the guy
         lipsync_crop=(0.45, 0.10, 0.55, 0.90)),

    dict(id="scene7", name="Pull string",
         # reference image of the desired action-figure toy; prompt copied from scene5 (waiter -> Lamar Jackson)
         image=dict(inputs=["GiveBonnie_Scene7_Ref.jpg", "GiveBonnie_Scene4.5_Generated.jpg"],
                    prompt="replace the Lamar Jackson toy with the {action_figure_short} toy in same style with a pull ring on its back.",
                    output="GiveBonnie_Scene7_Generated.jpg"),
         video=dict(start="GiveBonnie_Scene7_Generated.jpg", end=None,
                    prompt="String retracts into the back of the toy. 3D animation. Static shot.",
                    dur=4, trim_end=0),
         audio="tts",  # Gemini TTS toy voice — lands at the END of the clip (pull-string moment)
         tts=dict(text_key="scene7_toy_line", style_key="scene7_tts_style", at="end"),
         sfx=dict(file="PullRing_Sound.mp3", start_s=0, vol=1.0)),  # pull-ring SFX at the very start

    dict(id="scene8", name="Custom childhood shot",
         image=dict(inputs=["GiveBonnie_Scene5_Generated.jpg"],
                    prompt="{scene8_image}",
                    output="GiveBonnie_Scene8_Generated.jpg"),
         video=dict(start="GiveBonnie_Scene8_Generated.jpg", end=None,
                    prompt="{scene8_video}", dur=4, trim_end=0),
         audio="none", muted=True),

    dict(id="scene9", name="Running on street",
         image=dict(inputs=["GiveBonnie_Scene5_Generated.jpg"],
                    # keep the toy reference dead simple ("the <x> toy") so Nano Banana
                    # preserves its look from the input image instead of re-inventing it.
                    prompt="Close-up shot of a boy running down a suburban street, wearing {scene9_outfit}, with the {action_figure_short} toy on his shoulders, laughing. 3D animated style.",
                    output="GiveBonnie_Scene9_Generated.jpg"),
         video=dict(start="GiveBonnie_Scene9_Generated.jpg", end=None,
                    prompt="Extreme slow motion. The boy runs in dramatic cinematic slow motion, every movement slowed way down, hair and clothing drifting slowly. Slow-motion sports replay footage.",
                    dur=4, trim_end=0),
         audio="none", muted=True),

    dict(id="scene10", name="Andy hands Bonnie the toy",
         image=None,  # uses scene6 generated image directly as the start frame
         video=dict(start="GiveBonnie_Scene6_Generated.jpg", end=None,
                    prompt="Guy smiles and hands the {action_figure_short} toy to the girl as she smiles and hugs it. 3D animation. Static shot.",
                    dur=4, trim_end=0),
         audio="none", muted=True),
]

# Scenes 8-10 share a background layer: the ElevenLabs Andy {closing_voiceover}
# mixed with CLOSING_BG_MUSIC, played continuously under the three muted clips.
CLOSING_SEQUENCE = ["scene8", "scene9", "scene10"]

# Approved prompt overrides from the playground (output/prompt_overrides.json) win over the
# defaults above — keys: {scene_id: {image, video, overlay_image, overlay_video}}.
PROMPT_OVERRIDES = ROOT / "prompt_overrides.json"


def _apply_overrides():
    import json
    if not PROMPT_OVERRIDES.exists():
        return
    try:
        ov = json.loads(PROMPT_OVERRIDES.read_text())
    except Exception:
        return
    for sc in SCENES:
        o = ov.get(sc["id"], {})
        if o.get("image") and sc.get("image"):
            sc["image"]["prompt"] = o["image"]
        if o.get("video") and sc.get("video"):
            sc["video"]["prompt"] = o["video"]
        ovl = sc.get("overlay") or {}
        if o.get("overlay_image") and ovl.get("image"):
            ovl["image"]["prompt"] = o["overlay_image"]
        if o.get("overlay_video") and ovl.get("video"):
            ovl["video"]["prompt"] = o["overlay_video"]


_apply_overrides()
