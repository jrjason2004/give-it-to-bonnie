"""
Script brain: one user prompt (the thing being "given to Bonnie") -> a structured,
scene-by-scene comedic script, written by Gemini and constrained to SCRIPT_SCHEMA.
"""
import json
import sys
from pathlib import Path

import config
import gemini

SYSTEM = """You write scripts for "Give It To Bonnie" — a deadpan comedy bit that parodies the \
Toy Story 3 scene where a young man (call him "the guy" / Andy) lovingly hands his childhood \
"toys" down to a little girl (Bonnie). The joke: instead of real toys, he solemnly presents \
absurdly specific objects and concepts tied to the user's TOPIC, treating each like a cherished \
toy. It builds to a final "action figure" — a toy of the single person or character most central \
to the topic — which has a pull-string catchphrase.

Rules:
- Be FUNNY and hyper-specific. Specificity is the joke — name concrete things an insider to the topic \
instantly recognizes. No generic filler.
- ALWAYS use real BRAND names for objects; never generic. "an iPhone", "a Dunkin' iced coffee", \
"a Red Bull can", "a Stanley cup" — NOT "a smartphone", "a large iced coffee", "an energy drink". \
Brands are instantly recognizable and that recognition IS the joke. Pick the specific brand an insider \
to the topic would actually use.
- NAME objects by their real name; do NOT over-describe their physical geometry. The image model already \
knows these brands — hand it the precise brand NAME and let it render the look. e.g. "a neon green Speed \
Stacks cup", NOT "a neon green plastic cup with three holes in the bottom". Specific means the right \
brand NAME, not a longer description. Critical for every object fed to an image: scene1_item, scene3_items.
- CLIPS ARE SHORT (~3-7 seconds). EVERY spoken line must be very short — a few words up to one short \
sentence. NEVER a long monologue. Write how a person actually talks, with dry Toy-Story tenderness.

Per-line requirements (follow exactly):
- scene1_line: the guy is just STARTING to introduce the first item. Form: "Now this is [item]," + at \
most ~5 more words. Short. He has not finished his thought.
- scene1_item_short: a 1-2 word short noun for scene1_item (e.g. "a neon green Speed Stacks cup" -> "cup", \
"an empty Red Bull Sugarfree can" -> "can"), no adjectives — for brief references in prompts.
- scene2_line: he FINISHES that same sentence from scene 1 — a brief continuation (this clip is mostly \
the little girl grabbing the item, so keep his words minimal). It must read as the natural second half \
of scene 1's sentence. Do NOT introduce new exclamations like "careful".
- scene3_line: name BOTH items in this exact shape, using "this is" only ONCE: \
"Now this is [X] and [Y]. You gotta keep them together because they're madly in love." \
Never say "these two", and never a second "this is".
- scene3_items: the two items (X and Y), phrased to drop into an image swap.
- scene4_line: the little girl peers over the box at the special toy and, in VERY few words (2-4), \
excitedly calls the action-figure CHARACTER by the short kid-name she'd use for it — e.g. "My waiter!" \
or "My quarterback!". It must be what a kid would call THAT character, NOT a food item or object.
- action_figure: the PERSON/character central to the topic (becomes the toy). \
action_figure_short: ONE lowercase plain noun for that character for image prompts (e.g. "waiter").
- scene5_line: the guy lifts the toy, confused/fond: "[character]? What's he doing in here?"
- scene6_line: the little girl points at the toy and cutely says the toy's CATCHPHRASE exactly ONCE \
(do not repeat it).
- scene7_toy_line: the SAME catchphrase as scene6_line, written ONCE (the pull-string toy says it a \
single time — do NOT repeat it). Short and iconic — what that character actually says.
- scene7_tts_style: delivery instructions for that toy voice.
- IMPORTANT — in scenes 8 & 9 the kid is dressed as a NORMAL kid with at most ONE fun item nodding to \
the character (a single hat / jersey / accessory) — NEVER a full costume.
- scene8_image/scene8_video: wholesome childhood-bedroom flashback of the kid with the toy (still + a \
small motion), ending "3D animated style". CRUCIAL: refer to the toy ONLY as "the toy" — never \
re-describe its color, costume, or shape. The toy's look comes from a reference image; naming new \
details about it makes the model redraw it and break continuity. Customize the BEDROOM and the kid's \
small action, not the toy.
- scene9_outfit: normal-kid clothes plus that ONE fun item (e.g. "a t-shirt and jeans with a foam \
mascot hat"). (scene9_image is unused — that shot is built from a fixed template that calls it "the \
toy" — but still fill scene9_outfit.)
- closing_voiceover: SHORT heartfelt-but-funny send-off — 1 to 2 short sentences MAX, landing on a \
genuine-sounding life lesson twisted to the topic. Keep it brief.
Keep everything original. Do not quote copyrighted song lyrics or scripts."""

PROMPT = """TOPIC the user wants to give to Bonnie: "{topic}"

Write the full script. Make it genuinely funny and unmistakably about this exact topic."""

_FIELDS = list(config.SCRIPT_SCHEMA.keys())
SCHEMA = {
    "type": "object",
    "properties": {f: {"type": "string", "description": config.SCRIPT_SCHEMA[f]} for f in _FIELDS},
    "required": _FIELDS,
    "propertyOrdering": _FIELDS,
}


_SCENE1_SCHEMA = {
    "type": "object",
    "required": ["scene1_item", "scene1_item_short", "scene1_line"],
    "properties": {k: {"type": "string", "description": config.SCRIPT_SCHEMA[k]}
                   for k in ("scene1_item", "scene1_item_short", "scene1_line")},
}


def scene1_quick(topic: str) -> dict:
    """Fast first pass — ONLY scene 1's item + line, so scene 1 can start rendering immediately
    while the full script writes in parallel. No thinking → ~8s instead of ~37s."""
    r = gemini.generate_json(
        config.GEMINI_TEXT_MODEL,
        PROMPT.format(topic=topic) + "\n\nReturn ONLY scene 1's first item (scene1_item, "
        "scene1_item_short) and Andy's opening line (scene1_line).",
        _SCENE1_SCHEMA, system=SYSTEM, thinking=False)
    return {k: r[k] for k in ("scene1_item", "scene1_item_short", "scene1_line")}


def write_script(topic: str, seed: dict | None = None) -> dict:
    """Full script. `seed` (e.g. scene1_quick's output) is used VERBATIM and the rest is built
    consistently around it — so the fast scene-1 pass and the full script never disagree."""
    p = PROMPT.format(topic=topic)
    if seed:
        p += ("\n\nAlready decided — use these EXACT values and build everything else "
              "consistently around them:\n" + json.dumps(seed, ensure_ascii=False))
    script = gemini.generate_json(config.GEMINI_TEXT_MODEL, p, SCHEMA, system=SYSTEM)
    if seed:
        script.update(seed)            # enforce the seeded values
    script["_topic"] = topic
    return script


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "celiac disease"
    s = write_script(topic)
    print(json.dumps(s, indent=2, ensure_ascii=False))
