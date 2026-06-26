"""
Personalized intro clip. The static intro (assets/Bonnie_Intro.mp4, ~23.4s) plays first while
the first image + first video generate in the background, so the product feels real-time.

It is identical for every topic EXCEPT at the 8.0s mark, where we overlay ONE short word in
Andy's ElevenLabs voice, completing the line "...someone told me you're really good with ___"
(the word "toys" swapped for something tied to the topic — "cups", "overthinking", "anxiety").
No lip-sync; the word is mixed on top of the intro's own audio.

    python3 intro.py "cup stacking"     # -> output/intro_<topic>.mp4
"""
import sys
import json
from pathlib import Path

import config
import gemini
import voice
import composite

INTRO_SRC = config.ASSETS / "Bonnie_Intro.mp4"
LINE_AT_S = 5.5   # when Andy's line starts in the intro
INTRO_LINE = "someone told me you're really good with {word}"   # {word} = the topic swap-in
# ElevenLabs Andy voice settings for the intro line (tuned for the intro specifically)
INTRO_VOICE = {"stability": 0.8, "similarity_boost": 0.8, "style": 0.35,
               "use_speaker_boost": True, "speed": 0.8}

_WORD_SCHEMA = {"type": "object", "required": ["word"], "properties": {
    "word": {"type": "string", "description": "the swap-in word — 1 short word (occasionally 2) "
             "that finishes 'you're really good with ___' for this topic"}}}


def intro_word(topic: str) -> str:
    """One short word tied to the topic, completing 'you're really good with ___'."""
    r = gemini.generate_json(
        config.GEMINI_TEXT_MODEL,
        f'Topic: "{topic}". Finish the sentence "someone told me you\'re really good with ___" '
        f'with ONE short word (two max) that nods to this topic the way "toys" would. '
        f'Examples: "cup stacking"->"cups", "overthinking"->"overthinking", "my anxiety"->"anxiety", '
        f'"red bull addiction"->"Red Bull". Lowercase unless it\'s a proper noun. Just the word.',
        _WORD_SCHEMA, system="You write punchy, natural spoken copy. One short word only.")
    return r["word"].strip().strip('.').strip()


def make_intro(topic: str, out: str | None = None, word: str | None = None) -> str:
    """Build the per-topic intro: Andy says the full line ("someone told me you're really good
    with {word}") starting at 5.5s over the intro clip; {word} is the topic swap-in."""
    word = word or intro_word(topic)
    line = INTRO_LINE.format(word=word)
    safe = topic.replace(" ", "_")[:40]
    out = out or str(config.OUTPUT / f"intro_{safe}.mp4")
    wav = voice.andy_tts(line, str(config.OUTPUT / f"intro_{safe}_line.mp3"), settings=INTRO_VOICE)
    composite.mix_audio_at(str(INTRO_SRC), wav, out, start_s=LINE_AT_S, vol=2.6)
    return out


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "cup stacking"
    w = intro_word(topic)
    print(f'word: "{w}"')
    print("->", make_intro(topic, word=w))
