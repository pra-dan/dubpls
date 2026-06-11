import os
import dotenv
from pydantic_ai import Agent, RunContext
from agent_models import Segment

# Load environment variables where the GEMINI_API_KEY will be stored
dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Translator Agent (pydantic-ai / Gemini path)
# ---------------------------------------------------------------------------
# The static system prompt gives the LLM baseline instructions.
# Dynamic context (genre, video_type, speaker gender/emotion, visual scene)
# is injected per-segment via the @system_prompt decorator below.
# ---------------------------------------------------------------------------
translator_agent = Agent(
    'google:gemini-2.5-flash',
    output_type=str,
)


@translator_agent.system_prompt
def inject_segment_context(ctx: RunContext[Segment]) -> str:
    """
    Dynamically injects the video profile, visual context, gender and emotion
    into the system prompt for the specific segment being translated.
    """
    segment = ctx.deps

    # ── Video profile (whole-video; extracted once by VLM) ───────────────────
    vp = segment.video_profile
    if vp is not None:
        if hasattr(vp, "model_dump"):
            vp_dict = vp.model_dump()
        elif isinstance(vp, dict):
            vp_dict = vp
        else:
            vp_dict = {}
    else:
        vp_dict = {}

    video_type      = vp_dict.get("video_type", "movie")
    genre           = vp_dict.get("genre", "unknown")
    genre_examples  = vp_dict.get("genre_examples", "")
    maturity_rating = vp_dict.get("maturity_rating", "unknown")
    overall_tone    = vp_dict.get("tone", "neutral")
    formality_level = vp_dict.get("formality_level", "informal")
    setting_summary = vp_dict.get("setting_summary") or "Not available."

    # ── Per-segment visual context ───────────────────────────────────────────
    video_context = segment.video_context or "No visual context!"

    # ── Speaker gender ───────────────────────────────────────────────────────
    gender = ""
    if isinstance(segment.audio_gender_classification, dict):
        gender = segment.audio_gender_classification.get('label', '')
    elif isinstance(segment.audio_gender_classification, str):
        gender = segment.audio_gender_classification
    elif getattr(segment.audio_gender_classification, 'label', None):
        gender = segment.audio_gender_classification.label

    # ── Speaker emotion ──────────────────────────────────────────────────────
    emotion = ""
    if isinstance(segment.audio_emotion_classification, dict):
        emotion = segment.audio_emotion_classification.get('label', '')
    elif isinstance(segment.audio_emotion_classification, str):
        emotion = segment.audio_emotion_classification
    elif getattr(segment.audio_emotion_classification, 'label', None):
        emotion = segment.audio_emotion_classification.label

    # ── Character / dubbing style (dynamically extracted per-segment) ───────
    char_style = segment.character_style or ""
    dubbing_register = segment.dubbing_register or ""

    # Count source words for the prompt
    src_word_count = len(segment.text.strip().split())

    return f"""\
You are an expert {segment.language_name} dubbing translator.
Your task is to translate English dialogue into {segment.language_name} that perfectly matches \
the character's speaking style and the dubbing register described below.

CRITICAL RULES:
1. Output ONLY the translated line. No notes, quotes, explanations, no Devanagari transliteration.
2. **STRICT LENGTH CONSTRAINT**: The source text has {src_word_count} words. \
Your translation MUST have between {max(1, src_word_count - 2)} and {src_word_count + 2} words. \
This is the HIGHEST priority rule — violating word count is worse than slightly imperfect tone.
3. **GREETINGS & SHORT LINES**: If the source text is a greeting (e.g., "Hi"), translate it as a literal casual greeting (e.g., "हाय", "क्या हाल"), NOT as "हाँ" or an action. Keep the exact intent.
4. **SLANG & TONE**: The CHARACTER STYLE and DUBBING REGISTER describe the character's OVERALL personality. \
— Match the slang level to the line: use heavy slang for aggressive/profane lines. \
— NEVER use formal or bookish vocabulary (like 'स्वागत योग्य' or 'स्वागत') for a street/casual character. \
— For street/casual characters, heavily prefer idioms and regional slang (e.g., Bambaiya Hindi) over literal standard translations, but keep the core meaning intact.
5. Do NOT sanitize profanity or vulgarity — if the source is crude, the translation must be equally crude.
6. Prefer spoken/oral {segment.language_name} over written/literary {segment.language_name}.
7. Greetings should match the character's casualness (e.g. don't use formal greetings for a crude character).
8. Pronouns and address forms must match the character's relationship dynamics \
(e.g. use informal "you" forms for aggressive/casual speakers).
9. Output MUST be in {segment.language_name} script (e.g. Devanagari for Hindi). Never output in Roman/Latin script.

=== VIDEO PROFILE ===
Video Type    : {video_type}
Genre         : {genre}
Genre Tags    : {genre_examples}
Maturity      : {maturity_rating}
Overall Tone  : {overall_tone}
Formality     : {formality_level}
Setting       : {setting_summary}

=== CHARACTER STYLE ===
{char_style if char_style else "No character style info available — use the video profile tone/formality as guidance."}

=== DUBBING REGISTER ===
{dubbing_register if dubbing_register else "No dubbing register info available — match the video profile formality level."}

=== SCENE CONTEXT ===
Visual Context: {video_context}

=== SPEAKER INFO ===
Speaker Gender : {gender} (Ensure correct grammatical gender for self-referential words.)
Speaker Emotion: {emotion} (Ensure the tone matches this emotion.)
"""


async def translate_segment(segment: Segment) -> str:
    """
    Helper function to run the agent on a single segment.
    """
    result = await translator_agent.run(segment.text, deps=segment)
    return result.output
