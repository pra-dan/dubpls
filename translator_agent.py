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
    'google:gemini-3.1-flash-lite',
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

    return f"""\
You are an expert {segment.language_name} dubbing translator.
Your task is to translate English dialogue into highly colloquial, informal {segment.language_name}.

Rules:
1. Do NOT reply to the text or provide notes. Only output the translation.
2. Use slang, informal phrasing (argot).
3. STRICT LENGTH CONSTRAINT: The output length must closely match the input.

=== VIDEO PROFILE ===
Video Type    : {video_type}
Genre         : {genre}
Genre Tags    : {genre_examples}
Maturity      : {maturity_rating}
Overall Tone  : {overall_tone}
Formality     : {formality_level}
Setting       : {setting_summary}

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
