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
    if not vp:
        raise ValueError("Missing video_profile metadata during translation. Cannot proceed with limited context.")

    if hasattr(vp, "model_dump"):
        vp_dict = vp.model_dump()
    elif isinstance(vp, dict):
        vp_dict = vp
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
    dialogue_justification = segment.dialogue_justification or "No dialogue justification!"

    # ── Speaker gender ───────────────────────────────────────────────────────
    gender = ""
    if isinstance(segment.audio_gender_classification, dict):
        gender = segment.audio_gender_classification.get('label', '')
    elif isinstance(segment.audio_gender_classification, str):
        gender = segment.audio_gender_classification
    elif getattr(segment.audio_gender_classification, 'label', None):
        gender = segment.audio_gender_classification.label


    # ── Character / dubbing style (dynamically extracted per-segment) ───────
    char_style = segment.character_style or ""
    dubbing_register = segment.dubbing_register
    if not dubbing_register:
        raise ValueError("Missing dubbing_register metadata during translation. Cannot proceed with limited context.")

    # Count source words for the prompt
    src_word_count = len(segment.text.strip().split())

    return f"""\
You are an expert {segment.language_name} dubbing translator.
Your task is to translate English dialogue into {segment.language_name} that perfectly matches \
the character's speaking style and the dubbing register described below.

CRITICAL RULES:
1. Output ONLY the translated line. No notes, quotes, explanations, no Devanagari transliteration.
2. **NATURAL LENGTH**: The translation should be as concise as natural spoken Hindi/Hinglish allows. Do NOT artificially pad or truncate the sentence just to match the English word count if it destroys the slang, humor, or idiom.
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
10. **IDIOMS & SLANG ADAPTATION**: DO NOT translate English slang, jokes, or idioms literally (e.g., 'Whiskey Dick'). You MUST adapt them into culturally equivalent Hindi/Bambaiya slang or vulgarity (e.g., using words like 'नुन्नू', 'केले की मूँगफली', 'सुमड़ी', 'चिकने' if it fits) that captures the exact comedic or vulgar intent.
11. **HINGLISH USAGE**: For casual characters, frequently use English loanwords common in Hindi slang (e.g., 'सिस्टर', 'इंटरेस्ट', 'ड्रिंक', 'डेट', 'टच') instead of pure Hindi ('मैडम', 'मतलब', 'जाम').
12. **FORMALITY PRECISION**: Distinguish carefully between 'तू' (extreme intimacy/aggression) and 'तुम' (casual/mildly disrespectful). Do not use 'तू'/'देख' if 'तुम'/'सुनो' is more appropriate for a casual but not overly abrasive interaction. Do not add honorifics like 'साब' inappropriately.
13. **STRICT WORD COUNT ENFORCEMENT**: The translated word count MUST NOT differ by more than 1 word from the English source word count.

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
Dialogue Justification: {dialogue_justification}

=== SPEAKER INFO ===
Speaker Gender : {gender} (Ensure correct grammatical gender for self-referential words.)
"""


async def translate_segment(segment: Segment) -> str:
    """
    Helper function to run the agent on a single segment.
    """
    # Step 1: Initial Translation
    result = await translator_agent.run(segment.text, deps=segment)
    initial_translation = result.output if isinstance(result.output, str) else str(result.output)
    
    # Step 2: Proofreading / Grammar Check
    from proofreader_agent import proofread_segment
    final_translation = await proofread_segment(segment, initial_translation)
    
    return final_translation
