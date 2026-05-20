import os
import dotenv
from pydantic_ai import Agent, RunContext
from agent_models import Segment

# Load environment variables where the GEMINI_API_KEY will be stored
dotenv.load_dotenv()

# Initialize the Translator Agent using Pydantic AI and Google Gemini
translator_agent = Agent(
    'google:gemini-3.1-flash-lite',
    output_type=str,
    system_prompt=(
        "You are an expert French dubbing translator for an R-rated comedic action movie. "
        "Your task is to translate English dialogue into highly colloquial, informal French.\n\n"
        "Rules:\n"
        "1. Do NOT reply to the text or provide notes. Only output the translation.\n"
        "2. Use slang, informal phrasing (argot), and NEVER use 'vous' (always use 'tu').\n"
        "3. STRICT LENGTH CONSTRAINT: The output length must closely match the input.\n"
    ),
)

@translator_agent.system_prompt
def inject_segment_context(ctx: RunContext[Segment]) -> str:
    """
    Dynamically injects the visual, gender, and emotion context into the system prompt
    for the specific segment being translated.
    """
    segment = ctx.deps
    video_context = segment.video_context or "No visual context!"
    
    # Extract gender gracefully
    gender = ""
    if isinstance(segment.audio_gender_classification, dict):
        gender = segment.audio_gender_classification.get('label', '')
    elif isinstance(segment.audio_gender_classification, str):
        gender = segment.audio_gender_classification
    elif getattr(segment.audio_gender_classification, 'label', None):
        gender = segment.audio_gender_classification.label

    # Extract emotion gracefully
    emotion = ""
    if isinstance(segment.audio_emotion_classification, dict):
        emotion = segment.audio_emotion_classification.get('label', '')
    elif isinstance(segment.audio_emotion_classification, str):
        emotion = segment.audio_emotion_classification
    elif getattr(segment.audio_emotion_classification, 'label', None):
        emotion = segment.audio_emotion_classification.label

    return f"""
Context from Video: {video_context}
Speaker Gender: {gender} (Ensure correct grammatical gender for self-referential words).
Speaker Emotion: {emotion} (Ensure the tone reflects this emotion).
"""

async def translate_segment(segment: Segment) -> str:
    """
    Helper function to run the agent on a single segment.
    """
    result = await translator_agent.run(segment.text, deps=segment)
    return result.output
