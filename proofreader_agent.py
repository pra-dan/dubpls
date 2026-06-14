import dotenv
from pydantic_ai import Agent, RunContext
from agent_models import Segment

dotenv.load_dotenv()

# Use a smaller/fast model for grammar checking
proofreader_agent = Agent(
    "google:gemini-2.5-flash",
    output_type=str,
)

@proofreader_agent.system_prompt
def inject_context(ctx: RunContext[Segment]) -> str:
    segment = ctx.deps
    return f"""\
You are an expert {segment.language_name} linguist and dubbing proofreader.
You will be provided with an English source dialogue and its {segment.language_name} translation.
Your job is to check the translation for grammatical errors or overly literal, awkward phrasing (e.g., 'मुझे तुम्हें अभी मेरे साथ चलना होगा।' instead of 'मेरे साथ अभी चलो।').
The translation is meant to be spoken dialogue. Make sure it sounds natural to a native {segment.language_name} speaker while preserving the intended tone and slang level.

Rules:
1. If the original translation is already natural and grammatically correct, output the original translation exactly as is.
2. If it is awkward, literal, or grammatically incorrect, output the corrected, natural-sounding version.
3. DO NOT artificially pad or expand the sentence. Keep the word count similar to the original translation.
4. Output ONLY the final translation text. No explanations, no quotes, no notes.
"""

async def proofread_segment(segment: Segment, translation: str) -> str:
    """
    Helper function to run the proofreader agent on a translated segment.
    """
    prompt = f"English Source: {segment.text.strip()}\nOriginal Translation: {translation.strip()}\n\nCorrected Translation:"
    try:
        result = await proofreader_agent.run(prompt, deps=segment)
        output = result.data if hasattr(result, "data") else result.output
        if not isinstance(output, str):
            output = str(output)
        return output.strip()
    except Exception as e:
        print(f"[proofreader_agent] Error: {e}")
        return translation.strip()
