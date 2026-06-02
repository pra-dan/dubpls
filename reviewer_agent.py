import os
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from agent_models import Segment

class ReviewResult(BaseModel):
    rating: int = Field(description="Total rating between 1 and 5", ge=1, le=5)
    evaluation: str = Field(description="The rationale for the rating")

# The reviewer agent uses Google's Gemini for fast evaluation.
reviewer_agent = Agent(
    'google:gemini-3.1-pro-preview',
    # 'google:gemini-3.1-flash-lite',
    output_type=ReviewResult,
)

@reviewer_agent.system_prompt
def inject_evaluation_context(ctx: RunContext[Segment]) -> str:
    segment = ctx.deps
    english = segment.text
    translated = segment.translation or getattr(segment, segment.target_language, None) or ""
    gt = getattr(segment, f"{segment.target_language}_gt", None) or ""
    
    return f"""
You will be given an Input text (English), a translated text ({segment.language_name}), and a ground-truth reference translation ({segment.language_name}).
Your task is to provide a 'total rating' scoring how well the translated text translates the Input text.
Give your answer on a scale of 1 to 5, where 1 means that the translated text is not helpful at all, and 5 means that the translated text completely and helpfully translates the user input.

Rubric (each criterion is worth +1 point):
+1 if the number of words in the translated text is offset by less than 2 words (more or fewer) compared to the ground-truth reference translation.
+1 if the intent of the Input text is retained in the translated text.
+1 if the tone of formality (professional, casual, vulgar, etc) of the translated text matches the ground-truth reference translation.
+1 if people are referred to correctly — e.g. gender is not mixed up, correct pronouns, etc.
+1 if the colloquialism level of the translated text matches the ground-truth reference translation.

Input text: {english}
translated text: {translated}
gt_translation_text: {gt}
"""

async def review_segment(segment: Segment) -> ReviewResult:
    """
    Helper function to run the reviewer agent on a single segment.
    """
    result = await reviewer_agent.run("Provide your feedback based on the rubric.", deps=segment)
    return result.output
