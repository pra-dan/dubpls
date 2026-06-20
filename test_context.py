import asyncio
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from vlm_agent import extract_segment_context_cloud
from agent_models import Segment

async def test():
    seg = Segment(
        text="I told you, you're not welcome here.",
        start=1.432,
        end=3.134,
        speaker="SPEAKER_00"
    )

    scene_summaries = [
        {
            "scene_idx": 0,
            "start": 0.0,
            "end": 10.0,
            "summary": "A man wearing a grey suit is standing in an empty bar. He looks frustrated."
        }
    ]

    print("Running extract_segment_context_cloud...")
    result = await extract_segment_context_cloud(seg, scene_summaries)
    print("Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(test())
