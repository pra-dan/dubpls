#!/usr/bin/env python3
"""
Translation quality reviewer for dubpls.

Uses an LLM-as-judge approach: sends each (english, translation, gt) triple
to the model at http://127.0.0.1:8080 and asks it to score the translation
on a 1-5 rubric (character count, intent, formality, person references,
colloquialism).

Requires: a llama.cpp server running on port 8080.

Usage:
  /home/prashant/anaconda3/envs/whisperx2/bin/python review_translation.py \\
      --json-path media/en_fr_eval_output.json --total 5
"""
import argparse
import asyncio
import json
import os
import sys
from typing import Any

import dotenv
from agent_models import Segment
from reviewer_agent import review_segment
from utils import load_config, get_language_name

# Load environment variables (e.g., ANTHROPIC_API_KEY)
dotenv.load_dotenv()


# ---------------------------------------------------------------------------
# Build recommendations from aggregate results
# ---------------------------------------------------------------------------
def build_recommendations(segment_results: list[dict]) -> list[str]:
    """Generate concise, actionable recommendations.
    
    The previous implementation produced generic advice (length mismatches, formality, etc.) that
    largely duplicated the per‑segment evaluations already emitted by the reviewer agent. To avoid
    noise and keep the feedback focused, we now only surface recommendations for segments that
    scored poorly (≤2) or moderately (3). If all segments score well, we return a simple success
    message.
    """
    recs: list[str] = []
    low_score_segs = [r["idx"] for r in segment_results if r.get("rating", 5) <= 2]
    mid_score_segs = [r["idx"] for r in segment_results if r.get("rating", 5) == 3]

    if low_score_segs:
        recs.append(
            f"Segments {low_score_segs} scored ≤2/5. Prioritize fixing these – review their per‑segment evaluations for concrete failure patterns."
        )
    if mid_score_segs:
        recs.append(
            f"Segments {mid_score_segs} scored 3/5. Minor tweaks may improve them – consult the per‑segment evaluations for targeted adjustments."
        )
    if not recs:
        recs.append("All segments scored well. Consider fine‑tuning prompts for marginal gains.")
    return recs


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
async def review(json_path: str, total: int, output_path: str, target_language: str = "fr") -> dict[str, Any]:
    """Run LLM-as-judge evaluation and write review_feedback.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    language_name = get_language_name(target_language)
    segments = data.get("segments", [])
    total = min(len(segments), total)

    segment_results = []
    rating_sum = 0

    print(f"\n{'Seg':>4}  {'Score':>5}  {'Evaluation (first 80 chars)'}")
    print("-" * 100)

    for i in range(total):
        seg = segments[i]
        pred = (seg.get("translation") or seg.get(target_language) or "").strip()
        gt = (seg.get(f"{target_language}_gt") or "").strip()
        english = (seg.get("text") or "").strip()

        if not gt:
            continue

        if not pred:
            result = {
                "idx": i,
                "english": english,
                "pred": pred,
                "gt": gt,
                "rating": 0,
                "evaluation": "Empty translation — no output from translator.",
            }
        else:
            seg_obj = Segment(
                text=english,
                start=0.0,
                end=0.0,
                speaker="unknown",
                translation=pred,
                target_language=target_language,
                language_name=language_name,
                **{f"{target_language}_gt": gt}
            )
            try:
                judge_result = await review_segment(seg_obj)
                rating = judge_result.rating
                eval_text = judge_result.evaluation
            except Exception as e:
                rating = 0
                eval_text = f"Agent failed: {e}"

            result = {
                "idx": i,
                "english": english,
                "pred": pred,
                "gt": gt,
                "rating": rating,
                "evaluation": eval_text,
            }

        segment_results.append(result)
        rating_sum += result["rating"]

        eval_preview = result["evaluation"][:80].replace("\n", " ")
        print(f"{i:>4}  {result['rating']:>3}/5  {eval_preview}")

    evaluated = len(segment_results)
    avg_rating = rating_sum / evaluated if evaluated else 0
    perfect_count = sum(1 for r in segment_results if r["rating"] == 5)
    perfect_pct = (perfect_count / evaluated * 100) if evaluated else 0

    print(f"\n{'='*100}")
    print(f"  Segments evaluated : {evaluated}")
    print(f"  Average score      : {avg_rating:.1f} / 5")
    print(f"  Perfect scores     : {perfect_count}/{evaluated} ({perfect_pct:.1f}%)")
    print(f"{'='*100}")

    # Sort worst segments by rating ascending
    worst = sorted(segment_results, key=lambda r: r["rating"])[:10]
    recommendations = build_recommendations(segment_results)

    print("\n📋 Recommendations:")
    for idx, rec in enumerate(recommendations, 1):
        print(f"  {idx}. {rec}")

    feedback = {
        "overall": {
            "segments_evaluated": evaluated,
            "avg_rating": round(avg_rating, 2),
            "perfect_scores": perfect_count,
            "perfect_pct": round(perfect_pct, 1),
        },
        "per_segment": segment_results,
        "worst_segments": worst,
        "recommendations": recommendations,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Feedback saved to: {output_path}")

    return feedback


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(project_root, "config.yaml"))
    target_language = config.get("target_language", "fr")
    
    default_json = os.path.join(project_root, "media", f"en_{target_language}_eval_output.json")
    default_output = os.path.join(project_root, "media", "review_feedback.json")

    parser = argparse.ArgumentParser(
        description="Evaluate translations using LLM-as-judge rubric and write structured feedback.",
    )
    parser.add_argument(
        "--json-path",
        default=default_json,
        help=f"Path to translated segment JSON (default: {default_json})",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=999,
        help="Max segments to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"Output feedback JSON path (default: {default_output})",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.json_path):
        print(f"[E] JSON not found: {args.json_path}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(review(args.json_path, args.total, args.output, target_language))


if __name__ == "__main__":
    main()
