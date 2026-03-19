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
import json
import os
import re
import sys
import time
from typing import Any

import requests

# ---------------------------------------------------------------------------
# LLM Judge configuration
# ---------------------------------------------------------------------------
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

JUDGE_PROMPT = """You will be given an Input text (English), a translated text (French), and a ground-truth reference translation (French).
Your task is to provide a 'total rating' scoring how well the translated text translates the Input text.
Give your answer on a scale of 1 to 5, where 1 means that the translated text is not helpful at all, and 5 means that the translated text completely and helpfully translates the user input.

Rubric (each criterion is worth +1 point):
+1 if the number of words in the translated text is offset by less than 2 words (more or fewer) compared to the ground-truth reference translation.
+1 if the intent of the Input text is retained in the translated text.
+1 if the tone of formality (professional, casual, vulgar, etc) of the translated text matches the ground-truth reference translation.
+1 if people are referred to correctly — e.g. gender is not mixed up, correct choice between tu/toi vs vous, etc.
+1 if the colloquialism level of the translated text matches the ground-truth reference translation.

Provide your feedback as follows:

Feedback:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and answer.

Input text: {english_text}
translated text: {translated_text}
gt_translation_text: {gt_text}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company and lots of carrots.
Feedback:::
Evaluation: """


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------
def call_llm_judge(english: str, translated: str, gt: str, retries: int = 2) -> dict:
    """
    Send the judge prompt to the LLM and parse the rating + evaluation.
    Returns {"rating": int, "evaluation": str} or defaults on failure.
    """
    prompt = JUDGE_PROMPT.format(
        english_text=english,
        translated_text=translated,
        gt_text=gt,
    )

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 512,
    }

    for attempt in range(retries + 1):
        try:
            resp = requests.post(LLM_URL, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return parse_judge_response(content)
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
            else:
                print(f"  [!] LLM judge error: {e}")
                return {"rating": 0, "evaluation": f"LLM call failed: {e}"}


def parse_judge_response(text: str) -> dict:
    """
    Extract 'Total rating:' and evaluation text from the judge response.
    """
    rating = 0
    evaluation = text.strip()

    # Try to find "Total rating: N"
    match = re.search(r"Total\s+rating\s*:\s*(\d+)", text, re.IGNORECASE)
    if match:
        rating = int(match.group(1))
        rating = max(1, min(5, rating))  # clamp to 1-5

    # Try to extract evaluation text before "Total rating:"
    eval_match = re.search(
        r"Evaluation\s*:\s*(.*?)(?:Total\s+rating|$)", text, re.DOTALL | re.IGNORECASE
    )
    if eval_match:
        evaluation = eval_match.group(1).strip()

    return {"rating": rating, "evaluation": evaluation}


# ---------------------------------------------------------------------------
# Build recommendations from aggregate results
# ---------------------------------------------------------------------------
def build_recommendations(segment_results: list[dict]) -> list[str]:
    """Analyze segment-level results and produce actionable recommendations."""
    recs: list[str] = []

    low_score_segs = [r["idx"] for r in segment_results if r.get("rating", 5) <= 2]
    mid_score_segs = [r["idx"] for r in segment_results if r.get("rating", 5) == 3]

    # Scan evaluations for common patterns
    all_evals = " ".join(r.get("evaluation", "") for r in segment_results).lower()

    if "length" in all_evals or "word" in all_evals or "long" in all_evals or "short" in all_evals:
        recs.append(
            "Length mismatches detected. Tighten the length constraint in the "
            "translation prompt or add a max-word-count rule tied to the source word count."
        )
    if "vous" in all_evals or "formal" in all_evals:
        recs.append(
            "Formality/register issues detected. Reinforce the tu/toi rule in "
            "the prompt or add a post-processing step to replace 'vous' → 'tu'."
        )
    if "gender" in all_evals:
        recs.append(
            "Gender reference issues detected. Ensure the audio_gender_classification "
            "is being injected into the prompt and the model follows it."
        )
    if "tone" in all_evals or "colloquial" in all_evals or "slang" in all_evals:
        recs.append(
            "Tone/colloquialism mismatch detected. Strengthen slang/vulgar register "
            "instructions in the prompt or provide examples of expected output."
        )
    if "intent" in all_evals or "meaning" in all_evals:
        recs.append(
            "Intent/meaning loss detected. Consider switching to a stronger translation "
            "model or improving context injection."
        )

    if low_score_segs:
        recs.append(
            f"Segments {low_score_segs} scored ≤2/5. These need the most attention — "
            "review the judge evaluations for specific failure patterns."
        )
    if mid_score_segs:
        recs.append(
            f"Segments {mid_score_segs} scored 3/5. Minor improvements needed — "
            "check the per-segment evaluations for targeted fixes."
        )

    if not recs:
        recs.append("All segments scored well. Consider fine-tuning prompts for further marginal gains.")

    return recs


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def review(json_path: str, total: int, output_path: str) -> dict[str, Any]:
    """Run LLM-as-judge evaluation and write review_feedback.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    total = min(len(segments), total)

    segment_results = []
    rating_sum = 0

    print(f"\n{'Seg':>4}  {'Score':>5}  {'Evaluation (first 80 chars)'}")
    print("-" * 100)

    for i in range(total):
        seg = segments[i]
        pred = (seg.get("translation") or seg.get("fr") or "").strip()
        gt = (seg.get("fr_gt") or "").strip()
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
            judge_result = call_llm_judge(english, pred, gt)
            result = {
                "idx": i,
                "english": english,
                "pred": pred,
                "gt": gt,
                "rating": judge_result["rating"],
                "evaluation": judge_result["evaluation"],
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
    default_json = os.path.join(project_root, "media", "en_fr_eval_output.json")
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

    review(args.json_path, args.total, args.output)


if __name__ == "__main__":
    main()
