#!/usr/bin/env python3
"""
Translation quality reviewer for dubpls.

Evaluates translated segments against ground-truth French text and writes
structured feedback to media/review_feedback.json.  Designed to be run by
the Reviewer agent or as part of the automated pipeline.

Metrics:
  - Exact match
  - Sentence-level BLEU  (sacrebleu)
  - chrF                 (sacrebleu)
  - Length ratio          len(pred) / len(gt)

Usage:
  /home/prashant/anaconda3/envs/whisperx2/bin/python review_translation.py \\
      --json-path media/en_fr_eval_output.json --total 5
"""
import argparse
import json
import os
import sys
from typing import Any

import sacrebleu


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def sentence_bleu(pred: str, ref: str) -> float:
    """Return sentence-level BLEU (0–100)."""
    return sacrebleu.sentence_bleu(pred, [ref]).score


def sentence_chrf(pred: str, ref: str) -> float:
    """Return sentence-level chrF (0–100)."""
    return sacrebleu.sentence_chrf(pred, [ref]).score


def length_ratio(pred: str, ref: str) -> float:
    """Return character-level length ratio pred/ref (1.0 = perfect)."""
    if not ref:
        return 0.0
    return len(pred) / len(ref)


# ---------------------------------------------------------------------------
# Issue detection
# ---------------------------------------------------------------------------
def detect_issues(pred: str, gt: str, lr: float) -> list[str]:
    """Return a list of human-readable issue tags for one segment."""
    issues: list[str] = []

    if lr > 1.3:
        issues.append("too_long")
    elif lr < 0.7:
        issues.append("too_short")

    # Register check — 'vous' should not appear in informal French
    pred_lower = pred.lower()
    if "vous" in pred_lower.split():
        issues.append("formal_vous_detected")

    # Empty or near-empty prediction
    if not pred.strip():
        issues.append("empty_translation")

    # Untranslated — prediction is suspiciously close to English source
    # (simple heuristic: check if pred equals original English)
    # This is checked at the caller level where English text is available.

    return issues


# ---------------------------------------------------------------------------
# Build recommendations from aggregate stats
# ---------------------------------------------------------------------------
def build_recommendations(segment_results: list[dict]) -> list[str]:
    """Analyze segment-level results and produce actionable recommendations."""
    recs: list[str] = []

    long_segs = [r["idx"] for r in segment_results if "too_long" in r.get("issues", [])]
    short_segs = [r["idx"] for r in segment_results if "too_short" in r.get("issues", [])]
    vous_segs = [r["idx"] for r in segment_results if "formal_vous_detected" in r.get("issues", [])]
    empty_segs = [r["idx"] for r in segment_results if "empty_translation" in r.get("issues", [])]
    low_bleu = [r["idx"] for r in segment_results if r.get("bleu", 100) < 10]

    if long_segs:
        recs.append(
            f"Segments {long_segs} are too long (len_ratio > 1.3). "
            "Tighten the length constraint in the translation prompt or add a "
            "max-word-count rule tied to the source word count."
        )
    if short_segs:
        recs.append(
            f"Segments {short_segs} are too short (len_ratio < 0.7). "
            "The model may be truncating output. Check max_tokens or prompt formatting."
        )
    if vous_segs:
        recs.append(
            f"Segments {vous_segs} use formal 'vous'. "
            "Reinforce the tu/toi rule in the prompt or add a post-processing step to replace 'vous' → 'tu'."
        )
    if empty_segs:
        recs.append(
            f"Segments {empty_segs} produced empty translations. "
            "Check LLM server health, prompt formatting, or request timeout."
        )
    if low_bleu:
        recs.append(
            f"Segments {low_bleu} have very low BLEU (<10). "
            "Consider switching model, improving context injection, or adjusting temperature."
        )

    if not recs:
        recs.append("All segments look reasonable. Consider fine-tuning prompts for further marginal gains.")

    return recs


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def review(json_path: str, total: int, output_path: str) -> dict[str, Any]:
    """Run full evaluation and write review_feedback.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    total = min(len(segments), total)

    segment_results = []
    exact_matches = 0
    bleu_sum = 0.0
    chrf_sum = 0.0

    print(f"\n{'Seg':>4}  {'BLEU':>6}  {'chrF':>6}  {'LenR':>5}  {'Match':>5}  {'Issues'}")
    print("-" * 80)

    for i in range(total):
        seg = segments[i]
        pred = (seg.get("translation") or seg.get("fr") or "").strip()
        gt = (seg.get("fr_gt") or "").strip()
        english = (seg.get("text") or "").strip()

        if not gt:
            # No ground truth to compare — skip
            continue

        bleu = sentence_bleu(pred, gt)
        chrf = sentence_chrf(pred, gt)
        lr = length_ratio(pred, gt)
        exact = pred == gt and pred != ""

        issues = detect_issues(pred, gt, lr)
        # Check for untranslated text
        if pred and english and pred.strip().lower() == english.strip().lower():
            issues.append("untranslated")

        if exact:
            exact_matches += 1
        bleu_sum += bleu
        chrf_sum += chrf

        result = {
            "idx": i,
            "english": english,
            "pred": pred,
            "gt": gt,
            "bleu": round(bleu, 1),
            "chrf": round(chrf, 1),
            "length_ratio": round(lr, 2),
            "exact_match": exact,
            "issues": issues,
        }
        segment_results.append(result)

        flag = "✓" if exact else "✗"
        issues_str = ", ".join(issues) if issues else "—"
        print(f"{i:>4}  {bleu:>6.1f}  {chrf:>6.1f}  {lr:>5.2f}  {flag:>5}  {issues_str}")

    evaluated = len(segment_results)
    avg_bleu = bleu_sum / evaluated if evaluated else 0
    avg_chrf = chrf_sum / evaluated if evaluated else 0
    exact_pct = (exact_matches / evaluated * 100) if evaluated else 0

    print(f"\n{'='*80}")
    print(f"  Segments evaluated : {evaluated}")
    print(f"  Exact matches      : {exact_matches}/{evaluated} ({exact_pct:.1f}%)")
    print(f"  Average BLEU       : {avg_bleu:.1f}")
    print(f"  Average chrF       : {avg_chrf:.1f}")
    print(f"{'='*80}")

    # Sort worst segments by BLEU ascending
    worst = sorted(segment_results, key=lambda r: r["bleu"])[:10]
    recommendations = build_recommendations(segment_results)

    print("\n📋 Recommendations:")
    for idx, rec in enumerate(recommendations, 1):
        print(f"  {idx}. {rec}")

    feedback = {
        "overall": {
            "segments_evaluated": evaluated,
            "exact_matches": exact_matches,
            "exact_match_pct": round(exact_pct, 1),
            "avg_bleu": round(avg_bleu, 1),
            "avg_chrf": round(avg_chrf, 1),
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
        description="Evaluate translations against ground truth and write structured feedback.",
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
