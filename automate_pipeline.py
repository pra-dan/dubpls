#!/usr/bin/env python3
"""
Automated two-stage translation pipeline for dubpls.

Orchestrates the full flow:
  0. (Optional) Audio classification — gender & emotion (runs locally, no Docker)
  1. Start VLM container → extract video context → stop VLM container
  2. Start Translation container → translate segments → stop Translation container
  3. Evaluate translations against ground truth

Usage:
  /home/prashant/anaconda3/envs/whisperx2/bin/python automate_pipeline.py --total 5
  /home/prashant/anaconda3/envs/whisperx2/bin/python automate_pipeline.py --dry-run
  /home/prashant/anaconda3/envs/whisperx2/bin/python automate_pipeline.py --skip-stage1
  /home/prashant/anaconda3/envs/whisperx2/bin/python automate_pipeline.py --skip-stage1 --review
"""
import argparse
import json
import os
import subprocess
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

VLM_COMPOSE = os.path.join(PROJECT_ROOT, "docker-compose-minicpm2.6.yaml")
TRANSLATION_COMPOSE = os.path.join(PROJECT_ROOT, "docker-compose.yaml")

DEFAULT_JSON = os.path.join(PROJECT_ROOT, "media", "en_fr.json")
STAGE1_JSON = os.path.join(PROJECT_ROOT, "media", "en_fr_eval_stage1.json")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "media", "en_fr_eval_output.json")
REVIEW_JSON = os.path.join(PROJECT_ROOT, "media", "review_feedback.json")

DEFAULT_AUDIO = os.path.join(
    PROJECT_ROOT, "media", "deadpool-2025-12-18_15.27.22_extracted_dialog.wav"
)
DEFAULT_VIDEO = os.path.join(
    PROJECT_ROOT, "media", "deadpool-2025-12-18_15.27.22.mp4"
)

HEALTH_URL = "http://127.0.0.1:8080/health"
HEALTH_POLL_INTERVAL = 5   # seconds
HEALTH_TIMEOUT = 180       # seconds


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------
def _compose_cmd(compose_file: str) -> list[str]:
    return ["docker", "compose", "-f", compose_file]


def start_container(compose_file: str) -> None:
    """Bring up the service defined in *compose_file*."""
    print(f"\n{'='*60}")
    print(f"[Docker] Starting container: {os.path.basename(compose_file)}")
    print(f"{'='*60}")
    # Defensive: stop any running instance first
    stop_container(compose_file, quiet=True)
    subprocess.run(
        _compose_cmd(compose_file) + ["up", "-d"],
        check=True,
        cwd=PROJECT_ROOT,
    )


def stop_container(compose_file: str, quiet: bool = False) -> None:
    """Tear down the service defined in *compose_file*."""
    if not quiet:
        print(f"\n[Docker] Stopping container: {os.path.basename(compose_file)}")
    subprocess.run(
        _compose_cmd(compose_file) + ["down"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )


def wait_for_health(timeout: int = HEALTH_TIMEOUT) -> bool:
    """
    Poll the llama.cpp /health endpoint until it returns 200
    or we exceed *timeout* seconds.
    """
    print(f"[Health] Waiting for server at {HEALTH_URL} (timeout={timeout}s) ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(HEALTH_URL, timeout=5)
            if r.status_code == 200:
                elapsed = time.time() - start
                print(f"[Health] Server ready after {elapsed:.1f}s")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(HEALTH_POLL_INTERVAL)
    print(f"[Health] Timed out after {timeout}s!")
    return False


# ---------------------------------------------------------------------------
# Stage 0 — Audio classification (local HF models, no Docker)
# ---------------------------------------------------------------------------
def run_audio_classification(data: dict, audio_path: str, total: int) -> dict:
    """Classify speaker gender and emotion for each segment."""
    print(f"\n{'='*60}")
    print("[Stage 0] Audio classification (gender + emotion)")
    print(f"{'='*60}")

    import librosa
    from audio_classification import classify_segment_audio, classify_audio_emotion_clip

    print(f"  Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)

    segments = data.get("segments", [])
    total = min(len(segments), total)

    for i in range(total):
        seg = segments[i]
        start_time = seg.get("start")
        end_time = seg.get("end")
        if start_time is None or end_time is None:
            continue

        start_sample = int(round(sr * start_time))
        end_sample = int(round(sr * end_time))
        audio_clip = audio[start_sample:end_sample]

        try:
            gender_pred = classify_segment_audio(audio_clip, sr)
            seg["audio_gender_classification"] = {
                "label": gender_pred["label"],
                "score": gender_pred["probs"][gender_pred["label"]],
            }

            emotion_pred = classify_audio_emotion_clip(audio_clip, sr)
            seg["audio_emotion_classification"] = {
                "label": emotion_pred["label"],
                "score": emotion_pred["probs"][emotion_pred["label"]],
            }
        except Exception as e:
            print(f"  [!] Audio classification error on segment {i}: {e}")
            seg["audio_gender_classification"] = {"label": "unknown", "score": 0.0}
            seg["audio_emotion_classification"] = {"label": "unknown", "score": 0.0}

        if (i + 1) % 5 == 0 or i + 1 == total:
            print(f"  Classified {i+1}/{total} segments")

    print("[Stage 0] Done.")
    return data


# ---------------------------------------------------------------------------
# Stage 1 — VLM context extraction (requires MiniCPM container)
# ---------------------------------------------------------------------------
def run_vlm_context(data: dict, video_path: str, total: int) -> dict:
    """Extract visual context for each segment via VLM."""
    print(f"\n{'='*60}")
    print("[Stage 1] VLM video context extraction")
    print(f"{'='*60}")

    from vlm import analyze_video

    segments = data.get("segments", [])
    total = min(len(segments), total)

    for i in range(total):
        seg = segments[i]
        start_time = seg.get("start")
        end_time = seg.get("end")

        try:
            video_context = analyze_video(video_path, start_time, end_time)
            seg["video_context"] = video_context if video_context else ""
        except Exception as e:
            print(f"  [!] VLM error on segment {i}: {e}")
            seg["video_context"] = ""

        if (i + 1) % 5 == 0 or i + 1 == total:
            print(f"  Processed {i+1}/{total} segments")

    print("[Stage 1] Done.")
    return data


# ---------------------------------------------------------------------------
# Stage 2 — Translation (requires TowerInstruct container)
# ---------------------------------------------------------------------------
def run_translation(data: dict, total: int) -> dict:
    """Translate each segment using the Translation LLM."""
    print(f"\n{'='*60}")
    print("[Stage 2] Translation (English → French)")
    print(f"{'='*60}")

    from translations.english_french import EnglishToFrenchTranslator

    translator = EnglishToFrenchTranslator()
    segments = data.get("segments", [])
    total = min(len(segments), total)

    for i in range(total):
        seg = segments[i]
        try:
            translation = translator.translate_text(seg)
            seg["translation"] = translation
        except Exception as e:
            print(f"  [!] Translation error on segment {i}: {e}")
            seg["translation"] = ""

        if (i + 1) % 5 == 0 or i + 1 == total:
            print(f"  Translated {i+1}/{total} segments")

    print("[Stage 2] Done.")
    return data


# ---------------------------------------------------------------------------
# Evaluation — compare translations against ground truth
# ---------------------------------------------------------------------------
def evaluate(data: dict, total: int) -> None:
    """Compare translated text with ground-truth French text."""
    print(f"\n{'='*60}")
    print("[Eval] Comparing translations with ground truth")
    print(f"{'='*60}")

    segments = data.get("segments", [])
    total = min(len(segments), total)

    exact_matches = 0
    results = []

    for i in range(total):
        seg = segments[i]
        pred = (seg.get("translation") or "").strip()
        gt = (seg.get("fr_gt") or "").strip()
        match = pred == gt and pred != ""

        if match:
            exact_matches += 1

        results.append(
            {
                "idx": i,
                "english": seg.get("text", ""),
                "prediction": pred,
                "gt": gt,
                "exact_match": match,
            }
        )

    # Print summary table
    print(f"\n{'Seg':>4}  {'Match':>5}  {'Prediction (first 50 chars)':50}  {'GT (first 50 chars)':50}")
    print("-" * 115)
    for r in results:
        flag = "  ✓" if r["exact_match"] else "  ✗"
        print(
            f"{r['idx']:>4}  {flag:>5}  {r['prediction'][:50]:50}  {r['gt'][:50]:50}"
        )

    print(f"\nExact matches: {exact_matches}/{total}")
    print(f"Accuracy: {exact_matches/total*100:.1f}%" if total else "N/A")


# ---------------------------------------------------------------------------
# Dry-run — validate paths and Docker availability
# ---------------------------------------------------------------------------
def dry_run(args) -> None:
    """Validate that all required files and tools are available."""
    print("[Dry-run] Validating setup ...\n")
    ok = True

    checks = [
        ("Input JSON", args.json_path),
        ("Audio file", args.audio_path),
        ("Video file", args.video_path),
        ("VLM compose", VLM_COMPOSE),
        ("Translation compose", TRANSLATION_COMPOSE),
    ]
    for label, path in checks:
        exists = os.path.isfile(path)
        status = "✓" if exists else "✗ MISSING"
        print(f"  {label:25s}: {status}  ({path})")
        if not exists:
            ok = False

    # Check Docker availability
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            check=True,
            capture_output=True,
        )
        print(f"  {'Docker Compose':25s}: ✓")
    except Exception:
        print(f"  {'Docker Compose':25s}: ✗  docker compose not found")
        ok = False

    if ok:
        print("\n[Dry-run] All checks passed. Ready to run.")
    else:
        print("\n[Dry-run] Some checks failed. Please fix before running.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Automated two-stage translation pipeline for dubpls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --total 5              Run full pipeline on first 5 segments\n"
            "  %(prog)s --skip-stage1           Skip VLM context; run only translation\n"
            "  %(prog)s --dry-run               Validate paths and Docker setup\n"
        ),
    )
    parser.add_argument(
        "--json-path",
        default=DEFAULT_JSON,
        help=f"Input segment JSON (default: {DEFAULT_JSON})",
    )
    parser.add_argument(
        "--audio-path",
        default=DEFAULT_AUDIO,
        help=f"Path to dialogue WAV (default: {DEFAULT_AUDIO})",
    )
    parser.add_argument(
        "--video-path",
        default=DEFAULT_VIDEO,
        help=f"Path to video file (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=5,
        help="Number of segments to process (default: 5)",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage 0+1 (audio classification + VLM). Reads from stage1 JSON directly.",
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Skip Stage 2 (translation). Useful to run only context extraction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and Docker availability without processing.",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=HEALTH_TIMEOUT,
        help=f"Seconds to wait for container health (default: {HEALTH_TIMEOUT})",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Run review_translation.py after translation to produce review_feedback.json.",
    )

    args = parser.parse_args()

    # -- Dry run --------------------------------------------------------
    if args.dry_run:
        dry_run(args)
        return

    # -- Validate inputs ------------------------------------------------
    if not args.skip_stage1 and not os.path.isfile(args.json_path):
        print(f"[E] Input JSON not found: {args.json_path}", file=sys.stderr)
        sys.exit(1)
    if not args.skip_stage1 and not os.path.isfile(args.audio_path):
        print(f"[E] Audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)
    if not args.skip_stage1 and not os.path.isfile(args.video_path):
        print(f"[E] Video file not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  dubpls — Automated Translation Pipeline")
    print("=" * 60)
    print(f"  Segments to process : {args.total}")
    print(f"  Skip Stage 1 (VLM)  : {args.skip_stage1}")
    print(f"  Skip Stage 2 (Trans): {args.skip_stage2}")
    print()

    # ===================================================================
    # STAGE 0 + 1: Audio classification + VLM context
    # ===================================================================
    if not args.skip_stage1:
        # Load source data
        with open(args.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Stage 0 — Audio classification (local, no Docker)
        data = run_audio_classification(data, args.audio_path, args.total)

        # Stage 1 — VLM context (needs MiniCPM container)
        try:
            start_container(VLM_COMPOSE)
            if not wait_for_health(args.health_timeout):
                print("[E] VLM container failed to become healthy. Aborting.")
                stop_container(VLM_COMPOSE)
                sys.exit(1)

            data = run_vlm_context(data, args.video_path, args.total)
        finally:
            stop_container(VLM_COMPOSE)

        # Save intermediate results
        with open(STAGE1_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n[✓] Stage 1 results saved to: {STAGE1_JSON}")
    else:
        print("[Skip] Stage 0+1 skipped. Loading existing stage1 JSON ...")
        if not os.path.isfile(STAGE1_JSON):
            print(f"[E] Stage1 JSON not found: {STAGE1_JSON}", file=sys.stderr)
            print("    Run without --skip-stage1 first.", file=sys.stderr)
            sys.exit(1)
        with open(STAGE1_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

    # ===================================================================
    # STAGE 2: Translation
    # ===================================================================
    if not args.skip_stage2:
        try:
            start_container(TRANSLATION_COMPOSE)
            if not wait_for_health(args.health_timeout):
                print("[E] Translation container failed to become healthy. Aborting.")
                stop_container(TRANSLATION_COMPOSE)
                sys.exit(1)

            data = run_translation(data, args.total)

            # Save final results before review (so review can read them)
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n[✓] Final results saved to: {OUTPUT_JSON}")

            # Run LLM-judge review while container is still up
            if args.review:
                print(f"\n{'='*60}")
                print("[Review] Running LLM-as-judge evaluation ...")
                print(f"{'='*60}")
                from review_translation import review as run_review
                run_review(OUTPUT_JSON, args.total, REVIEW_JSON)
        finally:
            stop_container(TRANSLATION_COMPOSE)
    else:
        print("[Skip] Stage 2 skipped.")

    evaluate(data, args.total)

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
