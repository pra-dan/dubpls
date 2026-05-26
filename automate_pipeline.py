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
from utils import load_config, get_language_name

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

config = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
TARGET_LANGUAGE = config.get("target_language", "fr")
LANGUAGE_NAME = get_language_name(TARGET_LANGUAGE)

VLM_COMPOSE = os.path.join(PROJECT_ROOT, "docker-compose-minicpm2.6.yaml")
TRANSLATION_COMPOSE = os.path.join(PROJECT_ROOT, "docker-compose.yaml")

DEFAULT_JSON = os.path.join(PROJECT_ROOT, "media", f"en_{TARGET_LANGUAGE}.json")
STAGE1_JSON = os.path.join(PROJECT_ROOT, "media", f"en_{TARGET_LANGUAGE}_eval_stage1.json")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "media", f"en_{TARGET_LANGUAGE}_eval_output.json")
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
def run_vlm_context(data: dict, video_path: str, total: int, dialogue_llm: str = "local") -> dict:
    """Extract whole-video profile + per-segment visual context via VLM."""
    print(f"\n{'='*60}")
    print("[Stage 1] VLM video context extraction")
    print(f"{'='*60}")

    import asyncio
    from agent_models import Segment
    from vlm_agent import extract_visual_context, extract_video_profile, extract_dialogue_profile, merge_profiles

    segments = data.get("segments", [])
    total = min(len(segments), total)

    # --- 1a. Extract whole-video profile (called ONCE) ---------------------
    # Build transcript snippets:
    #   - short sample (first 5) passed to the VLM to give it dialogue context
    #   - full transcript passed to the text-only dialogue profiler
    sample_lines = [s.get("text", "") for s in segments[:5] if s.get("text")]
    transcript_sample = "\n".join(sample_lines) if sample_lines else None
    full_transcript = "\n".join(s.get("text", "") for s in segments if s.get("text"))

    async def _extract_profile_and_contexts():
        print("  [Stage 1a] Extracting VideoProfile from dialogue (text) ...")

        # The user noted that the dialogue model (especially the cloud one) is much better,
        # so we rely exclusively on the text dialogue for the overall video profile.
        profile = await extract_dialogue_profile(full_transcript, llm_type=dialogue_llm)

        # Store as a plain dict in data so it serialises cleanly to JSON
        data["video_profile"] = profile.model_dump()
        print(f"  [Stage 1a] VideoProfile: {data['video_profile']}")

        # --- 1b. Per-segment visual context --------------------------------
        for i in range(total):
            seg_dict = segments[i]
            # Propagate the video_profile onto each segment dict so Stage 2 can use it
            seg_dict["video_profile"] = data["video_profile"]

            seg = Segment(
                text=seg_dict.get("text", ""),
                start=seg_dict.get("start", 0.0),
                end=seg_dict.get("end", 0.0),
                speaker=seg_dict.get("speaker", "unknown"),
                collected_scenes_path=seg_dict.get("collected_scenes_path", video_path),
                audio_gender_classification=seg_dict.get("audio_gender_classification"),
                audio_emotion_classification=seg_dict.get("audio_emotion_classification"),
                video_context=seg_dict.get("video_context"),
                video_profile=profile,
                translation=seg_dict.get("translation"),
                target_language=TARGET_LANGUAGE,
                language_name=LANGUAGE_NAME,
                **{k: v for k, v in seg_dict.items() if k not in [
                    "text", "start", "end", "speaker", "collected_scenes_path",
                    "audio_gender_classification", "audio_emotion_classification",
                    "video_context", "video_profile", "translation"
                ]}
            )

            try:
                video_context = await extract_visual_context(seg)
                seg_dict["video_context"] = video_context
            except Exception as e:
                print(f"  [!] VLM error on segment {i}: {e}")
                seg_dict["video_context"] = ""

            if (i + 1) % 5 == 0 or i + 1 == total:
                print(f"  Processed {i+1}/{total} segments")

    asyncio.run(_extract_profile_and_contexts())

    print("[Stage 1] Done.")
    return data


# ---------------------------------------------------------------------------
# Stage 2 — Translation (requires TowerInstruct container)
# ---------------------------------------------------------------------------
def run_translation(data: dict, total: int) -> dict:
    """Translate each segment using the Translation Agent."""
    print(f"\n{'='*60}")
    print(f"[Stage 2] Translation (English → {LANGUAGE_NAME})")
    print(f"{'='*60}")

    import asyncio
    from agent_models import Segment, VideoProfile
    from translator_agent import translate_segment

    segments = data.get("segments", [])
    total = min(len(segments), total)

    # Re-hydrate the VideoProfile from the stored dict (if present)
    raw_profile = data.get("video_profile")
    video_profile_obj = None
    if raw_profile and isinstance(raw_profile, dict):
        try:
            video_profile_obj = VideoProfile(**raw_profile)
        except Exception as e:
            print(f"  [!] Could not parse VideoProfile from data: {e}")

    async def _process_all():
        for i in range(total):
            seg_dict = segments[i]
            seg = Segment(
                text=seg_dict.get("text", ""),
                start=seg_dict.get("start", 0.0),
                end=seg_dict.get("end", 0.0),
                speaker=seg_dict.get("speaker", "unknown"),
                collected_scenes_path=seg_dict.get("collected_scenes_path"),
                audio_gender_classification=seg_dict.get("audio_gender_classification"),
                audio_emotion_classification=seg_dict.get("audio_emotion_classification"),
                video_context=seg_dict.get("video_context"),
                video_profile=video_profile_obj,
                translation=seg_dict.get("translation"),
                target_language=TARGET_LANGUAGE,
                language_name=LANGUAGE_NAME,
                **{k: v for k, v in seg_dict.items() if k not in [
                    "text", "start", "end", "speaker", "collected_scenes_path",
                    "audio_gender_classification", "audio_emotion_classification",
                    "video_context", "video_profile", "translation"
                ]}
            )
            try:
                translation = await translate_segment(seg)
                seg_dict["translation"] = translation
            except Exception as e:
                print(f"  [!] Translation error on segment {i}: {e}")
                seg_dict["translation"] = ""

            if (i + 1) % 5 == 0 or i + 1 == total:
                print(f"  Translated {i+1}/{total} segments")

    asyncio.run(_process_all())

    print("[Stage 2] Done.")
    return data


# ---------------------------------------------------------------------------
# Evaluation — compare translations against ground truth
# ---------------------------------------------------------------------------

# def evaluate(data: dict, total: int) -> None:
#     """Compare translated text with ground‑truth French text.

#     The previous implementation only counted exact string matches, which is
#     overly strict for natural language translation. We now compute a simple
#     word‑overlap ratio (intersection over union) and report it as *Word
#     Overlap Accuracy*. If a `review_feedback.json` file exists (generated by
#     the reviewer agent), we also surface the average rating from that file.
#     """
#     print(f"\n{'='*60}")
#     print("[Eval] Comparing translations with ground truth")
#     print(f"{'='*60}")

#     segments = data.get("segments", [])
#     total = min(len(segments), total)

#     # Track word‑overlap statistics
#     overlap_sum = 0.0
#     exact_matches = 0
#     results = []

#     for i in range(total):
#         seg = segments[i]
#         pred = (seg.get("translation") or "").strip()
#         gt = (seg.get("fr_gt") or "").strip()
#         # Exact match (kept for backward compatibility)
#         match = pred == gt and pred != ""
#         if match:
#             exact_matches += 1

#         # Simple word‑overlap (Jaccard‑like) metric
#         pred_words = set(pred.split())
#         gt_words = set(gt.split())
#         if pred_words or gt_words:
#             overlap = len(pred_words & gt_words) / len(pred_words | gt_words)
#         else:
#             overlap = 0.0
#         overlap_sum += overlap

#         results.append({
#             "idx": i,
#             "english": seg.get("text", ""),
#             "prediction": pred,
#             "gt": gt,
#             "exact_match": match,
#             "word_overlap": f"{overlap:.2f}",
#         })

#     avg_overlap = overlap_sum / total if total else 0.0

#     # Print detailed table
#     print(f"\n{'Seg':>4}  {'Exact':>5}  {'Overlap':>7}  {'Prediction (first 50 chars)':50}  {'GT (first 50 chars)':50}")
#     print("-" * 130)
#     for r in results:
#         flag = "✓" if r["exact_match"] else "✗"
#         print(
#             f"{r['idx']:>4}  {flag:>5}  {r['word_overlap']:>7}  {r['prediction'][:50]:50}  {r['gt'][:50]:50}"
#         )

#     # print(f"\nExact matches: {exact_matches}/{total}")
#     print(f"Word‑overlap accuracy: {avg_overlap*100:.1f}%")

#     # If a review JSON is present, report its average rating
#     try:
#         import json as _json
#         from pathlib import Path as _Path
#         review_path = _Path('media/review_feedback.json')
#         if review_path.is_file():
#             with open(review_path, 'r', encoding='utf-8') as f:
#                 review = _json.load(f)
#             avg_rating = review.get('overall', {}).get('avg_rating')
#             if avg_rating is not None:
#                 print(f"Reviewer average rating: {avg_rating:.2f} / 5")
#     except Exception:
#         pass


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
        "--dialogue-llm",
        choices=["cloud", "local"],
        default="cloud",
        help="Which LLM to use for extracting dialogue profile: 'cloud' (Gemini) or 'local' (llama.cpp) (default: cloud)",
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

            data = run_vlm_context(data, args.video_path, args.total, dialogue_llm=args.dialogue_llm)
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
        data = run_translation(data, args.total)

        # Save final results before review (so review can read them)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n[✓] Final results saved to: {OUTPUT_JSON}")

        # Run LLM-judge (Smart) review 
        if args.review:
            print(f"\n{'='*60}")
            print("[Review] Running LLM-as-judge evaluation ...")
            print(f"{'='*60}")
            import asyncio
            from review_translation import review as run_review
            asyncio.run(run_review(OUTPUT_JSON, args.total, REVIEW_JSON))

        # Unsmart review
        # evaluate(data, args.total)
    else:
        print("[Skip] Stage 2 skipped.")


    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
