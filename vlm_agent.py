import asyncio
import base64
import json
import re
import cv2
import numpy as np
import requests
from typing import List, Optional

from agent_models import Segment, VideoProfile

# Local multimodal model endpoint (MiniCPM / LLaVA via llama.cpp)
API_URL = "http://127.0.0.1:8080/v1/chat/completions"


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------

def encode_image_to_base64(image) -> str:
    """Encodes a CV2 image (frame) to a base64 string."""
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    return base64.b64encode(buffer).decode('utf-8')


def get_video_frames(
    video_path: str,
    max_frames: int = 8,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> List[str]:
    """Extracts a limited number of frames to avoid context overflow."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * fps) if start_time is not None else 0
    end_frame = int(end_time * fps) if end_time is not None else total_frames

    if start_frame >= total_frames:
        start_frame = 0
    if end_frame > total_frames or end_frame <= start_frame:
        end_frame = total_frames

    duration_frames = end_frame - start_frame
    if duration_frames <= 0:
        return []

    interval = max(1, duration_frames // max_frames)

    frames = []
    for i in range(start_frame, end_frame, interval):
        if len(frames) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Resize large frames to reduce VRAM usage
            height, width = frame.shape[:2]
            if width > 512:
                scale = 512 / width
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            frames.append(encode_image_to_base64(frame))

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Whole-video profile extraction
# ---------------------------------------------------------------------------

_VIDEO_PROFILE_PROMPT = """\
You are a video content analyst. Analyze the provided frames (sampled from the \
entire video) and return a JSON object — and ONLY the JSON object, no prose, no \
markdown fences — with exactly these keys:

{
  "genre": "<e.g. comedic action, historical drama, sci-fi thriller>",
  "genre_examples": "<comma-separated tags, e.g. Comedy, Action, Buddy Cop>",
  "video_type": "<one of: movie, trailer, speech, tutorial, documentary, series, short_film, other>",
  "maturity_rating": "<e.g. R-rated, PG-13, PG, G, NC-17>",
  "tone": "<e.g. sarcastic, serious, urgent, comedic, dark, lighthearted>",
  "formality_level": "<one of: informal, formal, neutral, street slang>"
}

Be concise and precise. Do not guess if unsure — use 'other' or 'neutral' as safe \
defaults. Do NOT include any text outside the JSON object."""


def _parse_video_profile_json(raw: str) -> VideoProfile:
    """
    Robustly extracts a VideoProfile from the VLM's raw text output.
    Falls back to safe defaults if parsing fails (so the pipeline never dies here).
    """
    VALID_VIDEO_TYPES = {"movie", "trailer", "speech", "tutorial", "documentary", "series", "short_film", "other"}

    # Strip markdown fences if the model added them anyway
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Attempt to pull the first {...} block
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"[vlm_agent] VideoProfile JSON parse failed. Raw output:\n{raw[:400]}")
        return VideoProfile(
            genre="unknown",
            genre_examples="unknown",
            video_type="other",
            maturity_rating="unknown",
            tone="neutral",
            formality_level="neutral",
        )

    # Sanitize video_type to the allowed Literal values
    vtype = data.get("video_type", "other").lower().strip()
    if vtype not in VALID_VIDEO_TYPES:
        vtype = "other"

    return VideoProfile(
        genre=data.get("genre", "unknown"),
        genre_examples=data.get("genre_examples", ""),
        video_type=vtype,  # type: ignore[arg-type]
        maturity_rating=data.get("maturity_rating", "unknown"),
        tone=data.get("tone", "neutral"),
        formality_level=data.get("formality_level", "neutral"),
    )


async def extract_video_profile(video_path: str, transcript_sample: Optional[str] = None) -> VideoProfile:
    """
    Calls the local VLM once with frames spread across the full video to produce a
    structured VideoProfile (genre, video_type, maturity_rating, tone, etc.).

    Args:
        video_path:        Path to the source video file.
        transcript_sample: Optional short excerpt of dialogue to help the VLM
                           disambiguate genre/tone (e.g. first 3 dialogue lines).

    Returns:
        A validated VideoProfile instance (never raises — falls back to safe defaults).
    """
    print(f"[vlm_agent] Extracting whole-video profile from: {video_path}")

    # Sample more frames spread across the FULL video for a representative overview
    frames = get_video_frames(video_path, max_frames=12)
    if not frames:
        print("[vlm_agent] No frames extracted — returning default VideoProfile.")
        return VideoProfile(
            genre="unknown",
            genre_examples="",
            video_type="other",
            maturity_rating="unknown",
            tone="neutral",
            formality_level="neutral",
            setting_summary=None,
        )

    prompt_text = _VIDEO_PROFILE_PROMPT
    if transcript_sample:
        prompt_text += (
            f"\n\nFor additional context, here is a sample of the dialogue:\n{transcript_sample}"
        )

    content = [{"type": "text", "text": prompt_text}]
    for b64_img in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
        })

    payload = {
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.05,   # Near-deterministic for structured extraction
        "max_tokens": 512,
    }

    loop = asyncio.get_event_loop()

    def make_request():
        resp = requests.post(API_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    try:
        raw = await loop.run_in_executor(None, make_request)
        print(f"[vlm_agent] VideoProfile raw response:\n{raw[:600]}")
        profile = _parse_video_profile_json(raw)
        print(f"[vlm_agent] Parsed VideoProfile: {profile.model_dump()}")
        return profile
    except Exception as e:
        print(f"[vlm_agent] extract_video_profile failed: {e}")
        return VideoProfile(
            genre="unknown",
            genre_examples="",
            video_type="other",
            maturity_rating="unknown",
            tone="neutral",
            formality_level="neutral",
        )


# ---------------------------------------------------------------------------
# Maturity rating — safety-first merge helpers
# ---------------------------------------------------------------------------

# Ordered from least to most restrictive. Anything not in this list gets rank 0
# (treated as "unknown"/least adult), so a known rating always beats it.
_MATURITY_RANK: dict[str, int] = {
    "unknown": 0,
    "g":       1,
    "pg":      2,
    "pg-13":   3,
    "pg13":    3,
    "r":       4,
    "r-rated": 4,
    "nc-17":   5,
    "nc17":    5,
    "adult":   6,
    "explicit":6,
    "x":       6,
}


def _maturity_rank(rating: str) -> int:
    """Return a numeric severity rank for a maturity rating string."""
    return _MATURITY_RANK.get(rating.lower().strip(), 0)


def merge_profiles(vlm_profile: VideoProfile, dialogue_profile: VideoProfile) -> VideoProfile:
    """
    Merge a VLM-derived (visual) profile with a dialogue-derived (text) profile.

    Merge strategy — safety first:
    - **maturity_rating**: the MORE restrictive (higher rank) rating always wins.
    - **video_type / genre / genre_examples / tone / formality_level**:
        prefer dialogue_profile when VLM returned "unknown"/"other"/"neutral",
        otherwise keep the VLM value (visual signal is usually more reliable for
        structural/genre classification).
    - **setting_summary**: VLM wins (it can actually see the frames); use dialogue
        fallback only if VLM has nothing.
    - Log whenever the two sources disagree so the Reviewer agent can audit.
    """
    merged = vlm_profile.model_copy()

    # ── Maturity (safety-first) ──────────────────────────────────────────────
    vlm_rank  = _maturity_rank(vlm_profile.maturity_rating)
    dlg_rank  = _maturity_rank(dialogue_profile.maturity_rating)
    if dlg_rank > vlm_rank:
        print(
            f"[merge_profiles] Maturity override: VLM='{vlm_profile.maturity_rating}' "
            f"→ dialogue='{dialogue_profile.maturity_rating}' (dialogue is more restrictive — using it)"
        )
        merged.maturity_rating = dialogue_profile.maturity_rating
    elif dlg_rank < vlm_rank:
        print(
            f"[merge_profiles] Maturity: keeping VLM='{vlm_profile.maturity_rating}' "
            f"over dialogue='{dialogue_profile.maturity_rating}'"
        )

    # ── Fields where dialogue overrides VLM unknowns ─────────────────────────
    _UNKNOWN_SENTINELS = {"unknown", "other", "neutral", ""}

    def _prefer_dialogue(vlm_val: str, dlg_val: str, field: str) -> str:
        if vlm_val.lower().strip() in _UNKNOWN_SENTINELS and dlg_val.lower().strip() not in _UNKNOWN_SENTINELS:
            print(f"[merge_profiles] {field}: VLM='{vlm_val}' → using dialogue='{dlg_val}'")
            return dlg_val
        if vlm_val != dlg_val and vlm_val.lower().strip() not in _UNKNOWN_SENTINELS:
            print(f"[merge_profiles] {field}: disagreement — keeping VLM='{vlm_val}' (dialogue='{dlg_val}')")
        return vlm_val

    merged.genre          = _prefer_dialogue(vlm_profile.genre,          dialogue_profile.genre,          "genre")
    merged.genre_examples = _prefer_dialogue(vlm_profile.genre_examples,  dialogue_profile.genre_examples, "genre_examples")
    merged.tone           = _prefer_dialogue(vlm_profile.tone,            dialogue_profile.tone,           "tone")
    merged.formality_level= _prefer_dialogue(vlm_profile.formality_level, dialogue_profile.formality_level,"formality_level")

    # video_type: "other" is the unknown sentinel
    if vlm_profile.video_type == "other" and dialogue_profile.video_type != "other":
        print(f"[merge_profiles] video_type: VLM='other' → using dialogue='{dialogue_profile.video_type}'")
        merged.video_type = dialogue_profile.video_type

    # setting_summary: VLM wins; fall back to dialogue if VLM has nothing
    if not vlm_profile.setting_summary and dialogue_profile.setting_summary:
        merged.setting_summary = dialogue_profile.setting_summary

    print(f"[merge_profiles] Final merged profile: {merged.model_dump()}")
    return merged


# ---------------------------------------------------------------------------
# Dialogue-only (text) profile extraction
# ---------------------------------------------------------------------------

_DIALOGUE_PROFILE_PROMPT = """\
You are a content analyst. Analyze the following video transcript (dialogue lines \
only — no visual information) and return a JSON object — and ONLY the JSON object, \
no prose, no markdown fences — with exactly these keys:

{
  "genre": "<e.g. comedic action, historical drama, sci-fi thriller>",
  "genre_examples": "<comma-separated tags, e.g. Comedy, Action, Buddy Cop>",
  "video_type": "<one of: movie, trailer, speech, tutorial, documentary, series, short_film, other>",
  "maturity_rating": "<e.g. R-rated, PG-13, PG, G, NC-17, Adult>",
  "tone": "<e.g. sarcastic, serious, urgent, comedic, dark, lighthearted>",
  "formality_level": "<one of: informal, formal, neutral, street slang>"
}

Guidelines:
- For maturity_rating, lean towards the MOST restrictive label that fits.
  Any profanity, sexual content, graphic violence, or drug references → at least R-rated.
  Extreme or explicit adult content → Adult or NC-17.
- Be concise and precise. Use 'other' or 'neutral' only as last resorts.
- Do NOT include any text outside the JSON object.

Transcript:
"""


async def extract_dialogue_profile(transcript: str, llm_type: str = "local") -> VideoProfile:
    """
    Calls either the local text-only endpoint or Gemini with the full transcript (all segments) to derive a
    VideoProfile purely from dialogue content.

    This is intentionally text-only (no images) so it can be run in parallel
    with or after extract_video_profile() and then merged via merge_profiles().

    Returns:
        A validated VideoProfile instance (never raises — falls back to safe defaults).
    """
    _DEFAULT_DIALOGUE_PROFILE = VideoProfile(
        genre="unknown",
        genre_examples="",
        video_type="other",
        maturity_rating="unknown",
        tone="neutral",
        formality_level="neutral",
    )

    if not transcript or not transcript.strip():
        print("[vlm_agent] extract_dialogue_profile: empty transcript — returning defaults.")
        return _DEFAULT_DIALOGUE_PROFILE

    print(f"[vlm_agent] Extracting dialogue-based profile ({len(transcript)} chars of transcript) via {llm_type.upper()}...")

    if llm_type == "cloud":
        from pydantic_ai import Agent
        import dotenv
        dotenv.load_dotenv()
        
        # Note: We just ask for JSON output and reuse the robust parser we already have
        cloud_agent = Agent(
            'google:gemini-3.1-flash-lite',
            system_prompt=_DIALOGUE_PROFILE_PROMPT,
        )
        try:
            result = await cloud_agent.run(transcript)
            raw = result.data if hasattr(result, 'data') else result.output # Handle different pydantic-ai versions
            if not isinstance(raw, str):
                raw = str(raw)
            print(f"[vlm_agent] Dialogue profile raw response (Cloud):\n{raw[:600]}")
            profile = _parse_video_profile_json(raw)
            print(f"[vlm_agent] Parsed dialogue profile: {profile.model_dump()}")
            return profile
        except Exception as e:
            print(f"[vlm_agent] extract_dialogue_profile (Cloud) failed: {e}")
            return _DEFAULT_DIALOGUE_PROFILE

    # Local LLM path
    payload = {
        "messages": [
            {
                "role": "user",
                "content": _DIALOGUE_PROFILE_PROMPT + transcript,
            }
        ],
        "temperature": 0.05,
        "max_tokens": 512,
    }

    loop = asyncio.get_event_loop()

    def make_request():
        resp = requests.post(API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    try:
        raw = await loop.run_in_executor(None, make_request)
        print(f"[vlm_agent] Dialogue profile raw response (Local):\n{raw[:600]}")
        profile = _parse_video_profile_json(raw)
        print(f"[vlm_agent] Parsed dialogue profile: {profile.model_dump()}")
        return profile
    except Exception as e:
        print(f"[vlm_agent] extract_dialogue_profile failed: {e}")
        return _DEFAULT_DIALOGUE_PROFILE


# ---------------------------------------------------------------------------
# Cloud-based scene summarization (Gemini Flash)
# ---------------------------------------------------------------------------

def create_scene_mosaic(
    video_path: str,
    start_time: float,
    end_time: float,
    grid_cols: int = 6,
    grid_rows: int = 6,
    tile_width: int = 320,
    tile_height: int = 180,
) -> Optional[bytes]:
    """
    Samples frames from [start_time, end_time], deduplicates similar frames,
    and arranges them in a grid mosaic.

    Returns the raw JPEG bytes of the mosaic, or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[vlm_agent] create_scene_mosaic: cannot open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = max(0, int(start_time * fps))
    end_f   = min(total_frames - 1, int(end_time * fps))

    n_tiles = grid_cols * grid_rows
    if end_f <= start_f:
        cap.release()
        return None

    num_candidates = n_tiles * 3
    positions = [
        start_f + int(i * (end_f - start_f) / max(1, num_candidates - 1))
        for i in range(num_candidates)
    ]

    tiles = []
    last_gray = None

    for pos in positions:
        if len(tiles) >= n_tiles:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            tile = cv2.resize(frame, (tile_width, tile_height))
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            if last_gray is not None:
                diff = cv2.absdiff(gray, last_gray).mean()
                if diff < 30.0:  # Skip similar frame
                    continue
            tiles.append(tile)
            last_gray = gray

    cap.release()

    # If we didn't find enough unique frames, pad with black tiles
    while len(tiles) < n_tiles:
        tiles.append(np.zeros((tile_height, tile_width, 3), dtype="uint8"))

    # Build grid row by row
    rows = []
    for r in range(grid_rows):
        row_tiles = tiles[r * grid_cols : (r + 1) * grid_cols]
        rows.append(cv2.hconcat(row_tiles))
    mosaic = cv2.vconcat(rows)

    _, buf = cv2.imencode(".jpg", mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return bytes(buf)


async def extract_scene_summaries_cloud(
    video_path: str,
    threshold: float = 27.0,
    grid_cols: int = 3,
    grid_rows: int = 3,
) -> List[dict]:
    """
    Detects scenes using PySceneDetect, builds a mosaic for each scene,
    and asks Gemini Flash for a concise 1-2 sentence summary of each scene.

    Returns a list of dicts:
        [
          {
            "scene_idx": 0,
            "start": 0.0,    # seconds
            "end": 12.5,
            "summary": "Two men argue near a bar counter at night..."
          },
          ...
        ]
    """
    import dotenv
    dotenv.load_dotenv()
    from pydantic_ai import Agent, BinaryContent
    from pydantic_ai.models.gemini import GeminiModelSettings
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:
        print("[vlm_agent] scenedetect not installed — cannot extract scene summaries.")
        exit(1)
        return []

    print(f"[vlm_agent] Detecting scenes in: {video_path}")
    vid = open_video(video_path)
    fps = vid.frame_rate or 25.0
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=int(fps * 2)))
    sm.detect_scenes(vid, show_progress=False)
    scene_list = sm.get_scene_list()
    print(f"[vlm_agent] Found {len(scene_list)} scenes")

    if not scene_list:
        return []

    scene_agent = Agent(
        "google:gemini-3.1-flash-lite",
        output_type=str,
        system_prompt=(
            "You are a scene analyst for film dubbing. You are given a mosaic of frames "
            "sampled equidistantly from a single scene/shot of a video.\n"
            "Write a descriptive paragraph (3-5 sentences) covering:\n"
            "  - The setting (location, time of day, atmosphere)\n"
            "  - Characters present (appearance, body language, position)\n"
            "  - What is happening (actions, interactions, key events)\n"
            "  - The emotional tone or tension of the scene\n"
            "  - Any narrative significance if evident from the visuals\n"
            "  - If you recognize the movie or any famous actors/characters, explicitly state their names!\n"
            "Be factual and specific. Avoid vague openers like 'The scene shows...'. "
            "Do NOT describe the mosaic grid layout itself. "
            "Do NOT add headings or bullet points — write continuous prose."
        ),
    )

    results = []
    prev_summary: str = ""
    for scene_idx, (start_tc, end_tc) in enumerate(scene_list):
        start_s = float(start_tc.get_seconds())
        end_s   = float(end_tc.get_seconds())

        mosaic_bytes = create_scene_mosaic(
            video_path, start_s, end_s,
            grid_cols=grid_cols, grid_rows=grid_rows,
        )
        if mosaic_bytes is None:
            print(f"[vlm_agent] Scene {scene_idx}: mosaic failed, skipping.")
            results.append({"scene_idx": scene_idx, "start": start_s, "end": end_s, "summary": ""})
            continue

        import os
        os.makedirs("media/mosaics", exist_ok=True)
        with open(f"media/mosaics/scene_{scene_idx}.jpg", "wb") as f:
            f.write(mosaic_bytes)

        try:
            # Build a prompt that includes the previous scene's context for narrative continuity
            user_text = f"Scene duration: {end_s - start_s:.1f}s.\n"
            if prev_summary:
                user_text += (
                    f"Previous scene context (for narrative continuity):\n{prev_summary}\n\n"
                    "Now describe the CURRENT scene shown in the mosaic below:"
                )
            else:
                user_text += "Describe this scene."

            result = await scene_agent.run(
                [
                    BinaryContent(data=mosaic_bytes, media_type="image/jpeg"),
                    user_text,
                ]
            )
            summary = result.data if hasattr(result, "data") else result.output
            if not isinstance(summary, str):
                summary = str(summary)
            summary = summary.strip()
            prev_summary = summary  # carry forward for next scene
        except Exception as exc:
            print(f"[vlm_agent] Scene {scene_idx} Gemini call failed: {exc}")
            summary = ""
            # Don't update prev_summary on failure so we don't propagate empty context

        print(f"[vlm_agent] Scene {scene_idx} [{start_s:.1f}s-{end_s:.1f}s]: {summary[:100]}")
        results.append({"scene_idx": scene_idx, "start": start_s, "end": end_s, "summary": summary})

    return results


def _find_scene_for_segment(scene_summaries: List[dict], seg_start: float, seg_end: float) -> Optional[dict]:
    """Returns the scene dict that best overlaps with the segment's time range."""
    if not scene_summaries:
        return None
    # Use the segment midpoint as the primary lookup key
    mid = (seg_start + seg_end) / 2.0
    best = None
    best_overlap = -1.0
    for sc in scene_summaries:
        overlap_start = max(sc["start"], seg_start)
        overlap_end   = min(sc["end"],   seg_end)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = sc
        # If mid falls cleanly inside, that's always best
        if sc["start"] <= mid < sc["end"]:
            return sc
    return best


async def extract_segment_context_cloud(
    segment: "Segment",
    scene_summaries: List[dict],
) -> str:
    """
    Given a segment and the pre-computed scene summaries, asks Gemini to produce
    a 1-sentence visual context for this dialogue grounded in the parent scene.

    This replaces the local `extract_visual_context()` call.
    """
    import dotenv
    dotenv.load_dotenv()
    from pydantic_ai import Agent

    parent_scene = _find_scene_for_segment(scene_summaries, segment.start, segment.end)
    scene_info = parent_scene["summary"] if parent_scene and parent_scene.get("summary") else "No scene summary available."

    context_agent = Agent(
        "google:gemini-3.1-flash-lite",
        output_type=str,
        system_prompt=(
            "You are a dubbing context annotator. Your job is to write a single concise sentence "
            "describing the visual context for a specific dialogue line, grounded in the scene description provided. "
            "This context will be used to help a translator choose the most appropriate wording. "
            "Be specific about characters, mood, and setting. Do NOT just repeat the scene description."
        ),
    )

    prompt = (
        f"Scene description: {scene_info}\n\n"
        f"Dialogue line: \"{segment.text.strip()}\"\n\n"
        "In one sentence, describe the specific visual context for this dialogue line."
    )

    try:
        result = await context_agent.run(prompt)
        context = result.data if hasattr(result, "data") else result.output
        if not isinstance(context, str):
            context = str(context)
        return context.strip()
    except Exception as exc:
        print(f"[vlm_agent] extract_segment_context_cloud failed for segment [{segment.start:.1f}s]: {exc}")
        return scene_info  # Fallback: just use the scene summary


# ---------------------------------------------------------------------------
# Character style extraction (cloud)
# ---------------------------------------------------------------------------

async def extract_character_style(
    segment: "Segment",
    scene_summaries: List[dict],
    full_transcript: str,
    video_profile: dict,
) -> str:
    """
    Analyzes the dialogue line in context to determine the speaking character's
    verbal personality: tone, attitude, vulgarity level, relationship dynamics,
    and speaking mannerisms. This is used by the translator to match the character's
    voice in the target language.

    Returns a concise character style description string.
    """
    import dotenv
    dotenv.load_dotenv()
    from pydantic_ai import Agent

    parent_scene = _find_scene_for_segment(scene_summaries, segment.start, segment.end)
    scene_info = parent_scene["summary"] if parent_scene and parent_scene.get("summary") else "No scene context."

    # Extract gender/emotion for richer character profiling
    gender = ""
    if isinstance(segment.audio_gender_classification, dict):
        gender = segment.audio_gender_classification.get("label", "")
    elif isinstance(segment.audio_gender_classification, str):
        gender = segment.audio_gender_classification



    style_agent = Agent(
        "google:gemini-3.1-flash-lite",
        output_type=str,
        system_prompt=(
            "You are a character voice analyst for film dubbing. "
            "Given a dialogue line, its surrounding transcript, scene description, "
            "speaker metadata, and the video's overall profile, you must describe "
            "the speaking character's verbal style in 2-3 sentences.\n\n"
            "Cover ALL of these aspects:\n"
            "1. Personality/attitude (e.g. sarcastic, aggressive, timid, playful)\n"
            "2. Vulgarity level (e.g. clean, mildly crude, heavily vulgar/profane)\n"
            "3. Formality of address (e.g. uses 'you' casually, speaks down to others, deferential)\n"
            "4. Speaking register (e.g. street slang, educated professional, military, childlike)\n"
            "5. Relationship to the listener (e.g. confrontational, flirtatious, authoritative)\n\n"
            "Be VERY specific. Do NOT be vague or generic. "
            "Output ONLY the character style description, no headings or bullets."
        ),
    )

    prompt = (
        f"=== VIDEO PROFILE ===\n"
        f"Genre: {video_profile.get('genre', 'unknown')}\n"
        f"Video Type: {video_profile.get('video_type', 'unknown')}\n"
        f"Maturity: {video_profile.get('maturity_rating', 'unknown')}\n"
        f"Overall Tone: {video_profile.get('tone', 'neutral')}\n\n"
        f"=== SCENE ===\n{scene_info}\n\n"
        f"=== SPEAKER ===\nGender: {gender}\n\n"
        f"=== SURROUNDING TRANSCRIPT (for context) ===\n{full_transcript[:1500]}\n\n"
        f"=== DIALOGUE LINE TO ANALYZE ===\n\"{segment.text.strip()}\"\n\n"
        f"Describe this character's verbal style for a dubbing translator."
    )

    try:
        result = await style_agent.run(prompt)
        style = result.data if hasattr(result, "data") else result.output
        if not isinstance(style, str):
            style = str(style)
        return style.strip()
    except Exception as exc:
        print(f"[vlm_agent] extract_character_style failed for segment [{segment.start:.1f}s]: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Dubbing register extraction (cloud)
# ---------------------------------------------------------------------------

async def extract_dubbing_register(
    video_profile: dict,
    character_style: str,
    target_language: str,
    language_name: str,
) -> str:
    """
    Given the video profile, character style, and target language, determines the
    specific dialect/register/slang level that the translator should use.

    For example:
    - R-rated superhero comedy + aggressive crude character + Hindi → "Mumbaiyya tapori Hindi with heavy street slang"
    - Period drama + aristocratic character + French → "Formal classical French with vous forms"
    - Kids animation + friendly character + Spanish → "Clean, simple Latin American Spanish"

    Returns a concise dubbing register description string.
    """
    import dotenv
    dotenv.load_dotenv()
    from pydantic_ai import Agent

    register_agent = Agent(
        "google:gemini-3.1-flash-lite",
        output_type=str,
        system_prompt=(
            f"You are a dubbing localization expert specializing in {language_name}.\n"
            f"Given a video's genre/tone/maturity and a character's speaking style, "
            f"you must determine the EXACT {language_name} dialect, register, and slang level "
            f"that a dubbing translator should use.\n\n"
            f"Your output must be a concise directive that a translator can follow. "
            f"Include:\n"
            f"1. The specific regional dialect or register (e.g. 'Mumbaiyya tapori Hindi', "
            f"'Parisian argot French', 'Mexican street Spanish')\n"
            f"2. Pronoun/address conventions (e.g. 'use तू/तेरा, never आप', 'use tú, never usted')\n"
            f"3. How to handle profanity (e.g. 'transliterate English swears directly', "
            f"'use local equivalents', 'keep it clean')\n"
            f"4. EXTREME CREATIVE LOCALIZATION EXAMPLES: Provide 2 examples of how to rewrite jokes, idioms, or cultural references to fit {language_name} culture perfectly instead of literally translating them (like adapting an English joke into a completely different but equivalent cultural idiom in {language_name}).\n\n"
            f"Be EXTREMELY specific to {language_name}. A translator reading your output "
            f"should know EXACTLY what dialect and style to write in.\n"
            f"Output ONLY the directive, no headings or explanations."
        ),
    )

    prompt = (
        f"=== VIDEO PROFILE ===\n"
        f"Genre: {video_profile.get('genre', 'unknown')}\n"
        f"Video Type: {video_profile.get('video_type', 'unknown')}\n"
        f"Maturity: {video_profile.get('maturity_rating', 'unknown')}\n"
        f"Overall Tone: {video_profile.get('tone', 'neutral')}\n"
        f"Formality: {video_profile.get('formality_level', 'informal')}\n\n"
        f"=== CHARACTER STYLE ===\n{character_style}\n\n"
        f"What {language_name} dialect/register should the dubbing translator use for this character?"
    )

    try:
        result = await register_agent.run(prompt)
        register = result.data if hasattr(result, "data") else result.output
        if not isinstance(register, str):
            register = str(register)
        return register.strip()
    except Exception as exc:
        print(f"[vlm_agent] extract_dubbing_register failed: {exc}")
        return ""

