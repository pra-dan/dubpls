import asyncio
import base64
import json
import re
import cv2
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
  "formality_level": "<one of: informal, formal, neutral, street slang>",
  "setting_summary": "<1 sentence describing the overall visual setting / narrative context>"
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
            setting_summary=None,
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
        setting_summary=data.get("setting_summary"),
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
            setting_summary=None,
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
  "formality_level": "<one of: informal, formal, neutral, street slang>",
  "setting_summary": "<1 sentence describing the narrative setting inferred from dialogue>"
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
    Calls either the local text-only endpoint or Gemini with the full transcript to derive a
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
        setting_summary=None,
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
# Per-segment visual context extraction (unchanged from original)
# ---------------------------------------------------------------------------

async def extract_visual_context(segment: Segment) -> str:
    """
    Extracts visual context for a specific segment using a local VLM.
    We use standard async/requests here because local llama.cpp multimodal
    payloads can sometimes be finicky with standard library abstractions.
    """
    video_path = segment.collected_scenes_path
    if not video_path:
        return "No visual context available."

    frames = get_video_frames(video_path, max_frames=8)
    if not frames:
        return "No visual context available."

    ttext = "You are a visual context analyzer for movie scenes. "
    if segment.text:
        ttext += f"The current dialogue line is: '{segment.text}'. "
    ttext += (
        "Based on the visual frames, provide a 1-sentence summary of the visual setting "
        "and the emotion/relationship of the characters. "
        "Do NOT hallucinate subtitles, do NOT output Chinese, and keep it very brief."
    )

    content = [{"type": "text", "text": ttext}]
    for b64_img in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
        })

    payload = {
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    loop = asyncio.get_event_loop()

    def make_request():
        resp = requests.post(API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    try:
        context = await loop.run_in_executor(None, make_request)
        return context
    except Exception as e:
        print(f"Error extracting visual context for segment {segment.start}-{segment.end}: {e}")
        return "Visual context extraction failed."
