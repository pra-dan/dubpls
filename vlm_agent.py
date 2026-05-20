import asyncio
import base64
import cv2
import requests
from typing import List
from agent_models import Segment

# Local multimodal model endpoint (MiniCPM / LLaVA via llama.cpp)
API_URL = "http://127.0.0.1:8080/v1/chat/completions"

def encode_image_to_base64(image) -> str:
    """Encodes a CV2 image (frame) to a base64 string."""
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    return base64.b64encode(buffer).decode('utf-8')

def get_video_frames(video_path: str, max_frames: int = 8, start_time: float = None, end_time: float = None) -> List[str]:
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
    # Avoid capturing too many frames
    for i in range(start_frame, end_frame, interval):
        if len(frames) >= max_frames: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Resize large frames to reduce processing time and VRAM usage
            height, width = frame.shape[:2]
            if width > 512:
                scale = 512 / width
                frame = cv2.resize(frame, (int(width*scale), int(height*scale)))
            frames.append(encode_image_to_base64(frame))
    
    cap.release()
    return frames

async def extract_visual_context(segment: Segment) -> str:
    """
    Extracts visual context for a specific segment using a local VLM.
    We use standard async/requests here because local llama.cpp multimodal 
    payloads can sometimes be finicky with standard library abstractions.
    """
    video_path = segment.collected_scenes_path
    if not video_path:
        return "No visual context available."

    # Run frame extraction synchronously (it's CPU bound)
    frames = get_video_frames(video_path, max_frames=8)
    if not frames: 
        return "No visual context available."

    ttext = "You are a visual context analyzer for movie scenes. "
    if segment.text:
        ttext += f"The current dialogue line is: '{segment.text}'. "
    ttext += "Based on the visual frames, provide a 1-sentence summary of the visual setting and the emotion/relationship of the characters. Do NOT hallucinate subtitles, do NOT output Chinese, and keep it very brief."

    content = [{"type": "text", "text": ttext}]
    
    for b64_img in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
        })

    payload = {
        "messages": [
            {"role": "user", "content": content}
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }

    # Run the blocking request in a thread pool to keep it async-friendly
    loop = asyncio.get_event_loop()
    
    def make_request():
        resp = requests.post(API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']

    try:
        context = await loop.run_in_executor(None, make_request)
        return context
    except Exception as e:
        print(f"Error extracting visual context for segment {segment.start}-{segment.end}: {e}")
        return "Visual context extraction failed."
