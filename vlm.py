import cv2
import base64
import requests
import json
import math

"""
Requires llama.cpp running at below endpoint
models/MiniCPM-V-2_6-Q6_K.gguf + models/mmproj-model-f16.gguf
"""

API_URL = "http://127.0.0.1:8080/v1/chat/completions"
# Path to your video file
VIDEO_PATH = "/home/prashant/Documents/dubpls/merged_clips_per_segment/seg_000.mp4" 

def encode_image_to_base64(image):
    """Encodes a CV2 image (frame) to a base64 string."""
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    return base64.b64encode(buffer).decode('utf-8')

def get_video_frames(video_path, max_frames=10, start_time=None, end_time=None):
    """Extracts a limited number of frames to avoid context overflow."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video.")
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
    count = 0
    for i in range(start_frame, end_frame, interval):
        if len(frames) >= max_frames: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Resize large frames to reduce processing time
            height, width = frame.shape[:2]
            if width > 512:
                scale = 512 / width
                frame = cv2.resize(frame, (int(width*scale), int(height*scale)))
            frames.append(encode_image_to_base64(frame))
    
    cap.release()
    return frames

def analyze_video(video_path=VIDEO_PATH, start_time=None, end_time=None, text=None):
    frames = get_video_frames(video_path, max_frames=8, start_time=start_time, end_time=end_time) # Start small
    if not frames: return ""

    ttext = "You are a visual context analyzer for movie scenes. "
    if text:
        ttext += f"The current dialogue line is: '{text}'. "
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

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error: {e}")
        # Print server error detail if available
        if hasattr(e, 'response') and e.response:
            print(e.response.text)

if __name__ == "__main__":
    # analyze_video(video_path)
    # jpath = "/home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22_extracted_dialog_diarize_result.json"
    jpath = "temp.json"
    with open(jpath, 'r', encoding="utf-8") as f:
        data = json.load(f)

    for idx, segment in enumerate(data.get("segments", [])):
        vpath = segment.get("collected_scenes_path", "")
        response = analyze_video(vpath)
        data["segments"][idx]["video_context"] = response

    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)