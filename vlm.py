import cv2
import base64
import requests
import json
import math

# --- Configuration ---
API_URL = "http://127.0.0.1:8080/v1/chat/completions"
# Path to your video file
VIDEO_PATH = "/home/prashant/Documents/dubpls/merged_clips_per_segment/seg_000.mp4" 

def encode_image_to_base64(image):
    """Encodes a CV2 image (frame) to a base64 string."""
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    return base64.b64encode(buffer).decode('utf-8')

def get_video_frames(video_path, max_frames=10):
    """Extracts a limited number of frames to avoid context overflow."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)
    
    frames = []
    count = 0
    for i in range(0, total_frames, interval):
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

def analyze_video():
    frames = get_video_frames(VIDEO_PATH, max_frames=8) # Start small
    if not frames: return

    # Construct Multimodal Message
    # Llama-server expects content to be a list of text + image_url objects
    ttext = "Analyze the key interaction in this video. \
        1. Identify the gender of the last speaker. \
        2. Describe the relationship between the speaker and listener (e.g., Intimate, Professional, Hostile). \
        3. Describe the last speaker's emotion. \
        4. Describe the visual setting in few words. \
        Output in JSON format with keys being 'gender', 'relationship', 'emotion', 'setting']"

    content = [{"type": "text", "text": f"Reply in English. {ttext}"}]
    
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
        print(response.json()['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Error: {e}")
        # Print server error detail if available
        if hasattr(e, 'response') and e.response:
            print(e.response.text)

if __name__ == "__main__":
    analyze_video()