import yaml
import subprocess
import sys
import os

# load HF API key, etc
from dotenv import load_dotenv
load_dotenv() 

from utils import isolate_media_streams
from breakdown_audio import breakdown_audio
from diarize import diarize_with_forced_alignment
from translations import translate_segments

def main():
    with open("config.yaml","r") as f:  
        config = yaml.safe_load(f)
        # test_file_path = config["av_file_path"]

        # # Isolate audio/video
        # wav_path, muted_vid_path, ret = isolate_media_streams(test_file_path)
        # if(ret):
        #     print(f"Found audio in file {test_file_path}")
        # else:
        #     print(f"[E] No audio found in file {test_file_path}! Exiting")
        #     return

        # # Run speech separation using TIGER model
        # breakdown_audio(wav_path)
        
        # # Transcribe and force-align
        # base_name = os.path.splitext(wav_path)[0]
        # hf_read_token = os.getenv('HF_READ_TOKEN')
        # if not hf_read_token:
        #     raise ValueError("No API token found. Set the API_TOKEN environment variable.")
        # # diary_json_path = diarize_with_forced_alignment(f"{base_name}_dialog.wav", hf_read_token)
        diary_json_path = "media/deadpool-2025-12-18_15.27.22_extracted_dialog_diarize_result.json"

        # Extract scenes and derive context
        # scenes_clips = 
        
        # Translate T2T
        # translate_segments(diary_json_path, config)

        # export json and test audio to env, for next stage
        # env = os.environ.copy()
        # env["SESSION_JSON_PATH"] = diary_json_path
        # env["TEST_DIALOGUE_AUDIO_PATH"] = wav_path

if __name__ == "__main__":
    main()