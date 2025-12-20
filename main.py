import yaml
import subprocess
import sys
import os

from utils import isolate_media_streams
from breakdown_audio import breakdown_audio

def main():
    with open("config.yaml","r") as f:  
        config = yaml.safe_load(f)
        test_file_path = config["av_file_path"]

        # Isolate audio/video
        wav_path, muted_vid_path, ret = isolate_media_streams(test_file_path)
        if(ret):
            print(f"Found audio in file {test_file_path}")
        else:
            print(f"[E] No audio found in file {test_file_path}! Exiting")
            return

        # Run speech separation using TIGER model
        breakdown_audio(wav_path)
        # script_path = os.path.join("external", "TIGER", "inference_dnr.py")
        # if not os.path.exists(script_path):
        #     print(f"[E] Script not found at {script_path}! Check submodule imports. Exiting")
        #     return
        
        # print(f"Running speech separation on: {wav_path}")
        # try:
        #     result = subprocess.run(
        #         [sys.executable, script_path, "--audio_path", wav_path], # "--output_dir", config["out_file_path"]
        #         check=True,
        #         capture_output=False,  # Set to True if you want to capture output
        #         cwd=os.path.dirname(os.path.abspath(__file__))  # Run from project root
        #     )
        #     print("Speech separation completed successfully")
        # except subprocess.CalledProcessError as e:
        #     print(f"[E] Error running speech separation: {e}")
        #     return

        

if __name__ == "__main__":
    main()