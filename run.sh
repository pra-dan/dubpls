#!/bin/bash
# sudo su # for docker (not necessary if docker runs without sudo privileges)
# conda activate whisperx
# python3 main.py


# get visual context
docker-compose -f docker-compose-minicpm2.6.yaml up -d
conda activate scene_det && python3 vlm.py
docker-compose -f docker-compose-minicpm2.6.yaml down

# translate
conda activate whisperx
python3 main.py

# conda activate cosyvoice
# python3 cosyvoice3.py
# this will save the output audio in the dubpls/media directory

# Example FFmpeg command to mix three tracks into one final dub
ffmpeg -i /home/prashant/Documents/dubpls/tts/media/final_french_dialogue.wav \
       -i /home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22_extracted_music.wav \
       -i /home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22_extracted_effect.wav \
       -filter_complex "[0:a][1:a][2:a]amix=inputs=3:duration=longest" \
       media/final_dubbed_audio_tower_aud_vid_context.wav

# # merge audio and video
ffmpeg -i /home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22_muted.mp4 \
       -i media/final_dubbed_audio_tower_aud_vid_context.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 \
       media/deadpool-2025-12-18_15.27.22_fr_feb3_1657.mp4