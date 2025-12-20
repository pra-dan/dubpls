"""
Implements part A. Demixing
Input: Audio (wav) path
Output: None. Saves dialog, BG music and effects wav files in same path as input.
"""

import yaml
import os
from external.TIGER.look2hear.models import TIGERDNR
import argparse
import torch
import torchaudio

# audio path
def breakdown_audio(audio_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device}")

    # Load model
    model = TIGERDNR.from_pretrained("JusperLee/TIGER-DnR", cache_dir="cache")
    model.to(device)
    model.eval()

    audio = torchaudio.load(audio_path)[0].to(device)

    with torch.no_grad():
        all_target_dialog, all_target_effect, all_target_music = model(audio[None])

    base_name = os.path.splitext(audio_path)[0]
    torchaudio.save(f"{base_name}_dialog.wav", all_target_dialog.cpu(), 44100)
    torchaudio.save(f"{base_name}_effect.wav", all_target_effect.cpu(), 44100)
    torchaudio.save(f"{base_name}_music.wav", all_target_music.cpu(), 44100)