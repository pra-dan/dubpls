import os
import re
import sys
import json
import librosa
import numpy as np
import ffmpeg
import tempfile
import subprocess
import torchaudio 
from pydub import AudioSegment

# Add paths for dependencies
sys.path.append('../external/CosyVoice')
sys.path.append('../external/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel

def clip_all_segment_audio(segments: dict, input_audio_path: str):
    """
    clips and saves audio clip for each segment and returns save directory
    """
    if not segments:
        print("[E] No segments provided!")
        return None

    if not os.path.isfile(input_audio_path):
        raise FileNotFoundError(f"Input audio file {input_audio_path} not found.")

    # Prepare temp directory for individual clips
    temp_dir = tempfile.mkdtemp(prefix="segment_audio_clips_")
    temp_files = []

    for idx, seg in enumerate(segments):
        start = seg.get("start")
        end = seg.get("end")
        if start is None or end is None or end <= start:
            print(f"Skipping segment {idx}: start/end invalid")
            continue
        segment_out = os.path.join(temp_dir, f"segment_{idx}.wav")
        (
            ffmpeg.input(input_audio_path, ss=start, to=end)
            .output(segment_out, acodec='pcm_s16le', ac=2, ar='44100', loglevel="error")
            .overwrite_output()
            .run()
        )
        temp_files.append(segment_out)
    
    return temp_dir

def preprocess_prompts(idx: int, segment: dict, clips_dir: str, min_len: int=4):
    """
    duplicates prompts to extend their length, as required for prosody-context extraction
    """
    text = segment.get("text", None)
    if text is None:
        prompt_text = "<|endofprompt|>"
    else:
        word_count = len(re.findall(r'\w+', text))
        if word_count < min_len and idx < 3: # only extend for first few segments
            # Duplicate the text and its audio prompt
            prompt_text = (text.strip() + " ") * (3 // word_count + 1)
            prompt_text = prompt_text.strip()
            prompt_text = prompt_text + "<|endofprompt|>"
            
            # For audio, concat the file the same number of times
            base_audio_path = os.path.join(clips_dir, f"segment_{idx}.wav")
            base_audio = AudioSegment.from_wav(base_audio_path)
            repeat_times = (3 // word_count + 1)
            extended_audio = base_audio * repeat_times
            # Save to a new temp prompt audio file for this use
            extended_audio_path = os.path.join(clips_dir, f"segment_{idx}_extended.wav")
            extended_audio.export(extended_audio_path, format="wav")
            prompt_audio = extended_audio_path
        else:
            prompt_text = text + "<|endofprompt|>"
            prompt_audio = os.path.join(clips_dir, f"segment_{idx}.wav")

    return prompt_text, prompt_audio

# Suppress Numba's debug logs to stop the console spam
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

def get_leading_start(audio_path):
    """
    Finds the start time of speech in an audio file using librosa.
    Returns:
        float: Start time in milliseconds.
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    if len(y) == 0:
        print("audio extremely short")
        return 0.0

    # Trim silence (top_db=60 is very sensitive; standard is usually 20-40, 
    _, index = librosa.effects.trim(y, top_db=25)
    
    # Handle case where file is completely silent
    if len(index) < 2: 
        print("file is completely silent")
        return 0.0
        
    start_sample = index[0]
    start_time_ms = (start_sample / sr) * 1000
    
    return float(start_time_ms)

def check_and_postprocess(ref_path, test_path):
    """
    Synchronizes volume and duration using pitch-preserved time stretching.
    """
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference audio {ref_path} not found.")

    if not os.path.exists(test_path) or os.path.getsize(test_path) == 0:
        print("[W] generated audio missing; falling back to ref audio")
        return ref_path

    ref = AudioSegment.from_file(ref_path)
    test = AudioSegment.from_file(test_path)

    # Quality Check
    if len(test) < 250 or test.dBFS < -60:
        print("[W] generated audio invalid; falling back to ref audio")
        return ref_path

    # Speech Onset Alignment
    ref_onset = get_leading_start(ref_path)
    test_onset = get_leading_start(test_path)
    
    offset = int(test_onset - ref_onset)
    print("offset = ",offset)
    if offset > 0:
        test = test[offset:]
        print(f"[W] trimming {offset}ms leading silence")
    elif offset < 0:
        padding = AudioSegment.silent(duration=abs(offset))
        test = padding + test
        print(f"[W] adding {abs(offset)}ms leading silence")

    # Volume Matching & Pre-Export
    diff_in_db = ref.dBFS - test.dBFS
    test = test.apply_gain(diff_in_db)
    test.export(test_path, format="wav") 

    # Duration Matching via FFmpeg
    duration_ref = len(ref)
    duration_test = len(test)
    
    duration_delta = abs(duration_ref - duration_test)
    if duration_delta > 10:
        print("[W] Duration matching needed. Delta=",duration_delta)
        # tempo > 1.0 speeds up (shorter), < 1.0 slows down (longer)
        tempo = duration_test / duration_ref
        print("[W] Duration matching needed. tempo=",tempo)
        
        # if 0.5 <= tempo <= 2.0:
        if 0.5 <= tempo <= 3.5:
            # 
            temp_out = test_path.replace(".wav", "_stretched.wav")
            cmd = [
                'ffmpeg', '-y', '-i', test_path,
                '-filter:a', f'atempo={tempo}',
                '-vn', temp_out
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                os.replace(temp_out, test_path)
            except subprocess.CalledProcessError:
                print("[W] duration matching failed; falling back to ref")
                return ref_path
    
    return test_path

def tts(json_path: str, test_audio_path: str, target_lang='fr'):
    """
    Generates audio (TTS) audio clip for each translated segment and merges to 
    replace test audio
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Loading the 0.5B model
    cosyvoice = AutoModel(model_dir='../external/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')

    # Clip all segments
    segments = data.get("segments", [])
    clips_dir = clip_all_segment_audio(segments, test_audio_path)

    # Create a silent track based on original audio length
    waveform, sample_rate = torchaudio.load(test_audio_path) # Get duration of test_audio_path in ms
    full_duration_ms = int(waveform.shape[1] / sample_rate * 1000)
    final_dialogue_track = AudioSegment.silent(duration=full_duration_ms)

    # warm up
    # print("[I] Warm up for 5 segments")
    # for idx, segment in enumerate(segments):
    #     if(idx>5):
    #         break
    #     prompt_text, prompt_audio = preprocess_prompts(idx, segment, clips_dir)
    #     target_text = segment.get(target_lang, None)# + "<|endofprompt|>"
        
    #     for i, j in enumerate(cosyvoice.inference_zero_shot(target_text, prompt_text, prompt_audio, stream=False)):
    #         pass

    for idx, segment in enumerate(segments):
        # Run inference for the current segment
        # The output is a generator, so we iterate through it
        prompt_text, prompt_audio = preprocess_prompts(idx, segment, clips_dir)
        if("extended" in prompt_text):
            print("[W] This segment uses artificially prompt extension.")
        target_text = segment.get(target_lang, None) #+ "<|endofprompt|>"

        # if(idx==5):
        # print(f"---> {prompt_text} | {prompt_audio} | {target_text}")

        for i, j in enumerate(cosyvoice.inference_zero_shot(target_text, prompt_text, prompt_audio, stream=False)):
            # save generated audio with unique filename for each segment
            output_filename = f'tts_out_{target_lang}_segment_{idx}_output_{i}.wav'
            output_path = os.path.join("media", output_filename)
            torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)
            print(output_path)
            
            # check+modify if the generation was successful
            output_path = check_and_postprocess(prompt_audio, output_path)

            # use segment to create new audio
            seg_audio = AudioSegment.from_wav(output_path)
            # Place it at the start timestamp
            start_ms = int(segment.get("start") * 1000)
            final_dialogue_track = final_dialogue_track.overlay(seg_audio, position=start_ms)

    final_dialogue_track.export(os.path.join("media", "final_french_dialogue.wav"), format="wav")

if __name__ == '__main__':
    jpath = "/home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22_extracted_dialog_diarize_result.json"
    apath = "/home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22_extracted_dialog.wav"
    env = os.environ.copy()
    # jpath = env["SESSION_JSON_PATH"]
    # apath = env["TEST_DIALOGUE_AUDIO_PATH"]
    tts(jpath, apath)

    