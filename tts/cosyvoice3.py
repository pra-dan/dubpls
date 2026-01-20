import os
import re
import sys
import json
import torch
import shutil
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

# Suppress Numba's debug logs to stop the console spam
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

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

def is_name_or_expression(segment: dict) -> bool:
    """
    Returns True if the segment text is likely just a name (e.g., "Wade Wilson")
    or a repetitive expression (e.g., "Woah, woah, woah", "Wait.").
    """
    text = segment.get("text", "").strip()
    if not text:
        return False

    # Clean the text: remove punctuation and split into words
    # "Wade Wolfin?" -> "Wade Wolfin"
    clean_text = re.sub(r'[^\w\s]', '', text)
    words = clean_text.split()
    
    if not words:
        return False
    
    # Check for Repetition (e.g., "Woah, woah, woah")
    lower_words = [w.lower() for w in words]
    if len(words) > 1:
        unique_words = set(lower_words)
        # If there's only 1 unique word used repeatedly (e.g. "woah woah")
        if len(unique_words) == 1:
            return True

    # Check for Interjections (Short single words)
    # Common interjections list
    interjections = {
        "woah", "whoa", "oh", "ah", "hmm", "wait", "hey", 
        "wow", "huh", "shh", "ouch", "oops"
    }
    if len(words) <= 2 and lower_words[0] in interjections:
        return True

    # Check for Names (Title Case Strategy)
    # Logic: Names (Wade Wilson) are usually 1-3 words and ALL start with a capital letter.
    # Normal sentences (Who's asking?) usually have lowercase words after the first.
    
    is_title_case = all(w[0].isupper() for w in words if len(w) > 1) 
    # (len>1 check ignores "A" or "I" if they appear, though rare in names)
    
    if is_title_case and len(words) <= 3:
        # Extra check: avoid false positives like "Who Is" if text is weirdly cased,
        # but in standard English dialogue, simple sentences aren't Title Cased.
        return True

    return False
    

def check_and_postprocess(segment: dict, ref_path: str, test_path: str):
    """
    Synchronizes volume and duration using pitch-preserved time stretching.
    Also replaces translated audio with original if the text is a name or just exclamation
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

def preprocess_prompts(idx: int, segment: dict, clips_dir: str, all_segments: list, min_len: int=3, MIN_LEN_NEIGHBOR: int=4):
    """
    - If a segment is too short, replace its prompt with the closest neighbor 
    from the same speaker that has > 4 words.
    The neighbour-using logic for prompt audio should be removed eventually.
    """
    text = segment.get("text", "")
    if not text:
        return "You are a helpful assistant.<|endofprompt|>", os.path.join(clips_dir, f"segment_{idx}.wav")

    # Clean word count check
    word_count = len(re.findall(r'\w+', text))
    
    # Check if prompt is too short (default min_len=4)
    if word_count < min_len:
        current_speaker = segment.get("speaker")
        
        # Search for suitable nearest-neighbor
        best_neighbor_idx = -1
        min_distance = float('inf')
        
        for i, other_seg in enumerate(all_segments):
            if i == idx: continue
            
            if other_seg.get("speaker") == current_speaker:
                neighbor_text = other_seg.get("text", "")
                neighbor_wc = len(re.findall(r'\w+', neighbor_text))
                
                # STRICT CONDITION: Neighbor must be longer than N words
                if neighbor_wc > MIN_LEN_NEIGHBOR:
                    dist = abs(i - idx)
                    # Find the closest one (lowest distance)
                    if dist < min_distance:
                        min_distance = dist
                        best_neighbor_idx = i
        
        # Replacement Logic (Use neighbor directly)
        if best_neighbor_idx != -1:
            neighbor_seg = all_segments[best_neighbor_idx]
            neighbor_text = neighbor_seg.get("text", "").strip()
            
            # Use neighbor's text and audio directly
            prompt_text = "You are a helpful assistant.<|endofprompt|>" + neighbor_text
            prompt_audio = os.path.join(clips_dir, f"segment_{best_neighbor_idx}.wav")
            
            print(f"[W] Replaced short prompt for segment {idx} ('{text}') with neighbor {best_neighbor_idx} ('{neighbor_text}')")
            return prompt_text, prompt_audio
            
        else:
            # 3. Fallback: No suitable neighbor found. Use duplication.
            print(f"[W] Segment {idx} is short & no neighbor > 4 words found. Fallback to duplication.")
            prompt_text = (text.strip() + " ") * (3 // word_count + 1)
            prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text.strip()
            
            base_audio_path = os.path.join(clips_dir, f"segment_{idx}.wav")
            base_audio = AudioSegment.from_wav(base_audio_path)
            repeat_times = (3 // word_count + 1)
            extended_audio = base_audio * repeat_times
            
            extended_audio_path = os.path.join(clips_dir, f"segment_{idx}_extended.wav")
            extended_audio.export(extended_audio_path, format="wav")
            return prompt_text, extended_audio_path

    # Normal case: Prompt is long enough
    return "You are a helpful assistant.<|endofprompt|>" + text, os.path.join(clips_dir, f"segment_{idx}.wav")

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

    for idx, segment in enumerate(segments):
        print(f"====> Processing segment {idx}")
        # Run inference for the current segment
        # The output is a generator, so we iterate through it
        target_text = segment.get(target_lang, None)# + "<|endofprompt|>"
        output_filename = f'tts_out_{target_lang}_segment_{idx}_output_{0}.wav' # the 0 can be a problem in future
        output_path = os.path.join("media", output_filename)

        # don't use translation for names or expressions
        if(is_name_or_expression(segment)): 
            print(f"[W] is_name_or_expression=True. Skipping translation for segment '{target_text}'")
            prompt_audio = os.path.join(clips_dir, f"segment_{idx}.wav")
            shutil.copy(prompt_audio, output_path); print(prompt_audio)
        else:
            prompt_text, prompt_audio = preprocess_prompts(idx, segment, clips_dir, segments)

            # translate with zero-shot cloning
            for i, j in enumerate(cosyvoice.inference_zero_shot(target_text, prompt_text, prompt_audio, stream=False)):
                torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)

            # print(output_path)
            # check+modify if the generation was successful
            output_path = check_and_postprocess(segment, prompt_audio, output_path)
            print(f"---> {prompt_text} | {prompt_audio} | {target_text}")

        # use segment to create new final audio
        seg_audio = AudioSegment.from_wav(output_path)
        # Place it at the start timestamp
        start_ms = int(segment.get("start") * 1000)
        final_dialogue_track = final_dialogue_track.overlay(seg_audio, position=start_ms)

    final_dialogue_track.export(os.path.join("media", "final_french_dialogue.wav"), format="wav")

if __name__ == '__main__':
    jpath = "/home/prashant/Documents/dubpls/media/_deadpool-2025-12-18_15.27.22_extracted_dialog_diarize_result.json"#"/home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22_extracted_dialog_diarize_result.json"
    apath = "/home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22_extracted_dialog.wav"
    env = os.environ.copy()
    # jpath = env["SESSION_JSON_PATH"]
    # apath = env["TEST_DIALOGUE_AUDIO_PATH"]
    tts(jpath, apath)