"""
Implements part B. Speaker Diarization & Identification
Input: Isolated dialog audio (wav) path
Output: SRT file with timestamps

Reference: WhisperX Official example: https://github.com/m-bain/whisperX/issues/902#issuecomment-2646634513
"""

import whisperx
import gc
import os
import gc
from whisperx.diarize import DiarizationPipeline

device = "cuda"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

def diarize_with_forced_alignment(dialog_wav_path, hf_read_token):
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    audio = whisperx.load_audio(dialog_wav_path)
    result = model.transcribe(audio, batch_size=batch_size)
    # print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # print(result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    diarize_model = DiarizationPipeline(use_auth_token=hf_read_token, device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    # print(diarize_segments)
    # print(result["segments"]) # segments are now assigned speaker IDs
    # print(result)
    
    import json
    # Save result dictionary as a JSON file
    json_path = os.path.splitext(dialog_wav_path)[0] + "_diarize_result.json"
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)
        print("diarization results saved to ",json_path)

    del(model)
    del(diarize_model)
    gc.collect()
    return json_path

if __name__ == '__main__':
    diarize_with_forced_alignment("media/deadpool-2025-12-18_15.27.22_extracted_dialog.wav", os.getenv('HF_READ_TOKEN'))