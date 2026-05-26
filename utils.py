from typing import Dict
import ffmpeg
import json
import os
import tempfile
import shutil
import yaml

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except Exception as e:
        print(f"Failed to load config at {config_path}: {e}")
        return {}

def get_language_name(lang_code: str) -> str:
    mapping = {
        "fr": "French",
        "hi": "Hindi",
        "en": "English",
        "es": "Spanish"
    }
    return mapping.get(lang_code.lower(), lang_code.capitalize())

def isolate_media_streams(input_path):
    # Check file integrity
    if not os.path.exists(input_path):
        print("File path does not exist.")
        return None, None, False

    audio_out = None
    video_out = None
    success = False

    try:
        probe = ffmpeg.probe(input_path)
        streams = probe['streams']
        has_audio = any(s['codec_type'] == 'audio' for s in streams)
        has_video = any(s['codec_type'] == 'video' for s in streams)

        if not has_audio:
            print(f"No audio stream found in: {input_path}")
            return None, None, False

        base_name = os.path.splitext(input_path)[0]
        ext = os.path.splitext(input_path)[1].lower()

        # extract audio
        if ext == '.wav':
            audio_out = input_path
            print("File is already a WAV audio. Not modifying")
        else:
            audio_out = f"{base_name}_extracted.wav"
            ffmpeg.input(input_path).output(
                audio_out, acodec='pcm_s16le', ac=2, ar='44100'
            ).overwrite_output().run(quiet=True)

        # extract muted video
        if has_video:
            video_out = f"{base_name}_muted{ext}"
            ffmpeg.input(input_path).output(
                video_out, vcodec='copy', an=None
            ).overwrite_output().run(quiet=True)
            print(f"Video isolated: {video_out}")

        success = True

    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode() if e.stderr else e}")
    
    return audio_out, video_out, success

def get_segment_duration(segment: dict):
    """
    returns duration for a speech (speaker) segment
    """
    duration = 0.0
    duration = segment.get("end", None) - segment.get("start", None)
    return duration

# def collect_seconds_long_audio(json_path: str, audio_path: str, speaker_name: str = 'SPEAKER_01', seg_idx: int = 1, min_duration: float = 2):
#     """
#     For a specific speaker, it collects audio segments starting from current segment idx, towards 0
#     """
#     # collect all audios for a speaker, until duration > 2s
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
    
#     # import pdb;pdb.set_trace()
#     speaker_segments = []
#     for idx, segment in enumerate(data.get("segments", [])):
#         if (segment.get("speaker") == speaker_name and seg_idx >= idx):
#             speaker_segments.append(segment)

#     # collect recent audio segment upto target duration
#     rev_speaker_segments = speaker_segments[::-1]
#     duration = 0.0
#     idx = 0
#     curated_segments = []
#     while(duration < min_duration and idx < len(rev_speaker_segments)):
#         curated_segments.append(rev_speaker_segments[idx])
#         duration += get_segment_duration(rev_speaker_segments[idx])
#         idx += 1

#     # merge selected segments
#     path_to_merged_audio = clip_and_merge_audio_from_segments(curated_segments, audio_path)
#     return path_to_merged_audio

# def clip_and_merge_audio_from_segments(segments, input_audio_path=None, output_path=None):
#     """
#     Given a list of segments (with 'start' and 'end'), clips the corresponding portions
#     from an audio file and merges them into one file. Returns the path to the merged audio file.
#     """
#     # Guess the source audio if not given
#     if not segments:
#         print("[E] No segments provided!")
#         return None

#     if not os.path.isfile(input_audio_path):
#             raise FileNotFoundError(f"Input audio file {input_audio_path} not found.")

#     # Prepare temp directory for individual clips
#     temp_dir = tempfile.mkdtemp(prefix="clipmerge_")
#     temp_files = []

#     try:
#         for idx, seg in enumerate(segments):
#             start = seg.get("start")
#             end = seg.get("end")
#             if start is None or end is None or end <= start:
#                 print(f"Skipping segment {idx}: start/end invalid")
#                 continue
#             segment_out = os.path.join(temp_dir, f"clip_{idx}.wav")
#             (
#                 ffmpeg.input(input_audio_path, ss=start, to=end)
#                 .output(segment_out, acodec='pcm_s16le', ac=2, ar='44100', loglevel="error")
#                 .overwrite_output()
#                 .run()
#             )
#             temp_files.append(segment_out)

#         # Merge them together
#         if not temp_files:
#             print("[E] No valid segment clips to merge!")
#             return None

#         # If only one file, just copy it to output
#         if len(temp_files) == 1:
#             final_path = output_path or os.path.join(temp_dir, "merged_segments.wav")
#             shutil.copy(temp_files[0], final_path)
#             return final_path

#         # Prepare concat list file
#         concat_file = os.path.join(temp_dir, "concat_list.txt")
#         with open(concat_file, "w", encoding="utf-8") as f:
#             for path in temp_files:
#                 f.write(f"file '{os.path.abspath(path)}'\n")

#         if output_path is None:
#             output_path = "merged_segments.wav"

#         ffmpeg.input(concat_file, format="concat", safe=0).output(
#             output_path, acodec="pcm_s16le", ac=2, ar='44100', loglevel="error"
#         ).overwrite_output().run()
#         return output_path

#     finally:
#         # remove temp files
#         for f in temp_files:
#             try:
#                 if os.path.exists(f):
#                     os.remove(f)
#             except Exception as e:
#                 print(f"[W] Failed to delete temp file {f}: {e}")
#         if os.path.exists(temp_dir):
#             try:
#                 shutil.rmtree(temp_dir)
#             except Exception as e:
#                 print(f"[W] Failed to remove temp directory {temp_dir}: {e}")

if __name__ == "__main__":
    jpath = "/home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22_extracted_dialog_diarize_result.json"
    apath = "/home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22_extracted_dialog.wav"
    collect_seconds_long_audio(jpath, apath, 'SPEAKER_01', 6)