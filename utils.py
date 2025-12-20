import ffmpeg
import os

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