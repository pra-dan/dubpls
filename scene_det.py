from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from pathlib import Path
import shutil
import subprocess
import os

min_len_seconds = 2
SHOW_FFMPEG_VERBOSE=True
MIN_COLLECTION_DURATION=5.0 # seconds

MERGED_CLIPS_DIR = "merged_clips_per_segment"
SPLIT_SCENES_DIR = "split_scenes"

def get_scenes_per_segment(jpath: str, vpath: str):
    def split_video_into_scenes(video_path, threshold=27.0):
        # Open our video, create a scene manager, and add a detector.
        video = open_video(video_path)
        fps = video.frame_rate
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=threshold,
                            min_scene_len=int(fps)*min_len_seconds, # keeping min scene len as 1s
                            ))
        scene_manager.detect_scenes(video, show_progress=True)
        scene_list = scene_manager.get_scene_list()

        out_dir = Path(SPLIT_SCENES_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        split_video_ffmpeg(
            video_path,
            scene_list,
            output_dir=out_dir,
            show_progress=True,
            show_output=SHOW_FFMPEG_VERBOSE,
            # arg_override="-y -map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset veryfast -crf 22 -c:a aac",
        )

        return scene_list, out_dir

    scene_list, out_dir = split_video_into_scenes(vpath)
    # import pdb;pdb.set_trace()
    """
    ============================================================================= Mapping segment to scene
    """
    import json
    with open(jpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, segment in enumerate(data.get("segments", [])):
        start_time_seconds = segment.get("start", "")
        end_time_seconds = segment.get("end", "")

        def seconds_to_timestamp(sec):
            ms = int((sec - int(sec)) * 1000)
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            return f"{h:02}:{m:02}:{s:02}.{ms:03}"

        data["segments"][idx]["start_ts"] = seconds_to_timestamp(start_time_seconds)
        data["segments"][idx]["end_ts"] = seconds_to_timestamp(end_time_seconds)

    # map all segments to its scene 
    segments = data.get("segments", [])
    segment_idx, scene_idx = 0, 0
    segment2scene_map = {}

    try:
        while(segment_idx<len(segments) and scene_idx<len(scene_list)):
            # segment_start = segments[segment_idx]["start_ts"]
            segment_end = segments[segment_idx]["end_ts"]

            start_tc, end_tc = scene_list[scene_idx]
            print(f"[I] scene # {scene_idx}: start {start_tc} end {end_tc}")
            scene_start, scene_end = start_tc.get_timecode(), end_tc.get_timecode()
            if(scene_start <= segment_end and segment_end < scene_end):
                # segment falls in this scene. Save and check for next segment
                segment2scene_map[segment_idx] = scene_idx
                print(f"segment# {segment_idx}: {scene_start} <= {segment_end} and {segment_end} < {scene_end}")
                segment_idx+=1
            else:
                # move to next scene
                scene_idx+=1

        # If last segment got no match, give it the next/curr scene
        if(segment_idx not in list(segment2scene_map.keys())):
            segment2scene_map[segment_idx] = scene_idx if scene_idx<len(scene_list) else max(0, scene_idx-1)
            print(f"[I] For last segment ({segment_idx}), assigning scene {scene_idx}")

    except Exception as e:
        print(e)
        import pdb;pdb.set_trace()
        
    print(segment2scene_map)


    parsed_scene_times = []
    for start_tc, end_tc in scene_list:
        # Each element is a FrameTimecode object; use .get_timecode() to get the timestamp string
        parsed_scene_times.append((start_tc.get_timecode(), end_tc.get_timecode()))

    # NOTE: when fetching the scene clip from the scene list, just remember that the files are saved using 1-indexed method

    """
    ============================================================================= Collecting/Coalescing clips using length (for context extraction)
    """
    scene_collection_per_segment_map = {}

    for seg_idx in range(len(list(segment2scene_map.keys())[:-1]), -1, -1): # traversing segment ids in reverse order
        scn_idx = segment2scene_map[seg_idx]
        scene_duration = float(scene_list[scn_idx][1] - scene_list[scn_idx][0]) # converts scenedetect.frame_timecode.FrameTimecode to float seconds
        
        scene_collection_per_segment_map[seg_idx] = [scn_idx] # inits the dict 
        collected_duration = scene_duration
        while(collected_duration < MIN_COLLECTION_DURATION and scn_idx > 1):
            # add next scene until while fails
            scn_idx-=1
            scene_duration = float(scene_list[scn_idx][1] - scene_list[scn_idx][0])
            scene_collection_per_segment_map[seg_idx].append(scn_idx)
            collected_duration += scene_duration
        
        print(f"Scene collection: {seg_idx}: {scene_collection_per_segment_map[seg_idx]} | duration: {collected_duration}")
        # Scene collection: 11: [8, 7] | duration: 2.8
        # Scene collection: 8: [5] | duration: 9.0

    # merge and collect clips for each segment
    merged_clips_dir = MERGED_CLIPS_DIR
    os.makedirs(merged_clips_dir, exist_ok=True)

    for seg_idx in list(scene_collection_per_segment_map.keys()):
        input_clip_paths = []
        for clip_idx in scene_collection_per_segment_map[seg_idx]:
            clips_dir = SPLIT_SCENES_DIR
            clip_idx = clip_idx+1 # the clip indices are 0-indexed, while the clips are saved with names as 1-indexed
            clip_path = f"{os.path.splitext(os.path.basename(vpath))[0]}-Scene-{clip_idx:03d}.mp4"  # very specific to PySceneDet
            clip_path = os.path.join(clips_dir, clip_path)
            if not os.path.exists(clip_path):
                print(f"[W] Clip path doesn't exist {clip_path}")
            else:
                print(f"seg id {seg_idx} | clip {clip_idx} | path {clip_path}")
                input_clip_paths.append(clip_path)

        if input_clip_paths:
            input_clip_paths = sorted(input_clip_paths)
            # Build a file list in the format ffmpeg expects
            filelist_path = os.path.join(merged_clips_dir, f"seg{seg_idx:03d}_inputs.txt")
            with open(filelist_path, "w") as f:
                for p in input_clip_paths:
                    f.write(f"file '{os.path.abspath(p)}'\n")
            merged_output_path = os.path.join(merged_clips_dir, f"seg_{seg_idx:03d}.mp4")
            # ffmpeg concat (works for MP4 if codecs are same)
            ffmpeg_cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0", "-y", "-i", filelist_path, "-c", "copy", merged_output_path
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True)
                print(f"[I] Merged segment {seg_idx} saved to: {merged_output_path}")
                # save the collected-scenes vid path to segment
                data["segments"][seg_idx]["collected_scenes_path"] = merged_output_path
            except subprocess.CalledProcessError as e:
                print(f"[E] Error merging clips for segment {seg_idx}: {e}")
                # optionally: handle failure, fallback to alternative concat
            
            # Update json
            with open("temp.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
               
        else:
            print("[W] no input clip paths found!")

def main():
    """
    split_video_into_scenes
    map all segments to its scene 
    collect scenes for each segment

    env: scene_det
    """
    jpath = "/home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22_extracted_dialog_diarize_result.json"
    vpath = "/home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22.mp4"
    get_scenes_per_segment(jpath, vpath)


if __name__=='__main__':
    main()