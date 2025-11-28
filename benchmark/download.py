import os
import subprocess

# List of [URL, start_time, end_time, file_name]
data = [
    ["https://www.youtube.com/watch?v=3ePLfh4LRlc", "00:00:00.000","00:00:01.000", "interact_1"],
    ["https://www.youtube.com/watch?v=yrOGWaoRfKQ", "00:00:03.000","00:00:04.000", "interact_2"],
    ["https://www.youtube.com/watch?v=K95kpB0zNqc", "00:00:39.000","00:00:40.000", "interact_3"],
    ["https://www.youtube.com/watch?v=3ch44DSNnsM", "00:00:01.000","00:00:02.000", "interact_4"],
    ["https://www.youtube.com/watch?v=OZqGUFivO3o", "00:04:12.000","00:04:13.000", "interact_5"],
    ["https://www.youtube.com/watch?v=j-aEBQpeEy4", "00:06:00.000","00:06:01.000", "interact_6"],
    ["https://www.youtube.com/watch?v=BkhsWikdKq8", "00:02:09.000","00:02:10.000", "interact_7"],
    ["https://www.youtube.com/watch?v=7EOE4nU4Gks", "00:02:48.000","00:02:49.000", "interact_8"],
    ["https://www.youtube.com/watch?v=616yuKq1aeg", "00:00:01.000","00:00:02.000", "interact_9"],
    ["https://www.youtube.com/watch?v=LFdQ7ZiSDAs", "00:00:39.000","00:00:40.000", "interact_18"],
    ["https://www.youtube.com/watch?v=K9wBEUaTEQU", "00:02:22.000","00:02:23.000", "interact_11"],
    ["https://www.youtube.com/watch?v=kHmgRtNWOi0", "00:00:08.000","00:00:09.000", "interact_12"],
    ["https://www.youtube.com/watch?v=iJhkuhrz5OE", "00:15:09.000","00:15:10.000", "interact_13"],
    ["https://www.youtube.com/watch?v=yteOxz21duo", "00:07:00.000","00:07:01.000", "interact_14"],
    ["https://www.youtube.com/watch?v=0pELFCq43cw", "00:00:05.000", "00:00:06.000", "interact_15"],
    ["https://www.youtube.com/watch?v=aN7nKjBVD2o", "00:10:31.000", "00:10:32.000", "interact_16"],
    ["https://www.youtube.com/watch?v=3lC7N4c6Kpo", "00:01:40.000", "00:01:41.000", "interact_17"],
    ["https://www.youtube.com/watch?v=-112sJWNAYw", "00:00:59.000", "00:01:00.000", "interact_19"],
    ["https://www.youtube.com/watch?v=meQn39gSXe4", "00:00:06.000", "00:00:07.000", "interact_20"],
    ["https://www.youtube.com/watch?v=8e3Xxfsh2M4", "00:11:56.000", "00:11:57.000", "interact_21"],
    ["https://www.youtube.com/watch?v=3tKsPdQ1s8Q", "00:02:10.000", "00:02:11.000", "interact_22"],
    ["https://www.youtube.com/watch?v=2Znrk3JyKks", "00:35:52.000", "00:35:53.000", "interact_23"],
    ["https://www.youtube.com/watch?v=YacHIjPkK7k", "00:01:15.000", "00:01:16.000", "interact_24"],
    ["https://www.youtube.com/watch?v=3mzUmy4oqCI", "00:09:24.000", "00:09:25.000", "interact_25"],
    ["https://www.youtube.com/watch?v=6e8TE6n_R9s", "00:01:08.000", "00:01:09.000", "interact_26"],
    ["https://www.youtube.com/watch?v=-qb-FZzN2rY", "00:00:32.000", "00:00:33.000", "interact_27"],
    ["https://www.youtube.com/watch?v=VCynlBiZsdI", "00:00:06.000", "00:00:07.000", "interact_28"],
    ["https://www.youtube.com/watch?v=lm83VcaAhYA", "00:00:00.000", "00:00:01.000", "interact_29"],
]

# Root directories for temporary videos and output frames
VIDEO_DIR = "videos"
FRAME_DIR = "frames"

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(os.path.join(FRAME_DIR, "input_images"), exist_ok=True)
os.makedirs(os.path.join(FRAME_DIR, "output_images"), exist_ok=True)


def normalize_timecode(t: str) -> str:
    """
    Normalize timecode format.
    Accepts both 'HH:MM:SS.mmm' and 'HH:MM:SS:mmm', returns 'HH:MM:SS.mmm'.
    """
    parts = t.split(":")
    # If format is HH:MM:SS:mmm -> convert to HH:MM:SS.mmm
    if len(parts) == 4:
        h, m, s, ms = parts
        return f"{h}:{m}:{s}.{ms}"
    return t


def download_youtube_video(video_url: str, output_path: str) -> bool:
    """
    Download a YouTube video using yt-dlp.
    Returns True if download succeeds, False otherwise.
    """
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        command = [
            "yt-dlp",
            "-f",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "--merge-output-format", "mp4",
            "--output", output_path,
            video_url,
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )

        if result.returncode == 0:
            print(f"Download {video_url} successfully!")
            return True
        else:
            print(f"Fail to download {video_url}, error info:\n{result.stderr}")
            return False

    except Exception as e:
        print(f"Error downloading {video_url}: {e}")
        return False


def extract_first_frame_at_time(video_path: str, timestamp: str, output_image_path: str) -> bool:
    """
    Extract a single frame at a given timestamp using ffmpeg.
    """
    # Normalize the timecode to HH:MM:SS.mmm
    ts = normalize_timecode(timestamp)

    # ffmpeg -ss <ts> -i input.mp4 -frames:v 1 -q:v 2 output.jpg
    command = [
        "ffmpeg",
        "-ss", ts,
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-y",  # overwrite output if exists
        output_image_path,
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )

        if result.returncode == 0:
            print(f"Extract frame at {ts} from {video_path} -> {output_image_path}")
            return True
        else:
            print(f"Fail to extract frame from {video_path}, error info:\n{result.stderr}")
            return False

    except Exception as e:
        print(f"Error extracting frame from {video_path}: {e}")
        return False


def main():
    """
    Main pipeline:
    1. Download each YouTube video.
    2. Extract one frame at the given start time.
    3. Delete the temporary video file.
    """
    for url, start_time, end_time, file_name in data:
        print("=" * 80)
        print(f"Processing: {url} | start={start_time} | file_name={file_name}")

        video_path = os.path.join(VIDEO_DIR, f"{file_name}.mp4")
        frame_path = os.path.join(FRAME_DIR, f"{file_name}.jpg")

        # Step 1: download video
        ok = download_youtube_video(url, video_path)
        if not ok:
            print(f"Skip {url} due to download failure.")
            continue

        # Step 2: extract first frame at start_time
        ok = extract_first_frame_at_time(video_path, start_time, frame_path)
        if not ok:
            print(f"Frame extraction failed for {video_path}, keep video for debugging.")
            continue

        # Step 3: delete original video after successful frame extraction
        try:
            os.remove(video_path)
            print(f"Deleted temporary video: {video_path}")
        except OSError as e:
            print(f"Error deleting video {video_path}: {e}")


if __name__ == "__main__":
    main()
