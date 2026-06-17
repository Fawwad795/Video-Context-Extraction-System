"""Download live-stream chunks for the PhonMatchNet detector.

Self-contained (streamlink + m3u8 + moviepy). Writes <out>/audios/live_N.wav
and <out>/videos/live_N.mp4. Defaults to ./new_chunks next to this script.

    python downloader.py                      # 10 chunks -> ./new_chunks/audios
    python downloader.py --url <yt_url> --n 10 --out ./new_chunks

Run with a Python that has streamlink, m3u8 and moviepy (see requirements-infra.txt).
"""
import argparse
import os
import time
import urllib.request

import m3u8
import streamlink
from moviepy.video.io.VideoFileClip import VideoFileClip

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URL = "https://www.youtube.com/watch?v=YDvsBbKfLPA"


def convert_video_to_audio(video_path, audio_path):
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, logger=None)
        clip.close()
        print(f"  -> {os.path.basename(audio_path)}")
    except Exception as e:
        print(f"  audio conversion error: {e}")


def get_stream(url):
    streams = streamlink.streams(url)
    if not streams or "best" not in streams:
        raise Exception("no 'best' stream available")
    obj = m3u8.load(streams["best"].args["url"])
    if not obj.segments:
        raise Exception("no segments in m3u8 playlist")
    return obj


def download_chunks(url, out_dir, n):
    video_dir = os.path.join(out_dir, "videos")
    audio_dir = os.path.join(out_dir, "audios")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    seen, idx = set(), 0
    print(f"Downloading {n} chunks from {url}")
    while idx < n:
        try:
            for seg in get_stream(url).segments:
                if idx >= n:
                    break
                if seg.uri in seen:
                    continue
                seen.add(seg.uri)
                vpath = os.path.join(video_dir, f"live_{idx}.mp4")
                apath = os.path.join(audio_dir, f"live_{idx}.wav")
                print(f"chunk {idx}...")
                with urllib.request.urlopen(seg.uri) as resp:
                    open(vpath, "wb").write(resp.read())
                convert_video_to_audio(vpath, apath)
                idx += 1
            if idx < n:
                time.sleep(2)
        except Exception as e:
            print(f"stream error: {e}")
            time.sleep(5)
    print(f"Done. {n} chunks in {audio_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download live-stream chunks.")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "new_chunks"))
    args = ap.parse_args()
    download_chunks(args.url, args.out, args.n)
