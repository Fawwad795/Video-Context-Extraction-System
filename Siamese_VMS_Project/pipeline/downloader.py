"""Download live-stream chunks into a named chunk set.

Writes audios/<set>/live_N.wav + videos/<set>/live_N.mp4 under the project
root. Segments are de-duplicated on downloaded bytes (md5): YouTube
re-signs every segment URL (fresh query-string token) on each playlist
fetch, so the *same* piece of content can arrive under a different
`segment.uri` on a later poll - URI-based dedup alone lets duplicate
chunks through.

Usage:
  python pipeline/downloader.py --set Chunkset_F --chunks 20
"""

import argparse
import hashlib
import os
import time
import urllib.request

import m3u8
import streamlink
from moviepy.video.io.VideoFileClip import VideoFileClip

# Shared modules live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

import console as ui

DEFAULT_URL = "https://www.youtube.com/watch?v=YDvsBbKfLPA"  # Sky News live
PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


def convert_video_to_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    try:
        clip.audio.write_audiofile(audio_path, logger=None)
    finally:
        clip.close()


def get_stream(url):
    streams = streamlink.streams(url)
    if not streams:
        raise RuntimeError(f"No streams available for URL: {url}")
    if "best" not in streams:
        raise RuntimeError("'best' stream not found.")
    m3u8_obj = m3u8.load(streams["best"].args["url"])
    if not m3u8_obj.segments:
        raise RuntimeError("No segments found in m3u8 playlist")
    return m3u8_obj


def download_chunks(url, video_dir, audio_dir, target_chunks):
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    downloaded_uris = set()
    downloaded_hashes = set()
    chunk_index = 0
    skipped_dupes = 0

    while chunk_index < target_chunks:
        try:
            m3u8_obj = get_stream(url)
            for segment in m3u8_obj.segments:
                if chunk_index >= target_chunks:
                    break
                if segment.uri in downloaded_uris:
                    continue
                downloaded_uris.add(segment.uri)

                with urllib.request.urlopen(segment.uri) as response:
                    content = response.read()

                content_hash = hashlib.md5(content).hexdigest()
                if content_hash in downloaded_hashes:
                    skipped_dupes += 1
                    ui.warn(f"duplicate segment content "
                            f"(hash {content_hash[:8]}) - skipped")
                    continue
                downloaded_hashes.add(content_hash)

                video_path = os.path.join(video_dir, f"live_{chunk_index}.mp4")
                audio_path = os.path.join(audio_dir, f"live_{chunk_index}.wav")
                with open(video_path, "wb") as f:
                    f.write(content)
                convert_video_to_audio(video_path, audio_path)
                ui.item(f"[{chunk_index + 1}/{target_chunks}] "
                        f"live_{chunk_index}.mp4 -> .wav")
                chunk_index += 1

            # Wait a bit for new segments to appear in the live stream
            if chunk_index < target_chunks:
                time.sleep(2)

        except Exception as e:
            ui.warn(f"error fetching stream: {e} - retrying in 5s")
            time.sleep(5)

    return skipped_dupes


def main():
    ap = argparse.ArgumentParser(description="Download live-stream chunks into a chunk set.")
    ap.add_argument("--url", default=DEFAULT_URL, help="live stream URL")
    ap.add_argument("--chunks", type=int, default=10, help="unique chunks to download")
    ap.add_argument("--set", dest="set_name", default=None,
                    help="chunk set name, e.g. Chunkset_F "
                         "(writes audios/<set>/ + videos/<set>/); "
                         "omit for flat audios/ + videos/")
    args = ap.parse_args()

    t0 = time.perf_counter()
    audio_dir = os.path.join(PROJECT_ROOT, "audios", args.set_name or "")
    video_dir = os.path.join(PROJECT_ROOT, "videos", args.set_name or "")

    ui.banner("STREAM CHUNK DOWNLOAD", args.set_name or "audios/")
    ui.kv("source", args.url)
    ui.kv("target", f"{args.chunks} unique chunks (content-hash dedup)")
    ui.step("downloading ...")

    skipped = download_chunks(args.url, video_dir, audio_dir, args.chunks)

    ui.ok(f"{args.chunks} unique chunks downloaded "
          f"({skipped} duplicate segment(s) skipped)")
    ui.done(os.path.relpath(audio_dir, PROJECT_ROOT), t0)
    ui.item("next: python pipeline/transcribe_chunks.py "
            + (f"(with SIAMESE_AUDIO_DIR=audios/{args.set_name})"
               if args.set_name else ""))


if __name__ == "__main__":
    main()
