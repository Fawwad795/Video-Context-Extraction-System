import hashlib
import os
import urllib.request
from moviepy.video.io.VideoFileClip import VideoFileClip
import streamlink
import m3u8
import time

def convert_video_to_audio(video_path, audio_path):
    print(f"Converting {video_path} to audio...")
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path, logger=None)
        video_clip.close()
        print(f"Success: {audio_path}")
    except Exception as e:
        print(f"Error converting video to audio: {e}")

def get_stream(url):
    streams = streamlink.streams(url)
    if not streams:
        raise Exception(f"No streams available for URL: {url}")
    if "best" not in streams:
        raise Exception("'best' stream not found.")

    stream_url = streams["best"]
    m3u8_obj = m3u8.load(stream_url.args['url'])
    if not m3u8_obj.segments:
        raise Exception("No segments found in m3u8 playlist")
    return m3u8_obj

def download_10_chunks(url, out_dir):
    """De-duplication note: YouTube re-signs every segment URL (fresh
    query-string token) on each playlist fetch, so the *same* piece of
    content can arrive under a different `segment.uri` on a later poll -
    URI-based dedup alone lets duplicate chunks through. We dedupe on the
    actual downloaded bytes (md5) instead, which is correct regardless of
    URL signing or playlist windowing.
    """
    os.makedirs(out_dir, exist_ok=True)
    video_dir = os.path.join(out_dir, "videos")
    audio_dir = os.path.join(out_dir, "audios")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    downloaded_uris = set()
    downloaded_hashes = set()
    chunk_index = 0
    skipped_dupes = 0
    target_chunks = 10

    print(f"Starting download of {target_chunks} chunks from {url}")

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
                    print(f"  Skipping duplicate segment content (hash {content_hash[:8]}) "
                          f"- same bytes as a chunk already saved.")
                    continue
                downloaded_hashes.add(content_hash)

                video_file_path = os.path.join(video_dir, f'live_{chunk_index}.mp4')
                audio_file_path = os.path.join(audio_dir, f'live_{chunk_index}.wav')

                print(f"Downloading chunk {chunk_index}...")
                with open(video_file_path, 'wb') as file:
                    file.write(content)

                convert_video_to_audio(video_file_path, audio_file_path)
                chunk_index += 1

            # Wait a bit for new segments to appear in the live stream
            if chunk_index < target_chunks:
                time.sleep(2)

        except Exception as e:
            print(f"Error fetching stream: {e}")
            time.sleep(5)

    print(f"Successfully downloaded {target_chunks} unique chunks "
          f"({skipped_dupes} duplicate segment(s) skipped).")

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=YDvsBbKfLPA"
    base_path = r"d:\Video Context Extraction System\Siamese_VMS_Project"
    download_10_chunks(youtube_url, base_path)
