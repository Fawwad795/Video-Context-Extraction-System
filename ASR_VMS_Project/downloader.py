import os
import time
import json
import urllib.request
import streamlink
import m3u8
import re
from moviepy.video.io.VideoFileClip import VideoFileClip
from config import WORK_DIR

def convert_video_to_audio(video_path, audio_path):
    print(f"Converting {video_path} to audio...")
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path, logger=None)
        duration = video_clip.duration
        video_clip.close()
        print(f"Success: {audio_path} (Duration: {duration}s)")
        return duration
    except Exception as e:
        print(f"Error converting video to audio: {e}")
        return 5.0

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

def run_downloader(url, target_chunks=15, live_prefix="live"):
    print(f"Starting downloader for {url} with prefix {live_prefix}...")
    
    video_dir = os.path.join(WORK_DIR, "videos")
    audio_dir = os.path.join(WORK_DIR, "audios")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    # Clear old chunks
    for f in os.listdir(video_dir):
        if f.startswith(live_prefix): os.remove(os.path.join(video_dir, f))
    for f in os.listdir(audio_dir):
        if f.startswith(live_prefix): os.remove(os.path.join(audio_dir, f))

    durations_file = os.path.join(WORK_DIR, "durations.json")
    durations = {}

    downloaded_uris = set()
    chunk_index = 0

    while chunk_index < target_chunks:
        try:
            m3u8_obj = get_stream(url)
            for segment in m3u8_obj.segments:
                if chunk_index >= target_chunks:
                    break
                    
                # YouTube generates new signed URIs every time get_stream is called.
                # We must deduplicate using the sequence number (sq) to avoid downloading duplicates.
                seq_match = re.search(r'sq=(\d+)', segment.uri)
                segment_id = seq_match.group(1) if seq_match else segment.uri
                
                if segment_id not in downloaded_uris:
                    downloaded_uris.add(segment_id)
                    
                    video_file_path = os.path.join(video_dir, f'{live_prefix}_{chunk_index}.mp4')
                    audio_file_path = os.path.join(audio_dir, f'{live_prefix}_{chunk_index}.wav')
                    
                    print(f"Downloading chunk {chunk_index+1}/{target_chunks}...")
                    with urllib.request.urlopen(segment.uri) as response:
                        html = response.read()
                        with open(video_file_path, 'wb') as file:
                            file.write(html)
                    
                    duration = convert_video_to_audio(video_file_path, audio_file_path)
                    durations[f"{live_prefix}_{chunk_index}"] = duration
                    
                    with open(durations_file, "w") as f:
                        json.dump(durations, f)
                        
                    chunk_index += 1
            
            if chunk_index < target_chunks:
                time.sleep(2)
                
        except Exception as e:
            print(f"Error fetching stream: {e}")
            time.sleep(5)
            
    print(f"Successfully downloaded {target_chunks} chunks.")

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=YDvsBbKfLPA"
    run_downloader(url, target_chunks=15)
