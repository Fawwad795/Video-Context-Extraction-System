import os
import time
import urllib.request
import streamlink
import m3u8
from moviepy.video.io.VideoFileClip import VideoFileClip

def convert_video_to_audio(video_path, audio_path):
    print(f"Converting {video_path} to audio...")
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path, logger=None)
        video_clip.close()
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

def download_eval_chunks(url, out_dir, target_chunks=60):
    os.makedirs(out_dir, exist_ok=True)
    video_dir = os.path.join(out_dir, "eval_videos")
    audio_dir = os.path.join(out_dir, "eval_audios")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    downloaded_uris = set()
    chunk_index = 0

    print(f"Starting download of {target_chunks} evaluation chunks from {url}")
    
    while chunk_index < target_chunks:
        try:
            m3u8_obj = get_stream(url)
            for segment in m3u8_obj.segments:
                if chunk_index >= target_chunks:
                    break
                if segment.uri not in downloaded_uris:
                    downloaded_uris.add(segment.uri)
                    
                    video_file_path = os.path.join(video_dir, f'eval_{chunk_index}.mp4')
                    audio_file_path = os.path.join(audio_dir, f'eval_{chunk_index}.wav')
                    
                    print(f"Downloading eval chunk {chunk_index}/{target_chunks}...")
                    with urllib.request.urlopen(segment.uri) as response:
                        html = response.read()
                        with open(video_file_path, 'wb') as file:
                            file.write(html)
                    
                    convert_video_to_audio(video_file_path, audio_file_path)
                    chunk_index += 1
            
            if chunk_index < target_chunks:
                time.sleep(2)
                
        except Exception as e:
            print(f"Error fetching stream: {e}")
            time.sleep(5)
            
    print(f"Successfully downloaded {target_chunks} evaluation chunks.")

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=YDvsBbKfLPA"
    base_path = r"d:\Video Context Extraction System\Siamese_VMS_Project"
    download_eval_chunks(youtube_url, base_path, target_chunks=60)
