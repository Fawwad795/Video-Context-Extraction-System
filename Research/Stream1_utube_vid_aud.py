import os
import argparse
import urllib.request
from moviepy import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy as mp
import streamlink
import m3u8
import threading
import time

# Define the global variables
chunk_index = 0
count = 0

def get_stream(url):
    try:
        streams = streamlink.streams(url)
        if not streams:
            raise Exception(f"No streams available for URL: {url}")
        
        if "best" not in streams:
            available = list(streams.keys())
            print(f"'best' stream not available. Available streams: {available}")
            raise Exception(f"'best' stream not found. Available: {available}")
        
        stream_url = streams["best"]
        print(f"Stream URL obtained: {stream_url.args.get('url', 'N/A')[:100]}...")
        
        m3u8_obj = m3u8.load(stream_url.args['url'])
        if not m3u8_obj.segments:
            raise Exception("No segments found in m3u8 playlist")
        
        return m3u8_obj.segments[0]
    except Exception as e:
        print(f"Error in get_stream: {e}")
        raise

def convert_video_to_audio(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)
    video_clip.close()

def download_chunks(url, filename):
    global chunk_index
    while True:
        stream_segment = get_stream(url)
        cur_time_stamp = stream_segment.program_date_time.strftime("%Y%m%d-%H%M%S")
        print(cur_time_stamp)

        video_file_path = videopath + filename + '_' + str(chunk_index) + '.mp4'
        audio_file_path = audiopath + filename + '_' + str(chunk_index) + '.wav'

        with urllib.request.urlopen(stream_segment.uri) as response:
            html = response.read()

            with open(video_file_path, 'wb') as file:
                file.write(html)

        # Convert video to audio
        if not os.path.exists(audio_file_path):
            print(f"Converting {video_file_path} to audio")
            convert_video_to_audio(video_file_path, audio_file_path)
            print(f"{video_file_path} converted to {audio_file_path}")

        chunk_index += 1

def download_thread(url, filename):
    import traceback
    retry_count = 0
    max_retries = 5
    
    while True:
        try:
            print(f"\n[Attempt {retry_count + 1}] Starting download from URL: {url}")
            download_chunks(url, filename)
        except Exception as e:
            retry_count += 1
            print(f"\n[Error - Attempt {retry_count}] During download: {e}")
            traceback.print_exc()
            
            if retry_count >= max_retries:
                print(f"Max retries ({max_retries}) reached. Exiting download thread.")
                break
            
            wait_time = min(30, 5 * retry_count)  # Exponential backoff: 5, 10, 15, 20, 25, 30 seconds
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Video and Audio Downloader")
    parser.add_argument("url", type=str, help="YouTube video URL")
    args = parser.parse_args()

    url = args.url

    # Directory paths - Windows/Linux compatible
    base_path = os.path.expanduser(os.path.join('~', 'VMS', 'GUI2CHjetson'))
    audiopath = os.path.join(base_path, 'Stream1audios') + os.sep
    videopath = os.path.join(base_path, 'Stream1videos') + os.sep

    if not os.path.isdir(audiopath):
        os.makedirs(audiopath)
        print('Directory Created for Audio files')

    if not os.path.isdir(videopath):
        os.makedirs(videopath)
        print('Directory Created for Videos')

    # Start the download thread as a daemon so it stops when main process exits
    download_t = threading.Thread(target=download_thread, args=(url, "live"), daemon=True)
    download_t.start()
    # Don't join() - let the thread run in background indefinitely
    download_t.join()  # Wait indefinitely for thread to complete (it's infinite)

