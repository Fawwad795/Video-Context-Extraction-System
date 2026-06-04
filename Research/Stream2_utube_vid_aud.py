import os
import argparse
import urllib.request
from moviepy import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy as mp
import streamlink
import m3u8
import threading

# Define the global variables
chunk_index = 0
count = 0

def get_stream(url):
    streams = streamlink.streams(url)
    stream_url = streams["best"]

    m3u8_obj = m3u8.load(stream_url.args['url'])
    return m3u8_obj.segments[0]

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
    while True:
        try:
            download_chunks(url, filename)  # Download 10 video chunks
        except:
            print("Desired video not found. Trying to download again...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Video and Audio Downloader")
    parser.add_argument("url", type=str, help="YouTube video URL")
    args = parser.parse_args()

    url = args.url

    # Directory paths - Windows/Linux compatible
    base_path = os.path.expanduser(os.path.join('~', 'VMS', 'GUI2CHjetson'))
    audiopath = os.path.join(base_path, 'Stream2audios') + os.sep
    videopath = os.path.join(base_path, 'Stream2videos') + os.sep

    if not os.path.isdir(audiopath):
        os.makedirs(audiopath)
        print('Directory Created for Audio files')

    if not os.path.isdir(videopath):
        os.makedirs(videopath)
        print('Directory Created for Videos')

    # Start the download thread
    download_t = threading.Thread(target=download_thread, args=(url, "live"))
    download_t.start()
    download_t.join()

