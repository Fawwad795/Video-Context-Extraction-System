import argparse
import os
import glob
from stream_pipeline import StreamPipeline

def run_offline(audio_dir, keywords):
    pipeline = StreamPipeline(keywords)
    
    # Process existing chunks in directory
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    # Sort logically by index
    def get_index(f):
        base = os.path.basename(f)
        try:
            return int(base.split("_")[1].split(".")[0])
        except:
            return 0
            
    audio_files.sort(key=get_index)
    
    print(f"Found {len(audio_files)} audio chunks in {audio_dir}.")
    for f in audio_files:
        idx = get_index(f)
        prefix = os.path.basename(f).split("_")[0]
        pipeline.process_chunk(idx, f, live_prefix=prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Keyword Detector")
    parser.add_argument("--url", type=str, help="YouTube Live URL to process")
    parser.add_argument("--audio-dir", type=str, help="Directory of offline .wav chunks")
    parser.add_argument("--keywords", type=str, required=True, help="Comma separated list of keywords")
    
    args = parser.parse_args()
    
    keywords = [k.strip() for k in args.keywords.split(",")]
    
    if args.audio_dir:
        run_offline(args.audio_dir, keywords)
    elif args.url:
        print("Live URL mode requires downloader.py to be running in parallel or integrated.")
        print("Please use gui.py or implement the live integration in detector.py.")
    else:
        print("Must provide either --url or --audio-dir")
