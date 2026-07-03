import os
import subprocess
import time
import re
from moviepy import VideoFileClip

# ======== Paths Setup - Windows/Linux compatible ========
base_path = os.path.expanduser(os.path.join('~', 'VMS', 'GUI2CHjetson'))
video_path = os.path.join(base_path, '..', 'american.mp4')
video_out = os.path.join(base_path, 'Stream1videos')
audio_out = os.path.join(base_path, 'Stream1audios')
searchword_dir = os.path.join(base_path, 'Stream1_searchword1')
detection_log = os.path.join(base_path, 'Stream1_detection', 'keyword', 'timestamps.txt')
keyword_audio = os.path.join(searchword_dir, 'dummy-keyword.wav')
correlation_script = os.path.join(base_path, 'run_correlation.py')  # adjust name if needed

# ======== Create Folders ========
os.makedirs(video_out, exist_ok=True)
os.makedirs(audio_out, exist_ok=True)

# ======== Step 1: Split Video ========
clip = VideoFileClip(video_path)
duration = int(clip.duration)

print(f"[INFO] Splitting video into {duration//5} chunks...")
for i in range(0, duration, 5):
    subclip = clip.subclip(i, min(i + 5, duration))
    video_file = os.path.join(video_out, f'live_{i//5}.mp4')
    audio_file = os.path.join(audio_out, f'live_{i//5}.wav')

    subclip.write_videofile(video_file, codec='libx264', audio_codec='aac', verbose=False, logger=None)
    subclip.audio.write_audiofile(audio_file, verbose=False, logger=None)

print("[INFO] Video split completed.")

# ======== Step 2: Wait for HiFi-GAN Output ========
print(f"[INFO] Waiting for HiFi-GAN output at {keyword_audio}...")
while not os.path.exists(keyword_audio):
    time.sleep(1)

print("[INFO] Keyword audio detected.")

# ======== Step 3: Run Correlation Detection Script ========
print(f"[INFO] Running correlation detection...")
subprocess.run(['python3', correlation_script])

# ======== Step 4: Evaluate TP, FP, TN, FN ========
print(f"[INFO] Evaluating detection results...")

detected = []
if os.path.exists(detection_log):
    with open(detection_log, "r") as f:
        for line in f:
            match = re.search(r"live_(\d+)", line)
            if match:
                detected.append(int(match.group(1)))

# Update this based on known keyword timings (5s intervals)
actual = set(range(1, 4))  # Example: keyword in segments 1, 2, 3
all_segments = set(range(duration // 5))

tp = len([x for x in detected if x in actual])
fp = len([x for x in detected if x not in actual])
tn = len([x for x in all_segments if x not in detected and x not in actual])
fn = len([x for x in actual if x not in detected])

print(f"\n--- Detection Results ---")
print(f"True Positives:  {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives:  {tn}")
print(f"False Negatives: {fn}")

