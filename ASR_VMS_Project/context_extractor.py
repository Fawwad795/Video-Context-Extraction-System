import os
import subprocess
from config import WORK_DIR, N_BEFORE, N_AFTER

def extract(center_index, keyword, live_prefix="live"):
    """
    Build list of live_{center-N_BEFORE..center+N_AFTER}.mp4 that exist, 
    concat with FFmpeg concat demuxer into detections/{keyword}/{keyword}_detected_at_{center}.mp4
    """
    keyword_dir = os.path.join(WORK_DIR, "detections", keyword)
    os.makedirs(keyword_dir, exist_ok=True)
    
    out_file = os.path.join(keyword_dir, f"{keyword}_detected_at_{center_index}.mp4")
    if os.path.exists(out_file):
        return out_file
        
    start_idx = max(0, center_index - N_BEFORE)
    end_idx = center_index + N_AFTER
    
    concat_list_path = os.path.join(WORK_DIR, f"concat_list_{center_index}.txt")
    
    try:
        # Write concat list
        with open(concat_list_path, "w") as f:
            for i in range(start_idx, end_idx + 1):
                chunk_path = os.path.join(WORK_DIR, "videos", f"{live_prefix}_{i}.mp4")
                if os.path.exists(chunk_path):
                    # FFmpeg requires forward slashes and absolute paths or relative paths in a specific format
                    # Easiest is to use absolute paths with forward slashes
                    f_path = chunk_path.replace("\\", "/")
                    f.write(f"file '{f_path}'\n")
                    
        # Run ffmpeg concat
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", concat_list_path, "-c", "copy", out_file
        ]
        
        # Suppress output to avoid spamming the console
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"Extracted context video saved to: {out_file}")
        return out_file
        
    except Exception as e:
        print(f"Error extracting context: {e}")
        return None
    finally:
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)

def safe_remove(filepath, retries=3):
    import time
    for i in range(retries):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            break
        except Exception:
            time.sleep(1)
