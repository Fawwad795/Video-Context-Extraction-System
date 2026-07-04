"""Live monitoring worker for the Siamese KWS platform.

Launched by platform/gui.py as a subprocess. One keyword + one live-stream
URL per run. Everything it reads/writes lives under the platform data root
(platform/data by default) - never in the research folders - via the
SIAMESE_PROJECT_ROOT override in core/scoring.py.

Architecture: downloading and detection are decoupled.

  StreamDownloader (thread) - from the moment the worker starts until the
      user clicks Finish, polls the stream playlist and downloads EVERY new
      segment (dedup by URI path, since YouTube re-signs query strings, plus
      content md5 as a backstop), converts it to wav, and enqueues it for
      the detector. Downloading never pauses for processing.

  Main thread - setup phase first (build missing keyword artifacts by
      invoking the original pipeline scripts as subprocesses):
        keyword_generator.py     TTS prototype anchor + variants
        (bootstrap wait)         until N chunks have arrived from the stream
        cohort_builder.py        AS-norm impostor cohort
        calibrate.py             per-keyword detection threshold
      Then the endless live loop: take the next chunk off the queue, scan it
      with the AS-norm detector (50 ms hop - must match the calibration
      window grid), then:
        detection          -> chunk audio+video moved to
                              detections/<keyword>/ with a JSON record
        no detection       -> chunk audio+video deleted

Disk/memory stay flat no matter how long the run is: processed chunks are
deleted (or moved to detections/), and the unprocessed backlog is capped -
if the detector cannot keep up with the stream, the oldest unprocessed
chunk is dropped (during setup, new segments are skipped instead, so the
files setup is reading stay stable).

Status protocol on stdout (parsed by the GUI):
  @@PHASE <setup|live|error>
  @@STATUS <one-line status>
  @@DETECT <total detections this run>
  @@QUEUE <unprocessed chunks waiting>
"""

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime

PLATFORM_DIR = os.path.dirname(os.path.abspath(__file__))
SIAMESE_ROOT = os.path.dirname(PLATFORM_DIR)
PIPELINE_DIR = os.path.join(SIAMESE_ROOT, "pipeline")
CORE_DIR = os.path.join(SIAMESE_ROOT, "core")
DEFAULT_DATA_ROOT = os.path.join(PLATFORM_DIR, "data")
DEFAULT_WEIGHTS = os.path.join(SIAMESE_ROOT, "checkpoints", "siamese_v3_best.pth")

# 50 ms hop - must match the calibration window grid (calibrate.py samples
# negatives on the same hop, so the fitted threshold assumes it).
STEP_SECONDS = 0.05
SCALES = (0.6, 0.8, 1.0)
BATCH_SIZE = 16

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRINT_LOCK = threading.Lock()


def emit(tag, msg):
    with PRINT_LOCK:
        print(f"@@{tag} {msg}", flush=True)


def status(msg):
    emit("STATUS", msg)


def plain(msg):
    with PRINT_LOCK:
        print(msg, flush=True)


# --------------------------------------------------------------------------
# Setup phase: run the original pipeline scripts against the platform root
# --------------------------------------------------------------------------

def run_step(script, args, desc):
    status(f"Setup: {desc} ...")
    cmd = [sys.executable, "-u", os.path.join(PIPELINE_DIR, script)] + args
    p = subprocess.Popen(cmd, cwd=PIPELINE_DIR, env=os.environ.copy(),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in p.stdout:
        plain(f"  [{script}] {line.rstrip()}")
    if p.wait() != 0:
        raise RuntimeError(f"{script} failed (exit code {p.returncode})")


# --------------------------------------------------------------------------
# Continuous stream download (never stops until the worker is killed)
# --------------------------------------------------------------------------

def resolve_playlist_url(url):
    import streamlink
    streams = streamlink.streams(url)
    if not streams or "best" not in streams:
        raise RuntimeError(f"no 'best' stream available for URL: {url}")
    return streams["best"].args["url"]


def convert_video_to_audio(video_path, audio_path):
    from moviepy.video.io.VideoFileClip import VideoFileClip
    clip = VideoFileClip(video_path)
    try:
        clip.audio.write_audiofile(audio_path, logger=None)
    finally:
        clip.close()


class StreamDownloader(threading.Thread):
    """Downloads every new stream segment, indefinitely.

    Dedup: segment URI path (YouTube re-signs the query string on every
    playlist fetch, so the path - which carries the sequence number - is the
    stable identity) plus downloaded-bytes md5 as a backstop. Both sets are
    bounded so a multi-hour run cannot grow memory without limit.

    Chunks with index < protect_below are the setup bootstrap set: they are
    written to disk but NOT enqueued (the main thread processes them from
    disk after setup, and setup scripts read them meanwhile).

    Backlog cap policy when the detector falls behind:
      setup_mode=True  - skip new segments once the queue is full, so files
                         that setup scripts are reading never get deleted;
      setup_mode=False - drop the OLDEST unprocessed chunk (delete its
                         files), keeping the monitor close to live.
    """

    def __init__(self, url, video_dir, audio_dir, start_index, protect_below,
                 chunk_queue, backlog_cap, max_remember=8192):
        super().__init__(daemon=True)
        self.url = url
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.index = start_index
        self.protect_below = protect_below
        self.queue = chunk_queue
        self.backlog_cap = backlog_cap
        self.max_remember = max_remember
        self.setup_mode = True
        self.downloaded = 0
        self.skipped_full = 0
        self.seen_keys = {}
        self.seen_hashes = {}

    def _remember(self, d, key):
        d[key] = None
        if len(d) > self.max_remember:
            for k in list(d)[: len(d) - self.max_remember]:
                del d[k]

    def _handle_segment(self, uri):
        """Download one new segment; returns True if a chunk was kept."""
        if self.setup_mode and self.queue.qsize() >= self.backlog_cap:
            self.skipped_full += 1
            if self.skipped_full % 25 == 1:
                status(f"Backlog full during setup - skipping live segments "
                       f"({self.skipped_full} skipped so far)")
            return False

        with urllib.request.urlopen(uri, timeout=30) as r:
            content = r.read()
        content_hash = hashlib.md5(content).hexdigest()
        if content_hash in self.seen_hashes:
            return False
        self._remember(self.seen_hashes, content_hash)

        idx = self.index
        video_path = os.path.join(self.video_dir, f"live_{idx}.mp4")
        audio_path = os.path.join(self.audio_dir, f"live_{idx}.wav")
        with open(video_path, "wb") as f:
            f.write(content)
        try:
            convert_video_to_audio(video_path, audio_path)
        except Exception as e:
            status(f"Audio conversion failed for chunk {idx} ({e}) - dropped")
            for p in (video_path, audio_path):
                if os.path.exists(p):
                    os.remove(p)
            return False
        self.index += 1
        self.downloaded += 1

        if idx < self.protect_below:
            status(f"Bootstrap chunk live_{idx} downloaded "
                   f"({self.protect_below - idx - 1} more needed for setup)")
            return True

        # Live cap: drop the oldest unprocessed chunk(s) to stay near-live.
        while self.queue.qsize() >= self.backlog_cap:
            try:
                old_video, old_audio, old_idx = self.queue.get_nowait()
            except queue.Empty:
                break
            for p in (old_video, old_audio):
                if os.path.exists(p):
                    os.remove(p)
            status(f"Backlog cap ({self.backlog_cap}) hit - dropped "
                   f"unprocessed chunk live_{old_idx}")
        self.queue.put((video_path, audio_path, idx))
        emit("QUEUE", str(self.queue.qsize()))
        return True

    def run(self):
        import m3u8
        playlist_url = None
        while True:
            try:
                if playlist_url is None:
                    status("Resolving stream playlist ...")
                    playlist_url = resolve_playlist_url(self.url)
                    status("Stream connected - downloading chunks continuously")
                playlist = m3u8.load(playlist_url)
                got_new = False
                for seg in playlist.segments:
                    uri = seg.absolute_uri
                    key = uri.split("?", 1)[0]
                    if key in self.seen_keys:
                        continue
                    self._remember(self.seen_keys, key)
                    try:
                        got_new = self._handle_segment(uri) or got_new
                    except Exception as e:
                        status(f"Segment download error ({e}) - skipping")
                time.sleep(0.4 if got_new else 1.5)
            except Exception as e:
                status(f"Stream error ({e}) - reconnecting in 5s")
                playlist_url = None
                time.sleep(5)


def next_live_index(*dirs):
    idx = -1
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.startswith("live_"):
                try:
                    idx = max(idx, int(name.split("_")[1].split(".")[0]))
                except (IndexError, ValueError):
                    pass
    return idx + 1


# --------------------------------------------------------------------------
# Detection on a single chunk (per-chunk port of pipeline/detector.py)
# --------------------------------------------------------------------------

def scan_chunk(y, model, anchor, cohort, window_samples, threshold,
               sample_rate, top_k):
    import numpy as np
    from scoring import asnorm_windows, embed_batch

    step_samples = max(1, int(sample_rate * STEP_SECONDS))
    all_scores, all_raw, all_times, all_scales = [], [], [], []
    for scale in SCALES:
        ws = max(int(window_samples * scale), int(0.15 * sample_rate))
        starts = list(range(0, len(y) - ws + 1, step_samples))
        if not starts:
            continue
        windows = [y[s:s + ws].astype(np.float32) for s in starts]
        embs = embed_batch(model, windows, batch_size=BATCH_SIZE)
        normed, raw = asnorm_windows(embs, anchor, cohort, top_k)
        all_scores.append(normed)
        all_raw.append(raw)
        all_times.append(np.array(starts) / sample_rate)
        all_scales.append(np.full(len(starts), scale))

    if not all_scores:
        return None  # chunk shorter than the keyword window

    normed = np.concatenate(all_scores)
    raw = np.concatenate(all_raw)
    times = np.concatenate(all_times)
    win_scales = np.concatenate(all_scales)

    hit_idx = np.where(normed >= threshold)[0]
    detections = [{"time": float(times[i]), "score": float(normed[i]),
                   "raw_cos": float(raw[i]), "scale": float(win_scales[i])}
                  for i in hit_idx]
    detections.sort(key=lambda d: -d["score"])

    best = int(np.argmax(normed))
    return {"detections": detections,
            "best_score": float(normed[best]), "best_time": float(times[best]),
            "best_scale": float(win_scales[best])}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Siamese KWS live platform worker.")
    ap.add_argument("--keyword", required=True)
    ap.add_argument("--url", required=True, help="live stream URL")
    ap.add_argument("--root", default=DEFAULT_DATA_ROOT,
                    help="platform data root (isolated from the research folders)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--bootstrap-chunks", type=int, default=10,
                    help="chunks needed up-front for cohort/calibration")
    ap.add_argument("--backlog-cap", type=int, default=300,
                    help="max unprocessed chunks kept on disk before dropping")
    args = ap.parse_args()

    keyword = args.keyword.strip().lower()
    data_root = os.path.abspath(args.root)

    # All pipeline scripts + core modules resolve their paths through this.
    os.environ["SIAMESE_PROJECT_ROOT"] = data_root
    os.environ["SIAMESE_WEIGHTS"] = args.weights
    os.environ.setdefault("SIAMESE_BACKEND", "wavlm-trained")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    audio_dir = os.path.join(data_root, "audios")
    video_dir = os.path.join(data_root, "videos")
    kw_dir = os.path.join(data_root, "keywords")
    log_dir = os.path.join(data_root, "logs")
    det_dir = os.path.join(data_root, "detections", keyword)
    for d in (audio_dir, video_dir, kw_dir, log_dir, det_dir):
        os.makedirs(d, exist_ok=True)

    sys.path.insert(0, CORE_DIR)
    sys.path.insert(0, PIPELINE_DIR)

    emit("PHASE", "setup")

    # Artifact names carry a backend suffix (e.g. _wavlm10ft for the
    # production wavlm-trained backend) - resolve them through scoring so
    # the "built once per keyword, reused after" check matches the files
    # the pipeline scripts actually write. scoring reads the env overrides
    # set above at import time, so this import must come after them.
    from scoring import anchor_path, calibration_path, cohort_path
    anchor_file = anchor_path(keyword)
    cohort_file = cohort_path(keyword)
    calib_file = calibration_path(keyword)
    needs_setup = not all(os.path.exists(p) for p in
                          (anchor_file, cohort_file, calib_file))

    # Chunks left on disk by a previous run (killed mid-processing): they are
    # scanned before the live queue, and count toward the bootstrap set.
    pre_existing = sorted(
        (os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
         if f.startswith("live_") and f.endswith(".wav")),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))

    start_index = next_live_index(audio_dir, video_dir)
    protect_below = start_index
    if needs_setup:
        protect_below += max(0, args.bootstrap_chunks - len(pre_existing))

    # -- Downloading starts NOW and never stops until the user hits Finish --
    chunk_queue = queue.Queue()
    downloader = StreamDownloader(args.url, video_dir, audio_dir, start_index,
                                  protect_below, chunk_queue, args.backlog_cap)
    downloader.start()

    # -- 1. TTS prototype anchor (also creates <kw>_variants/) --------------
    if not os.path.exists(anchor_file):
        run_step("keyword_generator.py", ["--keyword", keyword],
                 f"synthesizing multi-voice TTS anchor for '{keyword}' "
                 "(first run downloads TTS models - please wait; the stream "
                 "keeps downloading in the background)")

    # -- 2. Wait until enough stream audio has arrived for setup ------------
    def wav_count():
        return len([f for f in os.listdir(audio_dir)
                    if f.startswith("live_") and f.endswith(".wav")])

    if needs_setup:
        if wav_count() < args.bootstrap_chunks:
            status(f"Waiting for {args.bootstrap_chunks} bootstrap chunks "
                   f"from the stream ...")
        while wav_count() < args.bootstrap_chunks:
            time.sleep(2)

    # -- 3. Cohort + threshold calibration ----------------------------------
    if not os.path.exists(cohort_file):
        run_step("cohort_builder.py", ["--keyword", keyword],
                 "building the AS-norm impostor cohort")
    if not os.path.exists(calib_file):
        run_step("calibrate.py", ["--keyword", keyword],
                 "calibrating the detection threshold")

    # -- 4. Load runtime models once ----------------------------------------
    status("Loading Siamese detector ...")
    import librosa
    import numpy as np
    from detector import load_anchor, resolve_threshold
    from scoring import DEFAULT_TOP_K, SAMPLE_RATE, load_cohort, load_siamese_model

    model = load_siamese_model()
    anchor, window_samples, anchor_desc = load_anchor(keyword, None, model)
    cohort = load_cohort(keyword)
    threshold, threshold_desc = resolve_threshold(keyword, None)
    window_seconds = window_samples / SAMPLE_RATE
    status(f"Anchor: {anchor_desc} | window {window_seconds:.2f}s | "
           f"threshold {threshold:.3f} ({threshold_desc})")

    # -- 5. Live loop (endless: runs until the user clicks Finish) ----------
    downloader.setup_mode = False   # backlog policy: drop oldest, stay live
    emit("PHASE", "live")
    live_log = os.path.join(log_dir, f"live_{keyword}.txt")
    total_detected = 0
    emit("DETECT", "0")

    def process_chunk(video_path, audio_path):
        nonlocal total_detected
        name = os.path.basename(audio_path)
        if not os.path.exists(audio_path):
            return
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        scan = scan_chunk(y, model, anchor, cohort, window_samples,
                          threshold, SAMPLE_RATE, DEFAULT_TOP_K)
        del y
        detections = scan["detections"] if scan is not None else []

        if detections:
            total_detected += len(detections)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{stamp}_{name[:-4]}"
            kept_video = os.path.join(det_dir, base + ".mp4")
            kept_audio = os.path.join(det_dir, base + ".wav")
            shutil.move(audio_path, kept_audio)
            if os.path.exists(video_path):
                shutil.move(video_path, kept_video)
            with open(os.path.join(det_dir, base + ".json"), "w") as f:
                json.dump({"keyword": keyword, "chunk": name,
                           "detected_at": datetime.now().isoformat(timespec="seconds"),
                           "threshold": threshold,
                           "best_score": scan["best_score"],
                           "detections": detections}, f, indent=2)
            top = detections[0]
            msg = (f"MATCH '{keyword}' in {name} at {top['time']:.1f}s "
                   f"(AS-norm {top['score']:.2f}, threshold {threshold:.2f}) "
                   f"-> saved {base}.mp4")
            with open(live_log, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
            status(msg)
            emit("DETECT", str(total_detected))
        else:
            detail = "too short" if scan is None else (
                f"best AS-norm {scan['best_score']:.2f}")
            status(f"{name}: no match ({detail}) - chunk deleted")
            for p in (audio_path, video_path):
                if os.path.exists(p):
                    os.remove(p)

    # Disk backlog first: chunks from a previous run + the bootstrap set
    # (real stream data whose setup role is done - scanned like any chunk).
    disk_backlog = list(pre_existing)
    for idx in range(start_index, protect_below):
        disk_backlog.append(os.path.join(audio_dir, f"live_{idx}.wav"))
    for audio_path in disk_backlog:
        video_path = os.path.join(
            video_dir, os.path.basename(audio_path)[:-4] + ".mp4")
        process_chunk(video_path, audio_path)

    status("Monitoring live stream - scanning chunks as they arrive ...")
    while True:
        video_path, audio_path, _idx = chunk_queue.get()
        process_chunk(video_path, audio_path)
        emit("QUEUE", str(chunk_queue.qsize()))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        emit("PHASE", "error")
        emit("STATUS", f"Worker failed: {e}")
        raise
