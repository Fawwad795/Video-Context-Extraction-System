"""Live monitoring worker for the Siamese KWS platform.

Launched by platform/gui.py as a subprocess. One live-stream URL per run;
the keyword can be given up-front (--keyword) or sent later as a
'KEYWORD <word>' line on stdin - downloading starts immediately either
way, so chunks accumulate while the user is still deciding what to search
for. Everything it reads/writes lives under the platform data root
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

Periodic recalibration: the bootstrap calibration is a small, one-shot
sample (~10 chunks, ~50s of audio) with no transcripts to exclude a chunk
that happens to contain the keyword - an unlucky draw sets the threshold
to that utterance's own score, guaranteeing a miss for it (and similar
future ones). No per-window statistical test can safely detect this after
the fact (tried and rejected - see core/scoring.evt_threshold's
docstring): a genuine hard negative and a leaked keyword utterance can be
statistically indistinguishable in a small sample.

The fix is NOT "grow the sample and take the max again" - a literal max
can only stay flat or rise as more data arrives, so one leaked point stays
its ceiling forever no matter how much clean data piles up afterward
(verified numerically before shipping this). Every scan already computes
AS-norm scores for every window of every chunk, so LiveScanner feeds
those scores (from chunks that did NOT trigger a detection - never from a
chunk currently believed to contain the keyword) into a growing, capped
pool, and the threshold is refit by excluding a small fixed number of the
pool's most extreme points (--recalib-tolerance) before taking the max -
first as soon as the pool reaches --recalib-min-pool, then every
--recalib-interval-seconds. A one-off leak occupies one of the excluded
slots once the pool has grown past it, recovering the clean ceiling - at
the honest cost of also tolerating that many genuinely extreme hard
negatives per pool, rather than the bootstrap sample's strict (but
small-sample-fragile) zero-false-alarm guarantee.

Held chunks close the remaining sequencing hole: the chunks most at risk
of a wrong no-match verdict are the bootstrap-era ones scanned right after
setup - exactly the audio whose keyword utterance may have contaminated
the calibration - and deleting them on first judgment made the error
unrecoverable (observed twice: 'south', then 'brain', each missing its own
calibration-contaminating utterance by one epsilon). Until the first
recalibration, no-match chunks are therefore held on disk with their
window scores kept in memory, then re-judged against the corrected
threshold (no rescan needed); late detections are saved with a
'late_after_recalibration' marker, everything else is deleted then.

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
from collections import deque
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

    Every kept chunk is enqueued. Backlog cap policy when the consumer is
    not keeping up (or has not started yet):
      setup_mode=True  - skip new segments once the queue is full, so files
                         that setup scripts are reading never get deleted;
      setup_mode=False - drop the OLDEST unprocessed chunk (delete its
                         files), keeping the monitor close to live.
    """

    def __init__(self, url, video_dir, audio_dir, start_index,
                 chunk_queue, backlog_cap, max_remember=8192):
        super().__init__(daemon=True)
        self.url = url
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.index = start_index
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
    from scoring import asnorm_windows, embed_batch, l2_normalize

    # Frame backends (the production wavlm-trained path) MUST score windows
    # pooled from one chunk-level forward pass: calibrate.py fits the
    # threshold through that exact protocol, and isolated-window embeddings
    # land in a measurably different score region (experiment log,
    # "scoring-protocol mismatch"). It is also ~100x faster per chunk.
    frame_mode = getattr(model, "is_frame_backend", False)
    if frame_mode:
        from embedders import FRAME_STRIDE, pooled_windows, samples_to_frames
        hop_frames = max(1, int(round(STEP_SECONDS * sample_rate / FRAME_STRIDE)))
        chunk_frames = model.frames(y.astype(np.float32))

    step_samples = max(1, int(sample_rate * STEP_SECONDS))
    all_scores, all_raw, all_times, all_scales = [], [], [], []
    for scale in SCALES:
        ws = max(int(window_samples * scale), int(0.15 * sample_rate))
        if frame_mode:
            wf = samples_to_frames(ws)
            if hasattr(model, "pool_windows"):
                embs, start_frames = model.pool_windows(
                    chunk_frames, wf, hop_frames)
            else:
                embs, start_frames = pooled_windows(
                    chunk_frames, wf, hop_frames)
            if len(embs) == 0:
                continue
            embs = l2_normalize(embs)
            starts = start_frames * FRAME_STRIDE
        else:
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

    best = int(np.argmax(normed))
    return {"detections": detections_at(normed, raw, times, win_scales,
                                        threshold),
            "best_score": float(normed[best]), "best_time": float(times[best]),
            "best_scale": float(win_scales[best]),
            # Full per-window arrays: LiveScanner pools the scores for
            # periodic recalibration and re-judges held chunks against a
            # recalibrated threshold without rescanning the audio.
            "all_scores": normed, "all_raw": raw,
            "all_times": times, "all_scales": win_scales}


def detections_at(normed, raw, times, win_scales, threshold):
    """Detection records for the windows at/above threshold."""
    import numpy as np
    hit_idx = np.where(normed >= threshold)[0]
    detections = [{"time": float(times[i]), "score": float(normed[i]),
                   "raw_cos": float(raw[i]), "scale": float(win_scales[i])}
                  for i in hit_idx]
    detections.sort(key=lambda d: -d["score"])
    return detections


# --------------------------------------------------------------------------
# Live loop state: keep/delete decisions, recalibration, held chunks
# --------------------------------------------------------------------------

class LiveScanner:
    """Owns the live monitoring loop's mutable state: the working
    threshold, the recalibration score pool, and the holding area for
    chunks judged before the threshold has been corrected once.

    Periodic recalibration (why not the bootstrap's literal p100): the
    bootstrap calibration is a small transcript-less sample - a keyword
    utterance inside it sets the p100 threshold to its own score,
    guaranteeing a miss. A literal max can only stay flat or rise as more
    data arrives, so growing the pool alone never recovers; instead the
    threshold is refit excluding the pool's `tolerance` most extreme
    points, which evicts a one-off leak almost immediately at the honest
    cost of tolerating that many genuinely extreme hard negatives
    (verified numerically + end-to-end; experiment log, "Calibration
    leakage without transcripts").

    Held chunks (why deletion is deferred at session start): the chunks
    most likely to be mis-judged are the bootstrap-era ones scanned
    seconds after setup - exactly the audio whose keyword utterance may
    have contaminated the calibration (observed twice: 'south' 4.3851 vs
    threshold 4.3852, 'brain' 5.74 vs 5.743 - both epsilon-misses of the
    utterance that set the threshold). Deleting them on first judgment
    makes the error unrecoverable. So until the first recalibration, no-
    match chunks are HELD on disk with their per-window scores kept in
    memory; when the first recalibration fires (as soon as the pool
    reaches min_pool - the interval only throttles refits after that),
    every held chunk is re-judged against the corrected threshold with no
    rescan needed. hold_cap bounds the held set: if the pool somehow
    never fills, held chunks are adjudicated under the current threshold
    rather than accumulating forever.
    """

    def __init__(self, keyword, det_dir, live_log, threshold,
                 interval_seconds, min_pool, pool_cap, tolerance, hold_cap,
                 scan_fn, on_detect=None):
        import numpy as np
        self._np = np
        self.keyword = keyword
        self.det_dir = det_dir
        self.live_log = live_log
        self.threshold = threshold
        self.interval_seconds = interval_seconds
        self.min_pool = min_pool
        self.tolerance = tolerance
        self.hold_cap = hold_cap
        self.scan_fn = scan_fn                  # audio_path -> scan dict|None
        self.on_detect = on_detect or (lambda total: None)
        self.pool = deque(maxlen=pool_cap)
        self.held = []                          # [(video, audio, name, scan)]
        self.recalibrated_once = False
        self.last_recalib = time.monotonic()
        self.total_detected = 0

    # -- outcomes ----------------------------------------------------------
    def _log(self, msg):
        with open(self.live_log, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
        status(msg)

    def _keep(self, video_path, audio_path, name, scan, detections,
              late=False):
        self.total_detected += len(detections)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{stamp}_{name[:-4]}"
        if os.path.exists(audio_path):
            shutil.move(audio_path, os.path.join(self.det_dir, base + ".wav"))
        if os.path.exists(video_path):
            shutil.move(video_path, os.path.join(self.det_dir, base + ".mp4"))
        with open(os.path.join(self.det_dir, base + ".json"), "w") as f:
            json.dump({"keyword": self.keyword, "chunk": name,
                       "detected_at": datetime.now().isoformat(timespec="seconds"),
                       "threshold": self.threshold,
                       "late_after_recalibration": late,
                       "best_score": scan["best_score"],
                       "detections": detections}, f, indent=2)
        top = detections[0]
        self._log(f"MATCH{' (late, after recalibration)' if late else ''} "
                  f"'{self.keyword}' in {name} at {top['time']:.1f}s "
                  f"(AS-norm {top['score']:.2f}, threshold "
                  f"{self.threshold:.2f}) -> saved {base}.mp4")
        self.on_detect(self.total_detected)

    @staticmethod
    def _drop(video_path, audio_path):
        for p in (audio_path, video_path):
            if os.path.exists(p):
                os.remove(p)

    # -- recalibration -----------------------------------------------------
    def _maybe_recalibrate(self):
        if self.interval_seconds <= 0:
            return
        if len(self.pool) < self.min_pool:
            return
        # The interval throttles REFITS; the first correction should land
        # as soon as there is enough data, not a full interval later -
        # held chunks are waiting on it.
        if self.recalibrated_once and (
                time.monotonic() - self.last_recalib < self.interval_seconds):
            return
        self.last_recalib = time.monotonic()
        np = self._np
        pool = np.fromiter(self.pool, dtype=np.float64)
        pct = 100.0 * (1.0 - self.tolerance / len(pool))
        old = self.threshold
        self.threshold = float(np.percentile(pool, pct)) + 1e-4
        self._log(f"Recalibrated: threshold {old:.3f} -> "
                  f"{self.threshold:.3f} (top {self.tolerance} of "
                  f"{len(pool)} live-observed scores excluded)")
        if not self.recalibrated_once:
            self.recalibrated_once = True
            self._adjudicate_held()

    def _adjudicate_held(self):
        held, self.held = self.held, []
        n_late = 0
        for video_path, audio_path, name, scan in held:
            detections = detections_at(
                scan["all_scores"], scan["all_raw"], scan["all_times"],
                scan["all_scales"], self.threshold)
            if detections:
                n_late += 1
                self._keep(video_path, audio_path, name, scan, detections,
                           late=True)
            else:
                self._drop(video_path, audio_path)
        if held:
            self._log(f"Re-judged {len(held)} held chunk(s) under the "
                      f"recalibrated threshold: {n_late} late detection(s), "
                      f"{len(held) - n_late} deleted")

    # -- per-chunk entry point ----------------------------------------------
    def process_chunk(self, video_path, audio_path):
        name = os.path.basename(audio_path)
        if not os.path.exists(audio_path):
            return
        scan = self.scan_fn(audio_path)
        if scan is None:
            status(f"{name}: too short - chunk deleted")
            self._drop(video_path, audio_path)
            return

        detections = detections_at(
            scan["all_scores"], scan["all_raw"], scan["all_times"],
            scan["all_scales"], self.threshold)
        if detections:
            self._keep(video_path, audio_path, name, scan, detections)
        else:
            # Never pool scores from a chunk currently believed to contain
            # the keyword - don't reinforce the threshold that decided it.
            self.pool.extend(scan["all_scores"].tolist())
            if self.recalibrated_once:
                status(f"{name}: no match (best AS-norm "
                       f"{scan['best_score']:.2f}) - chunk deleted")
                self._drop(video_path, audio_path)
            else:
                self.held.append((video_path, audio_path, name, scan))
                status(f"{name}: no match (best AS-norm "
                       f"{scan['best_score']:.2f}) - held until first "
                       f"recalibration ({len(self.held)} held)")
                if len(self.held) > self.hold_cap:
                    status(f"Hold cap ({self.hold_cap}) reached before the "
                           f"first recalibration - adjudicating held chunks "
                           f"under the current threshold")
                    self.recalibrated_once = True   # stop holding new ones
                    self._adjudicate_held()
        self._maybe_recalibrate()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Siamese KWS live platform worker.")
    ap.add_argument("--keyword", default=None,
                    help="keyword to monitor; omit to start downloading "
                         "immediately and receive the keyword later as a "
                         "'KEYWORD <word>' line on stdin")
    ap.add_argument("--url", required=True, help="live stream URL")
    ap.add_argument("--root", default=DEFAULT_DATA_ROOT,
                    help="platform data root (isolated from the research folders)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--bootstrap-chunks", type=int, default=10,
                    help="chunks needed up-front for cohort/calibration")
    ap.add_argument("--backlog-cap", type=int, default=300,
                    help="max unprocessed chunks kept on disk before dropping")
    ap.add_argument("--recalib-interval-seconds", type=int, default=300,
                    help="how often to refit the threshold from live-"
                         "observed scores (see 'Periodic recalibration' "
                         "below); 0 disables it")
    ap.add_argument("--recalib-min-pool", type=int, default=2000,
                    help="minimum accumulated window scores before the "
                         "first recalibration fires")
    ap.add_argument("--recalib-pool-cap", type=int, default=100_000,
                    help="max window scores kept in the recalibration "
                         "pool (oldest dropped first - bounds memory)")
    ap.add_argument("--recalib-tolerance", type=int, default=5,
                    help="number of the pool's most extreme scores "
                         "excluded before taking the max (NOT the "
                         "bootstrap's literal p100 - see LiveScanner "
                         "for why growing the pool alone can't fix a "
                         "literal max; this tolerates up to K extreme "
                         "hard negatives in exchange for actually "
                         "recovering from a leaked keyword)")
    ap.add_argument("--hold-cap", type=int, default=60,
                    help="max no-match chunks held on disk awaiting the "
                         "first recalibration before being adjudicated "
                         "under the current threshold anyway")
    args = ap.parse_args()

    keyword = args.keyword.strip().lower() if args.keyword else None
    data_root = os.path.abspath(args.root)

    # All pipeline scripts + core modules resolve their paths through this.
    os.environ["SIAMESE_PROJECT_ROOT"] = data_root
    # Both weight variables must be pinned: SIAMESE_WEIGHTS feeds the
    # baseline backend, SIAMESE_V3_WEIGHTS the wavlm-trained attentive head.
    # Their defaults resolve under PROJECT_ROOT, which the line above just
    # redirected into the (checkpoint-less) platform data root - without
    # this the trained head silently degrades to the identity-init head.
    os.environ["SIAMESE_WEIGHTS"] = args.weights
    os.environ["SIAMESE_V3_WEIGHTS"] = args.weights
    os.environ.setdefault("SIAMESE_BACKEND", "wavlm-trained")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    audio_dir = os.path.join(data_root, "audios")
    video_dir = os.path.join(data_root, "videos")
    kw_dir = os.path.join(data_root, "keywords")
    log_dir = os.path.join(data_root, "logs")
    for d in (audio_dir, video_dir, kw_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    sys.path.insert(0, CORE_DIR)
    sys.path.insert(0, PIPELINE_DIR)

    # Chunks left on disk by a previous run (killed mid-processing): they are
    # scanned before the live queue, and count toward the bootstrap set.
    pre_existing = sorted(
        (os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
         if f.startswith("live_") and f.endswith(".wav")),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))

    start_index = next_live_index(audio_dir, video_dir)

    # The downloader thread lazily imports moviepy, which imports numpy.
    # Importing a C-extension module inside a secondary thread deadlocks on
    # Windows while the main thread is blocked on the stdin keyword wait
    # (observed: numpy stuck in create_module in a faulthandler dump, first
    # chunk downloaded but never converted). Pre-import in the MAIN thread
    # before the downloader starts.
    status("Loading audio/video libraries ...")
    from moviepy.video.io.VideoFileClip import VideoFileClip  # noqa: F401

    # -- Downloading starts NOW, keyword or not, and never stops until the
    # user hits Finish. Every new chunk is enqueued; queued chunks stay on
    # disk until the live loop consumes them (setup_mode never deletes), so
    # chunks downloaded while the keyword is still undecided are neither
    # lost nor exempt from scanning.
    chunk_queue = queue.Queue()
    downloader = StreamDownloader(args.url, video_dir, audio_dir, start_index,
                                  chunk_queue, args.backlog_cap)
    downloader.start()

    # -- 0. Wait for the keyword if it was not given up-front ---------------
    if keyword is None:
        emit("PHASE", "download")
        status("Downloading chunks - enter a keyword and press "
               "Start Detection when ready.")
        while True:
            line = sys.stdin.readline()
            if not line:  # GUI went away without ever picking a keyword
                status("stdin closed with no keyword - stopping.")
                return
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[0].upper() == "KEYWORD":
                candidate = parts[1].strip().lower()
                if candidate and " " not in candidate:
                    keyword = candidate
                    break
                status(f"Invalid keyword {parts[1]!r} - send a single word.")

    det_dir = os.path.join(data_root, "detections", keyword)
    os.makedirs(det_dir, exist_ok=True)
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
    emit("DETECT", "0")

    def scan_audio(audio_path):
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        return scan_chunk(y, model, anchor, cohort, window_samples,
                          scanner.threshold, SAMPLE_RATE, DEFAULT_TOP_K)

    scanner = LiveScanner(
        keyword=keyword, det_dir=det_dir, live_log=live_log,
        threshold=threshold,
        interval_seconds=args.recalib_interval_seconds,
        min_pool=args.recalib_min_pool, pool_cap=args.recalib_pool_cap,
        tolerance=args.recalib_tolerance, hold_cap=args.hold_cap,
        scan_fn=scan_audio,
        on_detect=lambda total: emit("DETECT", str(total)))

    # Disk backlog first: chunks left over from a previous run. Everything
    # downloaded during THIS run (bootstrap included) is already in the
    # queue and gets scanned by the live loop below.
    for audio_path in pre_existing:
        video_path = os.path.join(
            video_dir, os.path.basename(audio_path)[:-4] + ".mp4")
        scanner.process_chunk(video_path, audio_path)

    status("Monitoring live stream - scanning chunks as they arrive ...")
    while True:
        video_path, audio_path, _idx = chunk_queue.get()
        scanner.process_chunk(video_path, audio_path)
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
