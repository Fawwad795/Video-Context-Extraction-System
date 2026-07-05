# Live Monitoring Platform

GUI front-end for the Siamese KWS pipeline, modeled on the original VMS
project's `vms.py`. Downloading and detection start independently:

- **Start Download** — begins pulling stream chunks with just the URL; the
  keyword can still be blank or changed while chunks accumulate.
- **Start Detection** — sends the entered keyword to the running worker
  (or starts everything at once if download hasn't begun), which builds
  the keyword artifacts and starts scanning — including the chunks that
  arrived while the keyword was being decided.
- **Finish** — stops the session.

```
python platform/gui.py
```

## Isolation

Everything the platform produces lives under `platform/data/` — the research
folders (`audios/`, `keywords/`, `logs/`, `videos/`) are never touched. This
works through the `SIAMESE_PROJECT_ROOT` environment override in
`core/scoring.py`, which every pipeline script resolves its paths through.

```
platform/
  gui.py            the GUI (start here)
  live_worker.py    worker subprocess: setup + live monitoring loop
  data/             created at runtime - the platform's own project root
    keywords/       TTS variants, anchor, cohort, calibration -
                    built once per keyword, reused after
    audios/ videos/ transient chunk files (deleted after analysis)
    logs/           session logs + per-keyword detection log
    detections/<keyword>/   saved chunks: .mp4 + .wav + .json record
```

## Flow

**Downloading** starts the moment you press Start and never pauses:
a dedicated downloader thread polls the stream playlist and fetches every
new segment (dedup by URI path — YouTube re-signs query strings — plus
content md5), converts it to audio, and queues it for the detector. It runs
until you click **Finish**.

**Setup (begins when Start Detection is pressed; first run per keyword is
slow, and downloading continues throughout):** synthesize the multi-voice
TTS anchor → wait until ~10 chunks exist on disk → build the AS-norm
cohort → calibrate the detection threshold. Later runs with the same
keyword skip all of this.

**Live:** the detector consumes the queue: each chunk is scanned by the
Siamese AS-norm detector (50 ms hop — must match the calibration window
grid). For each chunk:

- **detection** → chunk video+audio moved to
  `data/detections/<keyword>/` with a JSON record of times/scores;
- **no detection** → chunk files deleted immediately,

so disk and memory stay flat during arbitrarily long runs (same
keep-or-delete policy as the original VMS platform).

The platform ends at detection. `transcribe_chunks.py` and
`validate_detection.py` are offline research tools for measuring accuracy
against ground truth — the live product never transcribes stream audio or
second-guesses a detection; it saves the chunk and moves on.

Because there are no transcripts, calibration cannot exclude keyword-bearing
bootstrap chunks by label — a keyword spoken during the ~10-chunk bootstrap
window can end up in its own "negative" sample, setting the threshold to
its own score and guaranteeing a miss (the "south" bug). No per-window
statistic can safely detect and remove this after the fact (three
approaches tried and rejected — see `reports/EXPERIMENT_LOG.md`,
"Calibration leakage without transcripts"; a rigorous extreme-value fit
even agrees a leaked score is statistically unremarkable against a
legitimate hard negative). Instead, **periodic recalibration**: every scan
already computes AS-norm scores for every window, so `live_worker.py`
feeds those scores — from chunks that did NOT trigger a detection — into a
growing, capped pool (`--recalib-pool-cap`, default 100k), and every
`--recalib-interval-seconds` (default 300s) refits the threshold by
excluding a small fixed number of the pool's most extreme points
(`--recalib-tolerance`, default 5) before taking the max — deliberately
**not** the bootstrap's literal p100, which can only rise as the pool
grows and would keep any one-off leak as its ceiling forever. The cost is
honest: up to that many genuinely extreme hard negatives per pool are now
tolerated, in exchange for actually recovering from a leaked keyword.
