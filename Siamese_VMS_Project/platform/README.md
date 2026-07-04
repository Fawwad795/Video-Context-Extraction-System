# Live Monitoring Platform

GUI front-end for the Siamese KWS pipeline, modeled on the original VMS
project's `vms.py`. Enter a keyword and a live-stream (YouTube) URL, press
**Start**, and the platform monitors the stream for the keyword.

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

**Setup (first run per keyword, slow, runs while downloading continues):**
synthesize the multi-voice TTS anchor → wait for ~10 bootstrap chunks →
build the AS-norm cohort → calibrate the detection threshold. Later runs
with the same keyword skip all of this.

**Live:** the detector consumes the queue: each chunk is scanned by the
Siamese AS-norm detector (50 ms hop — must match the calibration window
grid). For each chunk:

- **detection** → chunk video+audio moved to
  `data/detections/<keyword>/` with a JSON record of times/scores;
- **no detection** → chunk files deleted immediately,

so disk and memory stay flat during arbitrarily long runs (same
keep-or-delete policy as the original VMS platform).
