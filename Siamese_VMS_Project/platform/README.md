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
- **View Detections** — opens the saved-chunks folder for the current keyword.
- **Clear Console** — clears the on-screen log only; the full session
  transcript still accumulates on disk at `data/logs/session_*.log`.

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
    audios_copy/    standing copy of every downloaded chunk - NOT cleaned
                    up, grows for the life of the session; for manually
                    spot-checking results later, untouched by any
                    pipeline script (see "Flow" below)
    logs/           session logs + per-keyword detection log
    detections/<keyword>/   saved chunks: .mp4 + .wav + .json record
```

## Flow

**Downloading** starts the moment you press Start and never pauses:
a dedicated downloader thread polls the stream playlist and fetches every
new segment (dedup by URI path — YouTube re-signs query strings — plus
content md5), converts it to audio, and queues it for the detector. It runs
until you click **Finish**. Every converted chunk is also copied to
`data/audios_copy/` at this point, before the detector has any chance to
delete or move the original — a standing archive you can browse by hand,
independent of the keep-or-delete policy below. Unlike the rest of the
platform's data, this folder is not cleaned up automatically.

**Setup (begins when Start Detection is pressed; first run per keyword is
slow, and downloading continues throughout):** synthesize the multi-voice
TTS anchor → wait until ~10 chunks exist on disk → transcribe those chunks
(whisper-base, one-time) → build the AS-norm cohort → calibrate the
detection threshold, using the transcript to exclude any keyword-bearing
chunk from its negative sample. Later runs with the same keyword (or a new
keyword against the same downloaded audio) skip whatever's already built.

**Live:** the detector consumes the queue: each chunk is scanned by the
Siamese AS-norm detector (50 ms hop — must match the calibration window
grid). For each chunk:

- **detection** → chunk video+audio moved to
  `data/detections/<keyword>/` with a JSON record of times/scores;
- **no detection** → chunk files deleted immediately,

so disk and memory stay flat during arbitrarily long runs (same
keep-or-delete policy as the original VMS platform).

The platform ends at detection — it never second-guesses a live match; it
saves the chunk and moves on. `validate_detection.py` remains an offline
research tool for measuring accuracy against ground truth, unused by the
platform. `transcribe_chunks.py` is the one exception on the live side: it
runs once, on the ~10 bootstrap chunks only, purely so calibration has
ground truth to exclude the keyword's own utterance from its negative
sample (see "Calibration leakage guard" below) — the live detection loop
itself never transcribes anything.

### Calibration leakage guard

Without ground truth, calibrate.py's small bootstrap sample can include a
chunk that happens to contain the keyword itself, setting the threshold to
that utterance's own score and guaranteeing a miss for it — and if the
keyword recurs often in the stream, no purely statistical fix after the
fact is reliable (several were tried: score-range excision, temporal-burst
excision, a Generalized Pareto tail fit, fixed-tolerance periodic
recalibration, held-chunk leave-one-out adjudication — each had a
different failure mode, most visibly a live 'mexico' session where the
keyword recurred often enough that the recovery mechanism could only ever
rescue one occurrence out of many; see `reports/EXPERIMENT_LOG.md`,
"Calibration leakage without transcripts", and commit `181ea7f` where that
whole line of attempts was reverted).

The fix restores what the offline research pipeline always relied on:
`transcribe_chunks.py --limit <bootstrap-chunks>` transcribes the
bootstrap window once (whisper-base — more accurate than the tiny model
used elsewhere, since a missed word here silently reintroduces the exact
leak this exists to prevent), and `calibrate.py`'s existing
`keyword_free_chunks()` guard excludes every chunk whose transcript
contains the keyword from the negative sample — correctly, regardless of
how many times the keyword occurs, unlike any of the reverted statistical
approaches. Validated end-to-end (real audio + model, not synthetic
scores): simulating a fresh platform bootstrap on a chunk set where a
keyword occurs 3 times recovers all 3 with zero false positives, matching
the offline pipeline's historical result exactly.
