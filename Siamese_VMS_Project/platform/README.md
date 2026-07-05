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
its own score and guaranteeing a miss (observed three times: "south",
"brain", "morning" — each missed the very utterance that set its
threshold by one epsilon). No per-window statistic can safely detect and
remove this after the fact (three approaches tried and rejected — see
`reports/EXPERIMENT_LOG.md`, "Calibration leakage without transcripts"; a
rigorous extreme-value fit even agrees a leaked score is statistically
unremarkable against a legitimate hard negative). A fixed-tolerance
periodic recalibration was tried next and is *also* unsound in a
different way: excluding a small fixed count of a growing pool's top
scores forces the threshold below scores already known to be negatives
once the pool is still small, causing false positives by construction
(observed live: "morning", threshold 5.463 → 2.065 on a 6-chunk pool, ~7
clean chunks then fired as false positives).

**What actually works — held-chunk adjudication, exactly once per
session:** no-match chunks are **held on disk** (not deleted) until
`--hold-min-chunks` (default 30) have been seen — these earliest chunks
are exactly the audio the bootstrap calibration may have leaked from. Then
a single adjudication runs: the held chunk with the highest peak may fire
*only if* it (a) beats every window of every other held chunk
(leave-one-out p100) *and* (b) sits at the current threshold's epsilon
(i.e. it is the very window that set the ceiling). At most one chunk per
session can ever satisfy both — so a contaminated ceiling releases (that
one chunk fires as a late, clearly-marked review candidate, and the
threshold drops to the next-highest observed ceiling), while a genuinely
clean session still surfaces exactly one bounded review candidate (its own
hardest negative is mathematically indistinguishable from a leak) — never
a sweep. After this one-time pass the detector runs exactly like the
offline research protocol: fixed threshold, no further adjustment. Late
detections are saved with a `late_after_adjudication` marker in their JSON
record so they're never confused with an ordinary live match.
