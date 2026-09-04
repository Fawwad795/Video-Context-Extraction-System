"""Shared Modal runtime for every published-benchmark harness.

One image, one volume pair, and one copy of the per-keyword enrollment unit,
composed into each benchmark app with `App.include`. Split out of
`libriphrase/modal_app.py` on 2026-09-04 so that the LibriPhrase, Speech
Commands and broadcast harnesses each live in their own folder without three
copies of the environment drifting apart.

Every function body below is a verbatim slice of the original file.

**STAGE_ARGS is the LibriPhrase voice budget and it is load-bearing.** `enroll`
applies it to every keyword it builds, so LibriPhrase and Speech Commands both
enrol with 12 voices. The broadcast rebuild must NOT use this path: its keywords
were originally built with the 7-voice default, and `broadcast/modal_rebuild.py`
therefore runs the pipeline stages itself with no extra arguments.
"""

import json
import os

import modal

runtime = modal.App("vms-runtime")
data = modal.Volume.from_name("vms-data", create_if_missing=True)
hfcache = modal.Volume.from_name("vms-hfcache", create_if_missing=True)

# Repo root on the machine running `modal run`.
REPO = os.environ.get(
    "VMS_REPO", r"d:/Video Context Extraction System/Siamese_VMS_Project")

RUN_ROOT = "/data/run"                  # SIAMESE_PROJECT_ROOT -> artifacts land here
COHORTS = {"devclean": "/data/cohort_src/devclean",
           "broadcast": "/data/cohort_src/broadcast"}

# DECLARED CONSTANT - the TTS voice budget, fixed for every keyword.
# keyword_generator aborts when `len(clips) <= holdout + 2`, i.e. <= 6. With the
# 7 canonical voices alone, a SINGLE degenerate synthesis in any one voice kills
# the keyword - which silently cost 38 of 809 selection keywords (4.7%). Adding
# 5 random-x-vector voices gives 12, so up to 5 may fail before the abort.
# Changing this changes every anchor, so it must be fixed BEFORE any run whose
# artifacts are kept, and stated as 12 voices in the paper.
STAGE_ARGS = {"keyword_generator": ["--n-random", "5"]}


def _prefetch():
    """Bake model weights into the image so HF_HUB_OFFLINE works at run time.

    core/scoring.py forces HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE at import. Without
    pre-baking, every container dies with a confusing network error instead of a
    missing-cache error. This runs at BUILD time, where the network is available
    and the offline flags are not yet set.
    """
    import os

    WAVLM = "microsoft/wavlm-base-plus"

    def load_everything_the_runtime_loads():
        """Mirror every from_pretrained/load_dataset in core/ and pipeline/.

        Enumerated from the source rather than guessed - an earlier version
        fetched AutoModel but not AutoFeatureExtractor, which AutoModel does not
        pull in, so every container died on
        core/embedders.py:54. Both the plain and use_safetensors=True variants
        are fetched because the pipeline tries the latter first and falls back.
        """
        from datasets import load_dataset
        from transformers import (AutoFeatureExtractor, AutoModel,
                                  SpeechT5ForTextToSpeech, SpeechT5HifiGan,
                                  SpeechT5Processor, WhisperTokenizer)

        def both(fn, *a):
            fn(*a)
            try:
                fn(*a, use_safetensors=True)
            except Exception:
                pass                      # model ships no safetensors: fine

        # core/embedders.py FrameBackend
        AutoFeatureExtractor.from_pretrained(WAVLM)
        both(AutoModel.from_pretrained, WAVLM)
        # pipeline/keyword_generator.py load_tts()
        SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        both(SpeechT5ForTextToSpeech.from_pretrained, "microsoft/speecht5_tts")
        both(SpeechT5HifiGan.from_pretrained, "microsoft/speecht5_hifigan")
        load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        # pipeline/rival_bank.py - only used when rebuilding the bank, but tiny
        WhisperTokenizer.from_pretrained("openai/whisper-base")

    load_everything_the_runtime_loads()

    import nltk
    for pkg in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger",
                "cmudict"):
        try:
            nltk.download(pkg, download_dir="/opt/nltk")
        except Exception as exc:          # non-fatal: g2p_en falls back
            print("nltk", pkg, exc)

    # Prove the bake works, at BUILD time, by re-running the SAME closure with
    # the offline flags on - exactly as a container will. If anything did not
    # land in the image layer, fail the build rather than all 9,030 enrollments.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1")
    load_everything_the_runtime_loads()
    print("offline reload OK - model cache is baked into the image layer")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "curl")
    # Pinned to requirements.txt so Modal matches the environment the local
    # artifacts were built in. datasets is the load-bearing pin: 3.6.0 still
    # supports script-based datasets, and `Matthijs/cmu-arctic-xvectors` (which
    # keyword_generator.load_tts() loads at runtime, not just at prefetch) is
    # script-based. datasets 4.x removes script support and fails with
    # "Dataset scripts are no longer supported".
    # CPU-only torch: enrollment runs on CPU containers, so the default Linux
    # wheel's bundled CUDA stack (nvidia-cublas/cudnn/triton, several GB) is
    # dead weight that every cold container has to pull.
    .pip_install("torch==2.10.0",
                 index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "transformers==5.10.1",
        "datasets==3.6.0",
        "librosa==0.11.0",
        "soundfile==0.13.1",
        "numpy==1.26.4",
        "scipy==1.17.1",
        "num2words==0.5.14",
        "g2p_en==2.1.0",
        "nltk==3.9.4",
        "scikit-learn==1.8.0",
        "pandas",
        "inflect",
        "sentencepiece",
    )
    # HF_HOME must be set BEFORE the build step so _prefetch downloads into a
    # known path, and that path must live in the IMAGE LAYER - not a Volume.
    # Mounting a volume here would put the cache somewhere `enroll` (which mounts
    # only /data) cannot see, and every container would then fail offline with
    # "Can't load feature extractor for 'microsoft/speecht5_tts'".
    .env({"HF_HOME": "/opt/hf", "NLTK_DATA": "/opt/nltk"})
    .run_function(_prefetch)
    .env({
        "SIAMESE_BACKEND": "wavlm-trained",
        "SIAMESE_V3_WEIGHTS": "/app/checkpoints/siamese_v3_best.pth",
        # ARCHITECTURE section 17: the rival bank resolves against the CODE repo,
        # not SIAMESE_PROJECT_ROOT, so it must be pointed at explicitly.
        "SIAMESE_RIVAL_BANK": "/app/keywords/rival_bank_wavlm10ft.npz",
        "SIAMESE_PROJECT_ROOT": RUN_ROOT,
        "SIAMESE_VERIFIER": "rival",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    })
    # Local assets last - they do not invalidate the cached build layers above.
    .add_local_python_source("common")
    .add_local_dir(f"{REPO}/core", remote_path="/app/core")
    .add_local_dir(f"{REPO}/pipeline", remote_path="/app/pipeline")
    .add_local_file(f"{REPO}/checkpoints/siamese_v3_best.pth",
                    remote_path="/app/checkpoints/siamese_v3_best.pth")
    .add_local_file(f"{REPO}/keywords/rival_bank_wavlm10ft.npz",
                    remote_path="/app/keywords/rival_bank_wavlm10ft.npz")
)


_COMMIT_EVERY = 20
_since_commit = {"n": 0}


def _maybe_commit(force: bool = False):
    _since_commit["n"] += 1
    if force or _since_commit["n"] >= _COMMIT_EVERY:
        _since_commit["n"] = 0
        data.commit()


# NOTE: an in-process, memoized-loader variant of enroll() was built and tested
# (verify_memo, 2026-08-28) to avoid reloading SpeechT5/WavLM per subprocess -
# ~49% of per-keyword time. It was REMOVED. core/scoring.py resolves
# PROJECT_ROOT/AUDIO_DIR at import time and the pipeline modules import several
# of those names by value, so a container that had already imported `scoring`
# kept the first job's root: the in-process arm silently wrote its artifacts
# into the other arm's directory and reported success. Cost saving was ~3x; the


@runtime.function(
    image=image, volumes={"/data": data},
    cpu=2.0, memory=4096, timeout=1800, retries=2,
    # Modal >=1.0 uses max_containers; older versions call it concurrency_limit.
    max_containers=100,
)
def enroll(job: dict):
    """Build one keyword's artifacts. Idempotent: existing stages are skipped.

    job = {keyword, cohort: 'devclean'|'broadcast', need_rivals: bool,
           root: optional override of SIAMESE_PROJECT_ROOT}
    """
    import subprocess
    import sys
    import time

    kw = str(job["keyword"]).lower().strip()
    root = job.get("root", RUN_ROOT)
    audio_dir = COHORTS[job["cohort"]]

    # Set the env BEFORE anything imports scoring: it resolves PROJECT_ROOT and
    # AUDIO_DIR at import time, so importing it first would freeze stale values
    # for the life of the container.
    os.environ["SIAMESE_PROJECT_ROOT"] = root
    os.environ["SIAMESE_AUDIO_DIR"] = audio_dir
    env = dict(os.environ)
    os.makedirs(f"{root}/keywords", exist_ok=True)

    # Build artifact paths from `root` explicitly rather than via
    # scoring.anchor_path(): scoring caches PROJECT_ROOT at import, so a
    # container reused across roots would check the wrong paths.
    # artifact_suffix() depends only on backend/backbone/layer, so it is safe.
    sys.path.insert(0, "/app/core")
    from scoring import artifact_suffix                     # noqa: E402
    suf = artifact_suffix()
    kdir = f"{root}/keywords"

    stages = [("keyword_generator", f"{kdir}/{kw}_anchor{suf}.npz"),
              ("cohort_builder", f"{kdir}/cohort_{kw}{suf}.npz")]
    if job.get("need_rivals"):
        stages.append(("rival_builder", f"{kdir}/{kw}_rivals{suf}.npz"))

    timings, skipped = {}, []
    for name, artifact in stages:
        if os.path.exists(artifact):
            skipped.append(name)
            continue
        t = time.perf_counter()
        p = subprocess.run(
            [sys.executable, f"/app/pipeline/{name}.py", "--keyword", kw]
            + STAGE_ARGS.get(name, []),
            cwd="/app", env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        timings[name] = round(time.perf_counter() - t, 1)
        if p.returncode != 0:
            _maybe_commit(force=True)
            return {"keyword": kw, "cohort": job["cohort"], "ok": False,
                    "stage": name, "timings": timings,
                    "err": (p.stderr or p.stdout)[-1500:]}
        # Do NOT trust the exit code alone. keyword_generator.main() bails out
        # with `ui.fail(...); return` - no sys.exit(1) - when too few voices
        # survive the degenerate-synthesis filter, so it exits 0 having written
        # nothing. cohort_builder then finds no anchor, silently falls back to a
        # default 0.70 s window (cohort_builder.py:74-79), and also "succeeds" -
        # leaving a keyword with a cohort built against the wrong window and no
        # anchor at all. Verify the artifact instead.
        if not os.path.exists(artifact):
            _maybe_commit(force=True)
            return {"keyword": kw, "cohort": job["cohort"], "ok": False,
                    "stage": name, "timings": timings,
                    "err": f"stage exited 0 but produced no artifact: {artifact}\n"
                           f"--- stdout tail ---\n{(p.stdout or '')[-900:]}"}
    _maybe_commit()
    return {"keyword": kw, "cohort": job["cohort"], "ok": True,
            "timings": timings, "skipped": skipped}


@runtime.function(image=image, volumes={"/data": data}, timeout=1800)
def purge(keywords: list, root: str = RUN_ROOT):
    """Delete artifacts for the given keywords so they will be rebuilt.

    Needed whenever an input the artifacts depend on changes (e.g. the cohort
    source), because `enroll` is idempotent and would otherwise keep the stale
    files forever.
    """
    import glob
    import shutil
    import sys

    sys.path.insert(0, "/app/core")
    from scoring import artifact_suffix
    suf = artifact_suffix()

    removed = 0
    for kw in keywords:
        kw = str(kw).lower().strip()
        for p in (f"{root}/keywords/{kw}_anchor{suf}.npz",
                  f"{root}/keywords/cohort_{kw}{suf}.npz",
                  f"{root}/keywords/{kw}_rivals{suf}.npz",
                  f"{root}/keywords/{kw}_calibration{suf}.json"):
            if os.path.exists(p):
                os.remove(p)
                removed += 1
        for d in glob.glob(f"{root}/keywords/{kw}_variants") + \
                glob.glob(f"{root}/keywords/{kw}_rivals"):
            if os.path.isdir(d):
                shutil.rmtree(d)
                removed += 1
    data.commit()
    return {"keywords": len(keywords), "paths_removed": removed, "root": root}


@runtime.function(image=image, volumes={"/data": data}, timeout=7200)
def cleanup_clips(root: str = RUN_ROOT, dry_run: bool = True):
    """Delete intermediate TTS clip directories, keeping every .npz artifact.

    keyword_generator caches 12 variant wavs per keyword and rival_builder ~112
    rival wavs per keyword, all under SIAMESE_PROJECT_ROOT - i.e. on the Volume.
    At this scale that is ~108k + ~220k files, and Modal Volumes cap at 500k
    INODES, not bytes: the run hit 99.6% (498,102/500,000) and further file
    creation would have started failing.

    The clips are pure intermediates - the anchor npz holds the centroid and
    positives, the rivals npz holds the centroids and delta - so scoring needs
    none of them. Removing them does mean a later re-run re-synthesises rather
    than reusing the cache, which is the correct trade at this scale.
    """
    import glob
    import shutil

    kdir = f"{root}/keywords"
    dirs = [d for d in glob.glob(f"{kdir}/*_variants") + glob.glob(f"{kdir}/*_rivals")
            if os.path.isdir(d)]
    n_files = 0
    for d in dirs:
        n_files += sum(len(f) for _, _, f in os.walk(d))
    if dry_run:
        return {"dry_run": True, "dirs": len(dirs), "files": n_files,
                "sample": [os.path.basename(d) for d in dirs[:5]]}
    removed = 0
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
        removed += 1
    data.commit()
    npz = len(glob.glob(f"{kdir}/*.npz"))
    return {"dry_run": False, "dirs_removed": removed, "files_freed": n_files,
            "npz_artifacts_kept": npz}


@runtime.function(image=image, volumes={"/data": data}, timeout=600)
def write_results(name: str, rows: list):
    os.makedirs(f"{RUN_ROOT}/results", exist_ok=True)
    path = f"{RUN_ROOT}/results/{name}.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    data.commit()
    return path


def _drive(name, jobs):
    """Fan out, summarise, and persist. Log retention is 1 day - the volume is
    the record, so every batch writes its results there."""
    rows, failed = [], []
    for r in enroll.map(jobs, order_outputs=False):
        rows.append(r)
        if not r.get("ok"):
            failed.append(r)
            print(f"FAIL {r['keyword']!r} at {r.get('stage')}: "
                  f"{r.get('err', '')[:300]}")
        if len(rows) % 100 == 0:
            print(f"  {len(rows)}/{len(jobs)} done, {len(failed)} failed")
    stage_totals = {}
    for r in rows:
        for k, v in (r.get("timings") or {}).items():
            stage_totals[k] = stage_totals.get(k, 0.0) + v
    print(f"\n{name}: {len(rows)} jobs, {len(failed)} failed")
    for k, v in sorted(stage_totals.items()):
        print(f"  {k:<20} {v / 3600:7.2f} core-hours total  "
              f"({v / max(1, len(rows)):6.1f}s mean)")
    print("results ->", write_results.remote(name, rows))
    return rows


@runtime.function(image=image, volumes={"/data": data}, cpu=2.0, memory=4096,
              timeout=7200)
def enroll_batch(jobs: list):
    """Run several keywords inside ONE container, sequentially.

    Needed for the memoization test: with .map() each keyword may land on its own
    container, where a memoized loader is never reused and the speedup is
    invisible. In the real run each of 100 containers handles ~90 jobs, so this
    is also the more faithful measurement of steady-state cost.
    """
    return [enroll.local(j) for j in jobs]


@runtime.function(image=image, volumes={"/data": data}, cpu=2.0, memory=4096,
              timeout=1800)
def fingerprint(keywords: list, root: str):
    """Per-array SHA-256 of every artifact, for equivalence testing.

    Compares ARRAY CONTENTS, not file bytes: .npz is a zip whose entry
    timestamps differ between runs even when the arrays are identical.
    """
    import hashlib
    import sys

    import numpy as np

    sys.path.insert(0, "/app/core")
    from scoring import artifact_suffix
    suf = artifact_suffix()

    out = {}
    for kw in keywords:
        kw = str(kw).lower().strip()
        for label, p in (("anchor", f"{root}/keywords/{kw}_anchor{suf}.npz"),
                         ("cohort", f"{root}/keywords/cohort_{kw}{suf}.npz"),
                         ("rivals", f"{root}/keywords/{kw}_rivals{suf}.npz")):
            if not os.path.exists(p):
                out[f"{kw}|{label}"] = None
                continue
            z = np.load(p, allow_pickle=False)
            per = {}
            for k in sorted(z.files):
                a = np.ascontiguousarray(z[k])
                h = hashlib.sha256()
                h.update(str(a.dtype).encode())
                h.update(str(a.shape).encode())
                h.update(a.tobytes())
                per[k] = h.hexdigest()[:16]
            out[f"{kw}|{label}"] = per
    return out
