"""LibriPhrase enrollment on Modal - image, manifests, and the per-keyword unit.

Scope: exact PhonMatchNet protocol, 500 anchor classes, seed 777.
Cohort constants (C4, approved 2026-08-27): size 50 / top_k 50 unchanged,
per-keyword cohorts, stem-family exclusion for cohort hygiene, TTS distractors off.

Entrypoints, in the order they should be run:
    modal run modal_app.py::make_manifests     # sampling + sample table (cheap)
    modal run modal_app.py::smoke              # one keyword end to end
    modal run modal_app.py::calibrate          # 20 keywords -> real billed cost
    modal run modal_app.py::selection          # 40 held-out classes, BOTH cohorts
    modal run modal_app.py::t1                 # 9,030 keywords, anchor + cohort
    modal run modal_app.py::t2                 # 1,969 LP-Hard keywords, + rivals

Scoring lives in a separate module, written once the cohort choice is frozen.
"""

import json
import os

import modal

app = modal.App("vms-libriphrase")
data = modal.Volume.from_name("vms-data", create_if_missing=True)
hfcache = modal.Volume.from_name("vms-hfcache", create_if_missing=True)

# Repo root on the machine running `modal run`.
REPO = os.environ.get(
    "VMS_REPO", r"d:/Video Context Extraction System/Siamese_VMS_Project")

LP_DIR = "/data/libriphrase"
RUN_ROOT = "/data/run"                  # SIAMESE_PROJECT_ROOT -> artifacts land here
MANIFEST = f"{RUN_ROOT}/manifests"
COHORTS = {"devclean": "/data/cohort_src/devclean",
           "broadcast": "/data/cohort_src/broadcast"}

N_CLASSES = 500
N_SELECTION = 40
SEED = 777

# DECLARED CONSTANT - the TTS voice budget, fixed for every keyword.
# keyword_generator aborts when `len(clips) <= holdout + 2`, i.e. <= 6. With the
# 7 canonical voices alone, a SINGLE degenerate synthesis in any one voice kills
# the keyword - which silently cost 38 of 809 selection keywords (4.7%). Adding
# 5 random-x-vector voices gives 12, so up to 5 may fail before the abort.
# Changing this changes every anchor, so it must be fixed BEFORE any run whose
# artifacts are kept, and stated as 12 voices in the paper.
STAGE_ARGS = {"keyword_generator": ["--n-random", "5"]}

# Verified locally 2026-08-27 with a FRESH default_rng(777) and a single 500-draw,
# under the per-split convention (see expand_split). Asserted so any drift in
# pandas/numpy sampling is caught before credits are spent.
# Internal check these satisfy: positives == unique clips in each split, because
# each clip contributes exactly one "paired with its own transcript" positive.
EXPECT = {
    "easy_samples": 29417, "easy_keywords": 7643, "easy_clips": 11746,
    "hard_samples": 20564, "hard_keywords": 1969, "hard_clips": 7325,
    "table_rows": 49981, "keywords": 9030, "clips": 16101,
    "sel_keywords": 809, "sel_rows": 3656,
}


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
    .add_local_dir(f"{REPO}/core", remote_path="/app/core")
    .add_local_dir(f"{REPO}/pipeline", remote_path="/app/pipeline")
    .add_local_file(f"{REPO}/checkpoints/siamese_v3_best.pth",
                    remote_path="/app/checkpoints/siamese_v3_best.pth")
    .add_local_file(f"{REPO}/keywords/rival_bank_wavlm10ft.npz",
                    remote_path="/app/keywords/rival_bank_wavlm10ft.npz")
)


# --------------------------------------------------------------------------
# Manifests - the single place the seed-777 sampling is defined
# --------------------------------------------------------------------------

@app.function(image=image, volumes={"/data": data}, timeout=1800)
def make_manifests():
    """Sample anchor classes, expand to samples, write manifests + sample table."""
    import numpy as np
    import pandas as pd

    dfs = [pd.read_csv(f"{LP_DIR}/libriphrase_diffspk_all_{n}word.csv")
           for n in (1, 2, 3, 4)]
    allf = pd.concat(dfs, ignore_index=True)
    neg = allf[allf["type"].isin(["diffspk_easyneg", "diffspk_hardneg"])].copy()

    # Draw order is load-bearing: a FRESH generator, the 500 first, then the
    # disjoint 40 from what remains. Changing the order changes both sets.
    rng = np.random.default_rng(SEED)
    anchors = np.array(sorted(neg["anchor_text"].astype(str).unique()))
    picked = rng.choice(anchors, N_CLASSES, replace=False)
    remaining = np.array(sorted(set(anchors) - set(picked)))
    selection = rng.choice(remaining, N_SELECTION, replace=False)
    assert not (set(picked) & set(selection)), "selection set must be disjoint"

    def expand(df):
        """PhonMatchNet dataset/libriphrase.py: one CSV row -> four samples.

        Positives come from pairing each clip with its OWN transcript; the
        negatives are the cross-pairings carrying the row's `target`. Both
        inherit the row's `type`, which is why filtering on type keeps both
        classes. Verified verbatim against upstream.
        """
        parts = [
            pd.DataFrame({"wav": df["anchor"], "text": df["anchor_text"],
                          "label": 1, "type": df["type"], "src": "anc_pos"}),
            pd.DataFrame({"wav": df["anchor"], "text": df["comparison_text"],
                          "label": df["target"], "type": df["type"], "src": "anc_neg"}),
            pd.DataFrame({"wav": df["comparison"], "text": df["comparison_text"],
                          "label": 1, "type": df["type"], "src": "com_pos"}),
            pd.DataFrame({"wav": df["comparison"], "text": df["anchor_text"],
                          "label": df["target"], "type": df["type"], "src": "com_neg"}),
        ]
        out = pd.concat(parts, ignore_index=True)
        out["text"] = out["text"].astype(str).str.lower()
        return out.drop_duplicates(subset=["wav", "text"])

    def expand_split(df):
        """LP-Easy and LP-Hard are INDEPENDENT evaluations.

        train.py constructs test_easy_dataset and test_hard_dataset as separate
        LibriPhraseDataloader instances, each filtering the full CSV on `type`.
        So a (wav, text) pair occurring in both splits belongs to both, and the
        (wav, text) dedup must happen WITHIN a split, never across them -
        deduping globally first silently reassigns shared pairs to whichever
        split happened to come first and shrinks the other.
        """
        e = expand(df[df["type"] == "diffspk_easyneg"])
        h = expand(df[df["type"] == "diffspk_hardneg"])
        return e, h

    def kws(df):
        return sorted(set(df["text"]))

    sub = neg[neg["anchor_text"].astype(str).isin(set(picked))]
    easy, hard = expand_split(sub)
    table = pd.concat([easy, hard], ignore_index=True)
    sel_easy, sel_hard = expand_split(
        neg[neg["anchor_text"].astype(str).isin(set(selection))])
    sel = pd.concat([sel_easy, sel_hard], ignore_index=True)

    got = {
        "easy_samples": len(easy), "easy_keywords": len(kws(easy)),
        "easy_clips": easy["wav"].nunique(),
        "hard_samples": len(hard), "hard_keywords": len(kws(hard)),
        "hard_clips": hard["wav"].nunique(),
        "table_rows": len(table), "keywords": len(kws(table)),
        "clips": table["wav"].nunique(),
        "sel_keywords": len(kws(sel)), "sel_rows": len(sel),
    }
    print("counts:", json.dumps(got, indent=2))
    bad = {k: (got[k], v) for k, v in EXPECT.items() if got[k] != v}
    assert not bad, f"sampling drift vs locally verified counts: {bad}"
    for nm, d in (("easy", easy), ("hard", hard)):
        assert int((d["label"] == 1).sum()) == d["wav"].nunique(), (
            f"{nm}: positives must equal unique clips (one self-paired positive "
            f"per clip)")

    os.makedirs(MANIFEST, exist_ok=True)
    manifest = {
        "seed": SEED, "n_classes": N_CLASSES, "n_selection": N_SELECTION,
        "counts": got,
        "anchor_classes_500": sorted(map(str, picked)),
        "anchor_classes_40_holdout": sorted(map(str, selection)),
        "t1_keywords": kws(table),      # enrollment: union over both splits
        "t2_keywords": kws(hard),       # rivals only where RAV is evaluated
        "selection_keywords": kws(sel),
    }
    with open(f"{MANIFEST}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    table.to_csv(f"{MANIFEST}/samples_500.csv", index=False)
    sel.to_csv(f"{MANIFEST}/samples_selection40.csv", index=False)
    data.commit()

    print(f"t1 keywords {len(manifest['t1_keywords']):,}  "
          f"t2 {len(manifest['t2_keywords']):,}  "
          f"selection {len(manifest['selection_keywords']):,}")
    return got


# --------------------------------------------------------------------------
# The per-keyword unit of work
# --------------------------------------------------------------------------

# Volume commits are throttled per container. Committing once per keyword from
# up to 100 containers hammers the commit API - it produced
# `DataLossError: failed to publish commit to server` during the C4 re-run, and
# would be ~9,030 commits during T1. Modal already background-commits every few
# seconds, so explicit commits only need to be occasional; and because enroll is
# idempotent, anything lost to a preemption before a commit is simply rebuilt.
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
# failure mode is silent mis-filing of artifacts that become paper numbers.
# Not worth it. Enrollment runs one subprocess per stage, which is slower and
# cannot inherit stale module state.

@app.function(
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


# --------------------------------------------------------------------------
# Driver helpers
# --------------------------------------------------------------------------

@app.function(image=image, volumes={"/data": data}, timeout=600)
def read_manifest():
    with open(f"{MANIFEST}/manifest.json") as f:
        return json.load(f)


@app.function(image=image, volumes={"/data": data}, timeout=1800)
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


@app.function(image=image, volumes={"/data": data}, timeout=600)
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


# --------------------------------------------------------------------------
# Entrypoints
# --------------------------------------------------------------------------

@app.function(image=image, volumes={"/data": data}, cpu=2.0, memory=4096,
              timeout=7200)
def enroll_batch(jobs: list):
    """Run several keywords inside ONE container, sequentially.

    Needed for the memoization test: with .map() each keyword may land on its own
    container, where a memoized loader is never reused and the speedup is
    invisible. In the real run each of 100 containers handles ~90 jobs, so this
    is also the more faithful measurement of steady-state cost.
    """
    return [enroll.local(j) for j in jobs]


@app.function(image=image, volumes={"/data": data}, cpu=2.0, memory=4096,
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


@app.local_entrypoint()
def voicefix_test(cohort: str = "devclean"):
    """Does the 12-voice budget recover the keywords that failed under 7?

    Enrolls the observed failures into a scratch root and reports the recovery
    rate. Must live on this app: `enroll` cannot be called from another App's
    entrypoint (the function has no hydrated metadata unless its own App runs).
    """
    failed = [
        "acquainted with", "across", "assented", "bruce", "bruised",
        "circumstanced", "cross", "dai e", "english had", "excited face",
        "far when", "flat one", "for walking", "glistened in", "had stol'n thy",
        "his intended", "impossibility of", "maintained", "mind", "my reputation",
    ]
    scratch = f"{RUN_ROOT}_voicefix"
    purge.remote(failed, scratch)
    jobs = [{"keyword": k, "cohort": cohort, "need_rivals": False,
             "root": scratch} for k in failed]
    ok, bad = [], []
    for r in enroll.map(jobs, order_outputs=False):
        (ok if r.get("ok") else bad).append(r)
        if not r.get("ok"):
            print(f"  STILL FAILS {r['keyword']!r} at {r.get('stage')}: "
                  f"{(r.get('err') or '')[:240]}")
    print(f"\nrecovered {len(ok)}/{len(failed)}   still failing {len(bad)}")
    if ok:
        mean = sum(sum(r["timings"].values()) for r in ok) / len(ok)
        print(f"mean enrollment {mean:.1f}s per keyword under 12 voices "
              f"(~41s under 7)")
    print("VERDICT:", "voice budget fix works" if not bad
          else f"{len(bad)} keywords need a different remedy")


@app.local_entrypoint()
def purge_kw(keywords: str, root: str = ""):
    """CLI wrapper for purge(): `modal run` cannot parse a list annotation.

    Comma-separated, e.g. --keywords "a catholic church,dragon".
    """
    kws = [k.strip() for k in keywords.split(",") if k.strip()]
    print(purge.remote(kws, root or RUN_ROOT))


@app.local_entrypoint()
def smoke(cohort: str = "devclean"):
    """One keyword, all three stages, to prove the image and paths work."""
    m = read_manifest.remote()
    kw = m["selection_keywords"][0]
    print("smoke keyword:", kw)
    print(json.dumps(enroll.remote(
        {"keyword": kw, "cohort": cohort, "need_rivals": True}), indent=2))


@app.local_entrypoint()
def calibrate(n: int = 20, cohort: str = "devclean"):
    """Measure REAL billed cost before scaling. Check the dashboard afterwards."""
    m = read_manifest.remote()
    kws = m["selection_keywords"][:n]
    _drive("calibrate", [{"keyword": k, "cohort": cohort, "need_rivals": True}
                         for k in kws])
    print("\nNow read actual cost/duration from the Modal dashboard and "
          f"extrapolate to {m['counts']['keywords']:,} keywords before running t1.")


@app.local_entrypoint()
def selection(fresh: bool = False):
    """C4 A/B on the 40 held-out classes: same anchors, two cohort sources.

    Anchors do not depend on the cohort, so each arm writes to its own root and
    the second arm reuses nothing - keeping the two cohorts strictly separate.

    --fresh purges both roots first. Required whenever an enrollment constant
    changes (e.g. the TTS voice budget), because enroll is idempotent and would
    otherwise keep artifacts built under the old configuration.
    """
    m = read_manifest.remote()
    kws = m["selection_keywords"]
    for arm in ("devclean", "broadcast"):
        root = f"{RUN_ROOT}_sel_{arm}"
        if fresh:
            print(f"purging {root} ...", purge.remote(kws, root))
        print(f"\n=== C4 arm: {arm} ({len(kws):,} keywords) ===")
        _drive(f"selection_{arm}",
               [{"keyword": k, "cohort": arm, "need_rivals": False,
                 "root": root} for k in kws])
    print("\nDecide on the HELD-OUT 40 only, then freeze. Do not look at the 500.")


@app.local_entrypoint()
def t1(cohort: str = "devclean"):
    """Detector-only enrollment: anchor + cohort for every keyword, both splits."""
    m = read_manifest.remote()
    kws = m["t1_keywords"]
    print(f"T1: {len(kws):,} keywords, cohort={cohort}")
    _drive("t1", [{"keyword": k, "cohort": cohort, "need_rivals": False}
                  for k in kws])


@app.local_entrypoint()
def t2(cohort: str = "devclean"):
    """+RAV enrollment: rivals for the LP-Hard keyword subset only."""
    m = read_manifest.remote()
    kws = m["t2_keywords"]
    print(f"T2: {len(kws):,} LP-Hard keywords, cohort={cohort}")
    _drive("t2", [{"keyword": k, "cohort": cohort, "need_rivals": True}
                  for k in kws])
