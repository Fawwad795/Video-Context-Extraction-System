"""Score the LibriPhrase sample table and compute the metrics PhonMatchNet reports.

Two-pass design, both shardable across Modal containers:

  Pass 1  embed_clips   - one WavLM whole-clip forward per UNIQUE clip (197 ms
                          each; 16,101 clips carry 47,006 samples, 2.92 apiece,
                          so embedding per clip rather than per sample is a ~3x
                          saving). Embeddings -> /data/run/embeddings/*.npz
  Pass 2  score_shard   - per keyword: load its anchor/cohort (+rivals), pull the
                          cached clip embeddings, and record the RAW COMPONENTS
                          s_det, margin and delta per sample. Lambda is NOT
                          applied here, so it can be swept afterwards without
                          re-scoring anything.

  metrics       - aggregates. Implements their EER exactly (batch-averaged) plus
                  pooled EER and pooled AUC.

Whole-clip scoring, no sliding window and no threshold: verified in B1 that the
multi-scale detector silently drops clips whose frame count is below the window
(8/840), while whole-clip scores every clip and scores slightly better.

Run:
    modal run common/modal_score.py::embed --split 500
    modal run common/modal_score.py::score --split 500 --need-margin
    modal run common/modal_score.py::report --split 500
"""

import json
import os

import modal

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.runtime import RUN_ROOT, data, image

app = modal.App("vms-libriphrase-score")

LP_AUDIO = "/data/libriphrase/audio"
MANIFEST = f"{RUN_ROOT}/manifests"
EMB_DIR = f"{RUN_ROOT}/embeddings"
SCORE_DIR = f"{RUN_ROOT}/scores"

EER_BATCH = 2048          # GLOBAL_BATCH_SIZE in their train.py (single replica)
EER_SEED = 777
N_EMB_SHARDS = 64
N_SCORE_SHARDS = 32

SPLITS = {"500": "samples_500.csv", "sel": "samples_selection40.csv",
          "gsc": "samples_gsc.csv"}

# Audio root per split. LibriPhrase splits keep LP_AUDIO; the Speech Commands
# prong (P5) reads its own staged copy. Additive: the "500"/"sel" behaviour that
# produced t1t2_metrics_500.json is unchanged.
AUDIO_ROOTS = {"gsc": "/data/gsc/audio"}


def _audio_root(split):
    return AUDIO_ROOTS.get(split, LP_AUDIO)


# --------------------------------------------------------------------------
# Pass 1 - embed every unique clip once
# --------------------------------------------------------------------------

@app.function(image=image, volumes={"/data": data}, cpu=2.0, memory=8192,
              timeout=5400, retries=2, max_containers=64)
def embed_clips(task: dict):
    import sys

    import librosa
    import numpy as np
    import pandas as pd

    sys.path.insert(0, "/app/core")
    from scoring import SAMPLE_RATE, embed_batch, l2_normalize, load_siamese_model

    shard, n_shards, split = task["shard"], task["n_shards"], task["split"]
    out = f"{EMB_DIR}/{split}_{shard:04d}.npz"
    if os.path.exists(out):
        return {"shard": shard, "skipped": True}

    df = pd.read_csv(f"{MANIFEST}/{SPLITS[split]}")
    clips = sorted(df["wav"].astype(str).unique())
    mine = [c for i, c in enumerate(clips) if i % n_shards == shard]

    model = load_siamese_model()
    names, embs, failed = [], [], []
    for rel in mine:
        try:
            y, _ = librosa.load(os.path.join(_audio_root(split), rel),
                                sr=SAMPLE_RATE)
            e = embed_batch(model, [y.astype(np.float32)])[0]
        except Exception as exc:
            failed.append(f"{rel}: {exc}")
            continue
        names.append(rel)
        embs.append(e)

    os.makedirs(EMB_DIR, exist_ok=True)
    np.savez(out, names=np.array(names), embs=l2_normalize(np.stack(embs)))
    data.commit()
    return {"shard": shard, "n": len(names), "failed": failed[:5],
            "n_failed": len(failed)}


# --------------------------------------------------------------------------
# Pass 2 - per-keyword scoring, storing raw components
# --------------------------------------------------------------------------

@app.function(image=image, volumes={"/data": data}, cpu=2.0, memory=16384,
              timeout=5400, retries=2, max_containers=32)
def score_shard(task: dict):
    import sys

    import numpy as np
    import pandas as pd

    sys.path.insert(0, "/app/core")
    import rival_verify as rv
    from scoring import (DEFAULT_TOP_K, artifact_suffix, asnorm_windows,
                         l2_normalize)

    shard, n_shards = task["shard"], task["n_shards"]
    split, need_margin = task["split"], task.get("need_margin", False)
    root = task.get("root", RUN_ROOT)
    tag = task.get("tag", "")          # distinguishes the two C4 cohort arms,
                                       # which otherwise collide on filename
    out = (f"{SCORE_DIR}/{split}{tag}_"
           f"{'cas' if need_margin else 'det'}_{shard:04d}.csv")
    # force=True is required after ANY change to the artifacts being scored
    # (e.g. the TTS voice budget): without it the shard is skipped and stale
    # scores from the previous configuration are silently reused.
    if os.path.exists(out) and not task.get("force"):
        return {"shard": shard, "skipped": True}

    # clip embeddings, gathered from every pass-1 shard
    import glob
    lookup = {}
    for f in sorted(glob.glob(f"{EMB_DIR}/{split}_*.npz")):
        z = np.load(f, allow_pickle=False)
        for nm, e in zip(z["names"], z["embs"]):
            lookup[str(nm)] = e

    df = pd.read_csv(f"{MANIFEST}/{SPLITS[split]}")
    df["text"] = df["text"].astype(str).str.lower()
    keywords = sorted(df["text"].unique())
    mine = {k for i, k in enumerate(keywords) if i % n_shards == shard}
    df = df[df["text"].isin(mine)]

    suf = artifact_suffix()
    kdir = f"{root}/keywords"
    rows, missing = [], []
    for kw, grp in df.groupby("text", sort=True):
        ap = f"{kdir}/{kw}_anchor{suf}.npz"
        cp = f"{kdir}/cohort_{kw}{suf}.npz"
        if not (os.path.exists(ap) and os.path.exists(cp)):
            missing.append(kw)
            continue
        anchor = l2_normalize(np.load(ap)["centroid"])
        cohort = l2_normalize(np.load(cp)["embeddings"])

        rcent = delta = None
        if need_margin:
            loaded = rv.load_rivals(f"{kdir}/{kw}_rivals{suf}.npz")
            if loaded is not None and len(loaded[1]):
                _, rcent, delta = loaded

        keys = grp["wav"].astype(str).tolist()
        have = [k for k in keys if k in lookup]
        if not have:
            continue
        E = np.stack([lookup[k] for k in have])
        s_det, _raw = asnorm_windows(E, anchor, cohort, DEFAULT_TOP_K)
        if rcent is not None:
            marg = np.asarray(rv.margins(E, anchor, rcent, cohort), float).ravel()
        else:
            # No armed rivals: declared fallback is RAV abstains, i.e. the
            # cascade score reduces to the detector score. Recorded explicitly
            # rather than imputed later.
            marg = np.full(len(have), np.nan)
            delta = np.nan

        idx = {k: i for i, k in enumerate(have)}
        for _, r in grp.iterrows():
            w = str(r["wav"])
            if w not in idx:
                continue
            i = idx[w]
            rows.append({"wav": w, "text": kw, "label": int(r["label"]),
                         "type": r["type"], "src": r.get("src", ""),
                         "s_det": float(s_det[i]),
                         "margin": float(marg[i]),
                         "delta": float(delta) if delta is not None else np.nan})

    os.makedirs(SCORE_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    data.commit()
    return {"shard": shard, "n": len(rows), "missing_keywords": len(missing),
            "missing_sample": missing[:5]}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

@app.function(image=image, volumes={"/data": data}, cpu=2.0, memory=16384,
              timeout=3600)
def metrics(split: str = "500", lambdas: str = "0,0.5,1,2,4", tag: str = ""):
    import glob

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve

    def compute_eer(label, pred):
        """Verbatim equivalent of their criterion/utils.py compute_eer."""
        fpr, tpr, _ = roc_curve(label, pred)
        fnr = 1 - tpr
        i = int(np.nanargmin(np.absolute(fnr - fpr)))
        return (fpr[i] + fnr[i]) / 2

    def their_eer(label, pred, batch=EER_BATCH, seed=EER_SEED):
        """Their reported EER is the MEAN OF PER-BATCH EERs, not pooled.

        criterion/utils.py's eer metric does score += compute_eer(batch);
        count += 1; result = score/count, and train.py drives it per batch with
        GLOBAL_BATCH_SIZE = 2048 and shuffle=True. AUC, by contrast, uses
        tf.keras.metrics.AUC and IS pooled.
        """
        idx = np.random.default_rng(seed).permutation(len(label))
        vals, skipped = [], 0
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            if len(np.unique(label[b])) < 2:
                skipped += 1          # EER undefined on a single-class batch
                continue
            vals.append(compute_eer(label[b], pred[b]))
        return float(np.mean(vals)) * 100, len(vals), skipped

    files = sorted(glob.glob(f"{SCORE_DIR}/{split}{tag}_*_*.csv"))
    if not files:
        return {"error": f"no score files under {SCORE_DIR} for split {split}"}
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Dedup must include `type`: LP-Easy and LP-Hard are independent evaluations,
    # so a (wav, text) pair present in both belongs to both. Deduping on
    # (wav, text) alone would delete it from one split.
    df = df.drop_duplicates(subset=["wav", "text", "type"])
    print(f"loaded {len(df):,} scored samples from {len(files)} shard files")

    lam_list = [float(x) for x in lambdas.split(",")]
    out = {"split": split, "tag": tag, "n_samples": int(len(df)),
           "eer_batch": EER_BATCH, "eer_seed": EER_SEED, "results": []}

    for name, sel in (("LP-Easy", df["type"] == "diffspk_easyneg"),
                      ("LP-Hard", df["type"] == "diffspk_hardneg")):
        d = df[sel]
        if d.empty:
            continue
        y = d["label"].to_numpy()
        det = d["s_det"].to_numpy(float)
        marg = d["margin"].to_numpy(float)
        delta = d["delta"].to_numpy(float)

        row = {"subset": name, "n": int(len(d)), "pos": int(y.sum()),
               "n_no_rivals": int(np.isnan(marg).sum())}
        e, nb, sk = their_eer(y, det)
        row["detector"] = {"eer_batchavg": round(e, 2),
                           "eer_pooled": round(compute_eer(y, det) * 100, 2),
                           "auc_pooled": round(roc_auc_score(y, det) * 100, 2),
                           "n_batches": nb, "skipped_batches": sk}
        row["cascade"] = {}
        for lam in lam_list:
            # RAV abstains where no rivals armed -> penalty 0 (NaN-safe).
            pen = np.nan_to_num(np.maximum(0.0, delta - marg), nan=0.0)
            s = det - lam * pen
            e, nb, sk = their_eer(y, s)
            row["cascade"][f"lambda={lam:g}"] = {
                "eer_batchavg": round(e, 2),
                "eer_pooled": round(compute_eer(y, s) * 100, 2),
                "auc_pooled": round(roc_auc_score(y, s) * 100, 2)}
        out["results"].append(row)

    os.makedirs(f"{RUN_ROOT}/results", exist_ok=True)
    p = f"{RUN_ROOT}/results/metrics_{split}{tag}.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    data.commit()
    print(json.dumps(out, indent=2))
    print("\nPhonMatchNet, as reported: LP-Easy 2.80 / 99.29   LP-Hard 18.82 / 88.52")
    return out


@app.function(image=image, volumes={"/data": data}, cpu=2.0, memory=16384,
              timeout=3600)
def metrics_gsc(lambdas: str = "0,0.5,1,2,4", tag: str = ""):
    """Speech Commands metrics: overall, then split by training-vocabulary group.

    Separate from metrics() rather than folded into it: that function produces
    the published LibriPhrase numbers and regrade_protocols.py asserts against
    them, so it is left byte-for-byte alone. compute_eer/their_eer below are
    copies of its definitions, which are in turn PhonMatchNet's criterion/utils.py.

    Groups come from the manifest's `src` column, set by modal_gsc.py from
    Journal_Paper/experiments/gsc/mswc_vocab_check.json: `unseen` = the four words
    absent from MSWC entirely, `seen` = the six that were training classes.
    """
    import glob

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve

    def compute_eer(label, pred):
        fpr, tpr, _ = roc_curve(label, pred)
        fnr = 1 - tpr
        i = int(np.nanargmin(np.absolute(fnr - fpr)))
        return (fpr[i] + fnr[i]) / 2

    def their_eer(label, pred, batch=EER_BATCH, seed=EER_SEED):
        idx = np.random.default_rng(seed).permutation(len(label))
        vals, skipped = [], 0
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            if len(np.unique(label[b])) < 2:
                skipped += 1
                continue
            vals.append(compute_eer(label[b], pred[b]))
        return float(np.mean(vals)) * 100, len(vals), skipped

    files = sorted(glob.glob(f"{SCORE_DIR}/gsc{tag}_*_*.csv"))
    if not files:
        return {"error": f"no gsc score files under {SCORE_DIR}"}
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["wav", "text", "type"])
    print(f"loaded {len(df):,} scored trials from {len(files)} shard files")

    lam_list = [float(x) for x in lambdas.split(",")]
    out = {"split": "gsc", "tag": tag, "n_samples": int(len(df)),
           "eer_batch": EER_BATCH, "eer_seed": EER_SEED,
           "protocol": "closed_10 (PhonMatchNet dataset/google.py)",
           "results": []}

    groups = [("all", df["text"].notna()),
              ("seen-6", df["src"] == "seen"),
              ("unseen-4", df["src"] == "unseen")]
    groups += [(f"word:{w}", df["text"] == w) for w in sorted(df["text"].unique())]

    for name, sel in groups:
        d = df[sel]
        if d.empty or d["label"].nunique() < 2:
            continue
        y = d["label"].to_numpy()
        det = d["s_det"].to_numpy(float)
        marg = d["margin"].to_numpy(float)
        delta = d["delta"].to_numpy(float)

        row = {"subset": name, "n": int(len(d)), "pos": int(y.sum()),
               "n_no_rivals": int(np.isnan(marg).sum())}
        e, nb, sk = their_eer(y, det)
        row["detector"] = {"eer_batchavg": round(e, 2),
                           "eer_pooled": round(compute_eer(y, det) * 100, 2),
                           "auc_pooled": round(roc_auc_score(y, det) * 100, 2),
                           "n_batches": nb, "skipped_batches": sk}
        row["cascade"] = {}
        for lam in lam_list:
            pen = np.nan_to_num(np.maximum(0.0, delta - marg), nan=0.0)
            s = det - lam * pen
            e, nb, sk = their_eer(y, s)
            row["cascade"][f"lambda={lam:g}"] = {
                "eer_batchavg": round(e, 2),
                "eer_pooled": round(compute_eer(y, s) * 100, 2),
                "auc_pooled": round(roc_auc_score(y, s) * 100, 2)}
        out["results"].append(row)

    os.makedirs(f"{RUN_ROOT}/results", exist_ok=True)
    p = f"{RUN_ROOT}/results/metrics_gsc{tag}.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    data.commit()
    print(json.dumps(out, indent=2))
    print("\nAs reported on this set: CED 13.45 EER / 93.94 AUC   "
          "CMCD 27.25 / 81.06")
    return out


# --------------------------------------------------------------------------
# Entrypoints
# --------------------------------------------------------------------------

@app.local_entrypoint()
def embed(split: str = "500", n_shards: int = N_EMB_SHARDS):
    tasks = [{"shard": i, "n_shards": n_shards, "split": split}
             for i in range(n_shards)]
    tot = nf = 0
    for r in embed_clips.map(tasks, order_outputs=False):
        tot += r.get("n", 0)
        nf += r.get("n_failed", 0)
        if r.get("failed"):
            print("  failures:", r["failed"])
    print(f"embedded {tot:,} clips, {nf} failed")


@app.local_entrypoint()
def score(split: str = "500", need_margin: bool = False,
          n_shards: int = N_SCORE_SHARDS, root: str = "", tag: str = "",
          force: bool = False):
    tasks = [{"shard": i, "n_shards": n_shards, "split": split, "tag": tag,
              "need_margin": need_margin, "force": force,
              **({"root": root} if root else {})}
             for i in range(n_shards)]
    tot = miss = 0
    for r in score_shard.map(tasks, order_outputs=False):
        tot += r.get("n", 0)
        miss += r.get("missing_keywords", 0)
        if r.get("missing_sample"):
            print("  missing anchors/cohorts for e.g.", r["missing_sample"])
    print(f"scored {tot:,} samples; {miss} keywords lacked artifacts")


@app.local_entrypoint()
def report(split: str = "500", lambdas: str = "0,0.5,1,2,4", tag: str = ""):
    metrics.remote(split=split, lambdas=lambdas, tag=tag)


@app.local_entrypoint()
def report_gsc(lambdas: str = "0,0.5,1,2,4", tag: str = ""):
    metrics_gsc.remote(lambdas=lambdas, tag=tag)
