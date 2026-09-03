"""Re-grade the stored LibriPhrase scores under several published protocols.

No re-scoring and no compute: this reads the per-sample CSVs written by
`modal_score.py` and only changes *which rows are counted* and *how they are
grouped*. Runs locally, CPU-only, in seconds.

Three outputs, matching E1/E2/E3 of
`Journal_Paper/meeting_notes_work/2026-09-02-cmcd-comparison.md`:

  E1  CMCD's own episode expansion: an enrolled anchor keyword scored against
      positive and negative audio phrases (`cmcd_enrolled`). Established from
      the data, not inferred from the prose - `anc_pos` and `com_neg` carry
      exactly the 500 anchor texts, while `com_pos` and `anc_neg` carry the
      7,238 negative phrases instead.
  E2  the same result under every protocol we can construct, to show the
      conclusion does not depend on the choice.
  E3  CMCD's per-word-length breakdown (its Figure 5), on our system.

The metric functions are copied verbatim from `modal_score.py::metrics`, which
in turn reproduces PhonMatchNet's `criterion/utils.py`. They are NOT reimplemented
here: variant `pmn_4trial` must reproduce `t1t2_metrics_500.json` exactly, and
that equality is asserted at the end as the harness sanity check.

Usage:
    python regrade_protocols.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP = os.path.join(REPO, "Journal_Paper", "experiments", "libriphrase")
SCORES = os.path.join(EXP, "scores", "scores")
MANIFESTS = os.path.join(EXP, "manifests")

EER_BATCH = 2048          # GLOBAL_BATCH_SIZE in their train.py (single replica)
EER_SEED = 777
LAMBDAS = [0.0, 0.5, 1.0, 2.0, 4.0]
REPORT_LAMBDA = 4.0       # fixed on 40 held-out classes; never re-selected here

SUBSETS = (("LP-Easy", "diffspk_easyneg"), ("LP-Hard", "diffspk_hardneg"))


# ---------------------------------------------------------------------------
# Metrics - verbatim from modal_score.py::metrics
# ---------------------------------------------------------------------------

def compute_eer(label, pred):
    """Verbatim equivalent of their criterion/utils.py compute_eer."""
    fpr, tpr, _ = roc_curve(label, pred)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.absolute(fnr - fpr)))
    return (fpr[i] + fnr[i]) / 2


def their_eer(label, pred, batch=EER_BATCH, seed=EER_SEED):
    """Their reported EER is the MEAN OF PER-BATCH EERs, not pooled."""
    idx = np.random.default_rng(seed).permutation(len(label))
    vals, skipped = [], 0
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        if len(np.unique(label[b])) < 2:
            skipped += 1              # EER undefined on a single-class batch
            continue
        vals.append(compute_eer(label[b], pred[b]))
    if not vals:
        return None, 0, skipped
    return float(np.mean(vals)) * 100, len(vals), skipped


def cascade_score(det, marg, delta, lam):
    """s_cascade = s_det - lambda * max(0, delta - margin); RAV abstains -> 0."""
    pen = np.nan_to_num(np.maximum(0.0, delta - marg), nan=0.0)
    return det - lam * pen


def score_block(d, lambdas=LAMBDAS, weights=None):
    """EER/AUC for the detector and each lambda over one selection of rows.

    `weights` repeats rows by integer multiplicity, for the non-deduplicated
    protocol. Duplicate trials carry identical scores by construction, so
    repeating rows is exact rather than an approximation.
    """
    y = d["label"].to_numpy()
    det = d["s_det"].to_numpy(float)
    marg = d["margin"].to_numpy(float)
    delta = d["delta"].to_numpy(float)
    if weights is not None:
        w = np.asarray(weights, dtype=int)
        rep = np.repeat(np.arange(len(d)), w)
        y, det, marg, delta = y[rep], det[rep], marg[rep], delta[rep]

    out = {"n": int(len(y)), "pos": int(y.sum()),
           "n_no_rivals": int(np.isnan(marg).sum())}
    if len(np.unique(y)) < 2:
        out["degenerate"] = True
        return out

    e, nb, sk = their_eer(y, det)
    out["detector"] = {"eer_batchavg": None if e is None else round(e, 2),
                       "eer_pooled": round(compute_eer(y, det) * 100, 2),
                       "auc_pooled": round(roc_auc_score(y, det) * 100, 2),
                       "n_batches": nb, "skipped_batches": sk}
    out["cascade"] = {}
    for lam in lambdas:
        s = cascade_score(det, marg, delta, lam)
        e, nb, sk = their_eer(y, s)
        out["cascade"][f"lambda={lam:g}"] = {
            "eer_batchavg": None if e is None else round(e, 2),
            "eer_pooled": round(compute_eer(y, s) * 100, 2),
            "auc_pooled": round(roc_auc_score(y, s) * 100, 2)}
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_scores():
    files = sorted(glob.glob(os.path.join(SCORES, "500_cas_*.csv")))
    if not files:
        sys.exit(f"no score shards under {SCORES}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["wav", "text", "type"])
    if len(df) != before:
        print(f"  note: dropped {before - len(df)} duplicate shard rows")
    df["n_words"] = df["text"].astype(str).str.split().str.len()
    print(f"loaded {len(df):,} scored samples from {len(files)} shard files")
    return df


# Protocols. Each returns a boolean mask over the loaded frame.
#   pmn_4trial    - PhonMatchNet: all four samples expanded from a CSV row.
#                   This is what t1t2_metrics_500.json reports.
#   cmcd_enrolled - CMCD's own description: "positive and negative audio phrases
#                   for enrolled keywords (anchor)". The enrolled text is always
#                   an anchor class (anc_pos and com_neg both carry exactly the
#                   500 anchor texts; com_pos and anc_neg carry the 7,238
#                   negative phrases instead). Audio varies, text is fixed.
#   cmcd_balanced - the comparison-clip side. Balanced 50/50, which is what
#                   CMCD's "3 positive and 3 negative" implies numerically, but
#                   it mixes two different enrolled keywords per episode, so it
#                   is reported as an alternative rather than as the protocol.
#   audio_fixed   - diagnostic: the anchor clip against its own text and against
#                   the other text. Fixed audio, varying text - the transpose of
#                   cmcd_enrolled.
PROTOCOLS = {
    "pmn_4trial": lambda d: pd.Series(True, index=d.index),
    "cmcd_enrolled": lambda d: d["src"].isin(["anc_pos", "com_neg"]),
    "cmcd_balanced": lambda d: d["src"].isin(["com_pos", "com_neg"]),
    "audio_fixed": lambda d: d["src"].isin(["anc_pos", "anc_neg"]),
}


def run_upstream_raw(df):
    """PhonMatchNet's dataloader as actually written: NO deduplication.

    Verified 2026-09-03 against upstream `dataset/libriphrase.py`, which appends
    all four expansions per CSV row and never filters duplicates. Our manifest
    builder DOES dedup on (wav, text) within a split, so every LibriPhrase number
    we have published is on a deduplicated trial set - a deviation that was never
    declared. `manifests/multiplicity_500.csv` carries the upstream repeat count
    for each of our 49,981 trials, so the undeduplicated metric is exact rather
    than estimated: duplicate trials are the same (clip, keyword) pair and
    therefore carry an identical score by construction.
    """
    mp = os.path.join(MANIFESTS, "multiplicity_500.csv")
    if not os.path.exists(mp):
        print("  raw_upstream    SKIPPED (no multiplicity_500.csv)")
        return None
    mult = pd.read_csv(mp)
    d = df.merge(mult, on=["wav", "text", "type"], how="left")
    if d["m"].isna().any():
        sys.exit(f"multiplicity file misses {int(d['m'].isna().sum())} rows")
    d["m"] = d["m"].astype(int)
    rows = []
    for label, typ in SUBSETS:
        dd = d[d["type"] == typ]
        blk = score_block(dd, weights=dd["m"].to_numpy())
        blk["subset"] = label
        blk["n_deduped"] = int(len(dd))
        rows.append(blk)
    total = int(d["m"].sum())
    print(f"  raw_upstream    {total:>6,} rows (from {len(d):,} unique trials)")
    return {"total_rows": total, "results": rows}


def run_protocols(df):
    out = {}
    for name, sel in PROTOCOLS.items():
        m = sel(df)
        rows = []
        for label, typ in SUBSETS:
            d = df[m & (df["type"] == typ)]
            if d.empty:
                continue
            blk = score_block(d)
            blk["subset"] = label
            rows.append(blk)
        out[name] = {"total_rows": int(m.sum()), "results": rows}
        print(f"  {name:<14} {int(m.sum()):>6,} rows")
    raw = run_upstream_raw(df)
    if raw is not None:
        out["raw_upstream"] = raw
    return out


def run_lengths(df):
    out = []
    for label, typ in SUBSETS:
        for n in sorted(df["n_words"].unique()):
            d = df[(df["type"] == typ) & (df["n_words"] == n)]
            if d.empty:
                continue
            blk = score_block(d, lambdas=[0.0, REPORT_LAMBDA])
            blk["subset"] = label
            blk["n_words"] = int(n)
            out.append(blk)
        # Multi-word keywords combined, for the "does the LP-Hard win survive
        # past one-word keywords" claim - a single number the per-bucket rows
        # cannot give directly (EER does not average linearly across buckets).
        d = df[(df["type"] == typ) & (df["n_words"] >= 2)]
        if not d.empty:
            blk = score_block(d, lambdas=[0.0, REPORT_LAMBDA])
            blk["subset"] = label
            blk["n_words"] = "2plus"
            out.append(blk)
    return out


# ---------------------------------------------------------------------------
# Reconciliation - every count must tie out, per section 8 of the plan
# ---------------------------------------------------------------------------

def reconcile(df, protocols, lengths):
    checks = []

    def chk(name, got, want):
        ok = got == want
        checks.append({"check": name, "got": got, "want": want, "ok": ok})
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: {got} vs {want}")
        return ok

    chk("total scored rows", int(len(df)), 49981)
    chk("LP-Easy rows", int((df["type"] == "diffspk_easyneg").sum()), 29417)
    chk("LP-Hard rows", int((df["type"] == "diffspk_hardneg").sum()), 20564)
    chk("subsets sum to total",
        int((df["type"] == "diffspk_easyneg").sum())
        + int((df["type"] == "diffspk_hardneg").sum()), 49981)

    for name, want in (("cmcd_enrolled", 19126), ("cmcd_balanced", 26413),
                       ("audio_fixed", 23568)):
        chk(f"{name} row count", protocols[name]["total_rows"], want)
    for r in protocols["cmcd_enrolled"]["results"]:
        chk(f"cmcd_enrolled {r['subset']} rows", r["n"],
            11779 if r["subset"] == "LP-Easy" else 7347)
    for r in protocols["cmcd_balanced"]["results"]:
        chk(f"cmcd_balanced {r['subset']} rows", r["n"],
            17633 if r["subset"] == "LP-Easy" else 8780)

    for label, typ in SUBSETS:
        # exclude the "2plus" combined row: it overlaps buckets 2/3/4, so
        # including it would double-count against the subset total.
        tot = sum(r["n"] for r in lengths
                  if r["subset"] == label and r["n_words"] != "2plus")
        chk(f"{label} length buckets sum",
            tot, int((df["type"] == typ).sum()))

    if "raw_upstream" in protocols:
        chk("raw_upstream total rows", protocols["raw_upstream"]["total_rows"],
            70704)
        for r in protocols["raw_upstream"]["results"]:
            chk(f"raw_upstream {r['subset']} rows", r["n"], 35352)

    return checks


def sanity_against_stored(protocols):
    """pmn_4trial must reproduce the published metrics file exactly."""
    p = os.path.join(EXP, "t1t2_metrics_500.json")
    stored = json.load(open(p, encoding="utf-8"))
    ours = {r["subset"]: r for r in protocols["pmn_4trial"]["results"]}
    diffs = []
    for s in stored["results"]:
        o = ours[s["subset"]]
        for k in ("n", "pos", "n_no_rivals"):
            if s[k] != o[k]:
                diffs.append(f"{s['subset']}.{k}: stored {s[k]} vs ours {o[k]}")
        for k, v in s["detector"].items():
            if o["detector"].get(k) != v:
                diffs.append(f"{s['subset']}.detector.{k}: "
                             f"stored {v} vs ours {o['detector'].get(k)}")
        for lam, vals in s["cascade"].items():
            for k, v in vals.items():
                if o["cascade"][lam].get(k) != v:
                    diffs.append(f"{s['subset']}.{lam}.{k}: "
                                 f"stored {v} vs ours {o['cascade'][lam].get(k)}")
    return diffs


SEEDS = [777, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 42, 123, 2024, 31337, 99991]


def run_seed_sensitivity(df):
    """How much does the batch-averaged EER move with the shuffle seed?

    Their metric averages EER over shuffled batches of 2,048, so the reported
    value is one draw from a distribution. Pooled EER has no seed. This
    quantifies the spread so the paper does not argue over differences smaller
    than the noise - including PhonMatchNet's own reported figures, whose seed
    is unknowable.
    """
    out = []
    for name, sel in PROTOCOLS.items():
        m = sel(df)
        for label, typ in SUBSETS:
            d = df[m & (df["type"] == typ)]
            if d.empty:
                continue
            y = d["label"].to_numpy()
            det = d["s_det"].to_numpy(float)
            vals = []
            for s in SEEDS:
                e, _, _ = their_eer(y, det, seed=s)
                if e is not None:
                    vals.append(round(e, 4))
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            out.append({
                "protocol": name, "subset": label, "n": int(len(d)),
                "n_seeds": len(vals),
                "eer_pooled": round(compute_eer(y, det) * 100, 2),
                "eer_batchavg_seed777": round(vals[0], 2),
                "min": round(min(vals), 2), "max": round(max(vals), 2),
                "spread": round(max(vals) - min(vals), 2),
                "mean": round(mean, 2), "sd": round(var ** 0.5, 3),
            })
            print(f"  {name:<14} {label:<8} spread {out[-1]['spread']:>5.2f} "
                  f"(sd {out[-1]['sd']:.3f})")

    # Same test over the E3 word-length buckets, which back the
    # "the LP-Hard win is a one-word effect" claim. Buckets smaller than one
    # batch (2,048) have a single batch, so batch-averaged == pooled and the
    # spread is necessarily 0 - reported rather than hidden.
    for label, typ in SUBSETS:
        for n in sorted(df["n_words"].unique()) + ["2plus"]:
            d = (df[(df["type"] == typ) & (df["n_words"] >= 2)] if n == "2plus"
                 else df[(df["type"] == typ) & (df["n_words"] == n)])
            if d.empty:
                continue
            y = d["label"].to_numpy()
            det = d["s_det"].to_numpy(float)
            vals = []
            for s in SEEDS:
                e, _, _ = their_eer(y, det, seed=s)
                if e is not None:
                    vals.append(round(e, 4))
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            out.append({
                "protocol": "pmn_4trial", "subset": label,
                "n_words": int(n) if n != "2plus" else "2plus",
                "n": int(len(d)), "n_seeds": len(vals),
                "eer_pooled": round(compute_eer(y, det) * 100, 2),
                "eer_batchavg_seed777": round(vals[0], 2),
                "min": round(min(vals), 2), "max": round(max(vals), 2),
                "spread": round(max(vals) - min(vals), 2),
                "mean": round(mean, 2), "sd": round(var ** 0.5, 3),
            })
            print(f"  by-length     {label:<8} {str(n):<6} "
                  f"spread {out[-1]['spread']:>5.2f}")
    return out


def main():
    print("== load ==")
    df = load_scores()

    print("== E1/E2: protocols ==")
    protocols = run_protocols(df)

    print("== E3: word-length breakdown ==")
    lengths = run_lengths(df)

    print("== seed sensitivity of batch-averaged EER ==")
    seeds = run_seed_sensitivity(df)

    print("== reconciliation ==")
    checks = reconcile(df, protocols, lengths)

    print("== sanity: pmn_4trial vs t1t2_metrics_500.json ==")
    diffs = sanity_against_stored(protocols)
    if diffs:
        print("  MISMATCH - the regrade does not reproduce the published run:")
        for d in diffs:
            print("   ", d)
    else:
        print("  [ok ] every stored value reproduced exactly")

    os.makedirs(EXP, exist_ok=True)
    meta = {"eer_batch": EER_BATCH, "eer_seed": EER_SEED,
            "lambdas": LAMBDAS, "report_lambda": REPORT_LAMBDA,
            "n_rows_loaded": int(len(df)),
            "reconciliation": checks,
            "reproduces_t1t2_metrics_500": not diffs,
            "reproduction_diffs": diffs}

    p1 = os.path.join(EXP, "e1e2_protocol_variants_500.json")
    json.dump({**meta, "protocols": protocols}, open(p1, "w", encoding="utf-8"),
              indent=2)
    p2 = os.path.join(EXP, "e3_length_breakdown_500.json")
    json.dump({**meta, "lengths": lengths}, open(p2, "w", encoding="utf-8"),
              indent=2)
    p3 = os.path.join(EXP, "eer_seed_sensitivity_500.json")
    json.dump({**meta, "seeds_tested": SEEDS, "results": seeds},
              open(p3, "w", encoding="utf-8"), indent=2)
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    print(f"wrote {p3}")

    if diffs or not all(c["ok"] for c in checks):
        sys.exit("FAILED: reconciliation or reproduction check did not pass")


if __name__ == "__main__":
    main()
