"""LibriPhrase benchmark harness - app, manifests and entrypoints.

The shared image and the per-keyword enrollment unit live in
`benchmarks/common/runtime.py` and are composed in with `App.include`; this file
holds only what is specific to LibriPhrase.

Scope: exact PhonMatchNet protocol, 500 anchor classes, seed 777.
Cohort constants (C4, approved 2026-08-27): size 50 / top_k 50 unchanged,
per-keyword cohorts, stem-family exclusion for cohort hygiene, TTS distractors off.

Entrypoints, in the order they should be run:
    modal run harness.py::make_manifests     # sampling + sample table (cheap)
    modal run harness.py::smoke              # one keyword end to end
    modal run harness.py::calibrate          # 20 keywords -> real billed cost
    modal run harness.py::selection          # 40 held-out classes, BOTH cohorts
    modal run harness.py::t1                 # 9,030 keywords, anchor + cohort
    modal run harness.py::t2                 # 1,969 LP-Hard keywords, + rivals

Scoring for this benchmark and for Speech Commands lives in
`benchmarks/common/modal_score.py`.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal

from common.runtime import (COHORTS, REPO, RUN_ROOT, STAGE_ARGS, data,  # noqa: F401
                            _drive, _maybe_commit, enroll, enroll_batch,
                            fingerprint, hfcache, image, runtime,
                            write_results)

app = modal.App("vms-libriphrase")
app.include(runtime)

LP_DIR = "/data/libriphrase"
MANIFEST = f"{RUN_ROOT}/manifests"

N_CLASSES = 500
N_SELECTION = 40
SEED = 777

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


@app.function(image=image, volumes={"/data": data}, timeout=600)
def read_manifest():
    with open(f"{MANIFEST}/manifest.json") as f:
        return json.load(f)


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


@app.function(image=image, volumes={"/data": data}, timeout=900)
def selection_hard_keywords():
    """LP-Hard keywords of the 40-class holdout - the set lambda is chosen on."""
    import pandas as pd
    df = pd.read_csv(f"{MANIFEST}/samples_selection40.csv")
    hard = df[df["type"] == "diffspk_hardneg"]
    return sorted(set(hard["text"].astype(str).str.lower()))


@app.local_entrypoint()
def lambda_select(cohort: str = "devclean"):
    """Enroll rivals for the holdout's LP-Hard keywords so lambda can be chosen
    OFF the reported set.

    Reading lambda off the 500 would be selection on the test set - the same
    fault we note in PhonMatchNet's section 4.2 ("the best model was selected
    based on performance on the test sets"). The 40 anchor classes here are
    disjoint from the reported 500 by construction (make_manifests draws them
    from the remainder and asserts disjointness).

    Artifacts go into RUN_ROOT: keywords shared with the 500 already exist and
    are skipped, and per-keyword artifacts are deterministic, so reuse is exact
    rather than approximate.
    """
    kws = selection_hard_keywords.remote()
    print(f"holdout LP-Hard keywords: {len(kws):,}")
    _drive("lambda_select", [{"keyword": k, "cohort": cohort, "need_rivals": True}
                             for k in kws])


@app.local_entrypoint()
def cleanup(root: str = "", apply: bool = False):
    """Free Volume inodes by deleting intermediate clip dirs. Dry run by default."""
    print(cleanup_clips.remote(root or RUN_ROOT, dry_run=not apply))


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
