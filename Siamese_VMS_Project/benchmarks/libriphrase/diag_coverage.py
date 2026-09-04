"""Why are ~4% of selection samples missing, and why do the two arms differ?

Checks three possible causes independently:
  1. keywords with no anchor/cohort artifact in a given root
  2. clips with no embedding (embed_clips failures)
  3. samples dropped because either of the above applied
"""
import modal

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.runtime import RUN_ROOT, data, image

MANIFEST = f"{RUN_ROOT}/manifests"

# Inlined rather than imported from modal_score: Modal automounts the entrypoint
# file and modal_app (which defines the image object it has to serialise), but
# NOT a third local module - importing modal_score here crash-loops every
# container with ModuleNotFoundError before any user code runs.
EMB_DIR = f"{RUN_ROOT}/embeddings"
SPLITS = {"500": "samples_500.csv", "sel": "samples_selection40.csv"}

app = modal.App("vms-diag-coverage")


@app.function(image=image, volumes={"/data": data}, cpu=2.0, memory=8192,
              timeout=1800)
def diagnose(split: str = "sel"):
    import glob
    import os
    import sys

    import numpy as np
    import pandas as pd

    sys.path.insert(0, "/app/core")
    from scoring import artifact_suffix
    suf = artifact_suffix()

    df = pd.read_csv(f"{MANIFEST}/{SPLITS[split]}")
    df["text"] = df["text"].astype(str).str.lower()
    kws = sorted(df["text"].unique())
    clips = sorted(df["wav"].astype(str).unique())
    print(f"manifest: {len(df):,} rows, {len(kws):,} keywords, {len(clips):,} clips")

    # 2. embeddings present?
    have_emb = set()
    for f in sorted(glob.glob(f"{EMB_DIR}/{split}_*.npz")):
        z = np.load(f, allow_pickle=False)
        have_emb.update(str(n) for n in z["names"])
    missing_clips = [c for c in clips if c not in have_emb]
    print(f"embeddings: {len(have_emb):,} present, {len(missing_clips):,} missing")
    for c in missing_clips[:5]:
        p = os.path.join("/data/libriphrase/audio", c)
        print(f"    MISSING EMB {c}  exists_on_disk={os.path.exists(p)}")

    out = {"rows": len(df), "keywords": len(kws), "clips": len(clips),
           "missing_clips": len(missing_clips), "arms": {}}

    # 1 + 3. per-arm artifact coverage and resulting row loss
    roots = ([("run", RUN_ROOT)] if split == "500"
             else [("devclean", f"{RUN_ROOT}_sel_devclean"),
                   ("broadcast", f"{RUN_ROOT}_sel_broadcast")])
    for arm, root in roots:
        miss_kw = [k for k in kws
                   if not (os.path.exists(f"{root}/keywords/{k}_anchor{suf}.npz")
                           and os.path.exists(f"{root}/keywords/cohort_{k}{suf}.npz"))]
        ok_kw = set(kws) - set(miss_kw)
        keep = df[df["text"].isin(ok_kw) & df["wav"].astype(str).isin(have_emb)]
        print(f"\n{arm}: {len(miss_kw)} keywords lack artifacts, "
              f"scoreable rows {len(keep):,} of {len(df):,}")
        for k in miss_kw[:8]:
            a = os.path.exists(f"{root}/keywords/{k}_anchor{suf}.npz")
            c = os.path.exists(f"{root}/keywords/cohort_{k}{suf}.npz")
            n = int((df["text"] == k).sum())
            print(f"    {k!r:<34} anchor={a} cohort={c}  rows={n}")
        out["arms"][arm] = {"missing_keywords": len(miss_kw),
                            "scoreable_rows": int(len(keep)),
                            "examples": miss_kw[:20]}

    # do the two arms differ in which keywords they cover?
    a = set(out["arms"]["devclean"]["examples"])
    b = set(out["arms"]["broadcast"]["examples"])
    print(f"\nkeywords missing in devclean only : {sorted(a - b)[:10]}")
    print(f"keywords missing in broadcast only: {sorted(b - a)[:10]}")
    return out


@app.local_entrypoint()
def main(split: str = "sel"):
    import json
    print(json.dumps(diagnose.remote(split), indent=2)[:1500])
