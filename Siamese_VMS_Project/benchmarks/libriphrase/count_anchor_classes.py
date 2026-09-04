"""How many distinct anchor classes does LibriPhrase offer?

Section VI wants to state the reported set's scope as "500 of N classes".
The N carried in the paper plan was 4,003 and no artifact anywhere in the
repo backs it, so this recomputes it from the published CSVs on the volume,
using exactly the pool `harness.py::make_manifests` samples the 500 from:
the unique `anchor_text` values over the diffspk easy/hard negative rows of
the 1-, 2-, 3- and 4-word files.

Run:
    modal run count_anchor_classes.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal

from common.runtime import data, image, runtime  # noqa: F401

app = modal.App("vms-lp-classcount")
app.include(runtime)

LP_DIR = "/data/libriphrase"


@app.function(image=image, volumes={"/data": data}, timeout=1800)
def count():
    import pandas as pd

    dfs = [pd.read_csv(f"{LP_DIR}/libriphrase_diffspk_all_{n}word.csv")
           for n in (1, 2, 3, 4)]
    allf = pd.concat(dfs, ignore_index=True)
    neg = allf[allf["type"].isin(["diffspk_easyneg", "diffspk_hardneg"])]
    anchors = sorted(neg["anchor_text"].astype(str).unique())

    per_len = {}
    for n, df in zip((1, 2, 3, 4), dfs):
        d = df[df["type"].isin(["diffspk_easyneg", "diffspk_hardneg"])]
        per_len[n] = int(d["anchor_text"].astype(str).nunique())

    out = {"total_anchor_classes": len(anchors),
           "unique_by_word_count_file": per_len,
           "reported_sample": 500,
           "holdout_sample": 40,
           "source": "libriphrase_diffspk_all_{1,2,3,4}word.csv, "
                     "diffspk_easyneg + diffspk_hardneg rows",
           "note": "same pool harness.py::make_manifests draws the 500 from"}

    os.makedirs("/data/libriphrase", exist_ok=True)
    with open("/data/libriphrase/anchor_class_count.json", "w") as f:
        json.dump(out, f, indent=2)
    data.commit()
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def main():
    count.remote()
