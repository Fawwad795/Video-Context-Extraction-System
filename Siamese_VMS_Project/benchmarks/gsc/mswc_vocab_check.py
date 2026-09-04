"""Did the attentive head train on the ten Speech Commands words?

Reproduces, exactly, the two selection rules that decided the head's training
vocabulary. Both consume the same object: a Counter over the `keyword` column of
MSWC English train.

  training/tts_bank.py (seed 42, --bank-words 1500, --eval-words 150)
      candidates = [w for w, c in counts.most_common()
                    if c >= 8 and w.isascii() and w.isalpha() and len(w) >= 3]
      pool       = candidates[:1650]
      order      = np.random.default_rng(42).permutation(len(pool))
      eval_words = [pool[i] for i in order[:150]]          # held out, never trained

  training/dataset_v3.py::select_classes (--n-classes 1000, --min-samples 8)
      classes    = [w for w in candidates if w not in eval_words][:1000]

Counts come from the dataset's own splits metadata - `data/splits/en/splits.tar.gz`
on the HF repo, 365 MB - streamed and discarded, so no audio is downloaded. The
`ml_spoken_words.py` loading script builds its keyword map from exactly this file
(its `_SPLITS_URL`), so the words and counts are the same ones the training run saw.

Two caveats, both recorded in the output:
  * The loader yields one example per audio file present in the archives and looks
    its keyword up in this CSV. A CSV row whose audio is missing would be counted
    here but not there. Any difference is small and matters only for a word sitting
    on the 1000-class boundary, which the output flags.
  * Counter.most_common() breaks count ties by first-encountered order. Here that is
    CSV order; in training it was audio-archive order. Again only a boundary concern.

Run:
    modal run mswc_vocab_check.py
"""

import json

import modal

app = modal.App("vms-mswc-vocab")
data = modal.Volume.from_name("vms-data", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy<2"))

SPLITS_URL = ("https://huggingface.co/datasets/MLCommons/ml_spoken_words/"
              "resolve/main/data/splits/en/splits.tar.gz")

# The ten keywords, fixed by TC-ResNet sec. 3.1 -> CMCD -> CED.
GSC_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# tts_bank.py / dataset_v3.py defaults, read from their argparse blocks.
MIN_SAMPLES = 8
MIN_LEN = 3
BANK_WORDS = 1500
EVAL_WORDS = 150
N_CLASSES = 1000
SEED = 42


@app.function(image=image, volumes={"/data": data}, timeout=1800)
def check():
    import csv

    import tarfile
    import urllib.request
    from collections import Counter

    import numpy as np

    counts = Counter()
    rows = 0
    req = urllib.request.Request(SPLITS_URL, headers={"User-Agent": "vms/1.0"})
    with urllib.request.urlopen(req) as resp:
        # r|gz = streaming read, never materialises the 365 MB archive on disk.
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not member.name.endswith("train.csv"):
                    continue
                f = tar.extractfile(member)
                # Decode line by line rather than wrapping in TextIOWrapper: a
                # streamed tar member is not seekable, which TextIOWrapper probes.
                # This is what ml_spoken_words.py itself does.
                reader = csv.reader(line.decode("utf-8") for line in f)
                for i, row in enumerate(reader):
                    if i == 0 or len(row) < 2:
                        continue          # header, per the loading script
                    rows += 1
                    if row[1]:
                        counts[row[1]] += 1
                break                     # train.csv is all we need

    candidates = [w for w, c in counts.most_common()
                  if c >= MIN_SAMPLES and w.isascii() and w.isalpha()
                  and len(w) >= MIN_LEN]
    rank = {w: i for i, w in enumerate(candidates)}

    pool = candidates[:BANK_WORDS + EVAL_WORDS]
    order = np.random.default_rng(SEED).permutation(len(pool))
    eval_words = [pool[i] for i in order[:EVAL_WORDS]]
    eval_set = set(eval_words)

    classes = [w for w in candidates if w not in eval_set][:N_CLASSES]
    class_set = set(classes)
    # The candidate rank of the last word that made the cut - the boundary a
    # near-miss word would have to cross for this verdict to be fragile.
    cutoff_rank = rank[classes[-1]]

    out = {
        "source": SPLITS_URL,
        "csv_rows": rows,
        "distinct_keywords": len(counts),
        "candidates_after_filter": len(candidates),
        "constants": {"min_samples": MIN_SAMPLES, "min_len": MIN_LEN,
                      "bank_words": BANK_WORDS, "eval_words": EVAL_WORDS,
                      "n_classes": N_CLASSES, "seed": SEED},
        "cutoff_candidate_rank": int(cutoff_rank),
        "words": {},
        "caveats": [
            "counts are from splits/en/train.csv; the loader yields per audio file "
            "present in the archives, so a row with missing audio is counted here "
            "and not there",
            "most_common() ties break on first-encountered order: CSV order here, "
            "audio-archive order in training",
        ],
    }

    for w in GSC_WORDS:
        c = counts.get(w, 0)
        excluded = []
        if len(w) < MIN_LEN:
            excluded.append(f"len<{MIN_LEN}")
        if c < MIN_SAMPLES:
            excluded.append(f"count<{MIN_SAMPLES}")
        if not (w.isascii() and w.isalpha()):
            excluded.append("non-alpha")
        r = rank.get(w)
        out["words"][w] = {
            "mswc_train_count": c,
            "candidate_rank": None if r is None else int(r),
            "excluded_by_filter": excluded or None,
            "held_out_as_eval_word": w in eval_set,
            "TRAINED_ON": w in class_set,
            "margin_to_cutoff": None if r is None else int(cutoff_rank - r),
        }

    import os
    os.makedirs("/data/gsc", exist_ok=True)
    with open("/data/gsc/mswc_vocab_check.json", "w") as f:
        json.dump(out, f, indent=2)
    data.commit()

    print("===JSON===")
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def main():
    check.remote()
