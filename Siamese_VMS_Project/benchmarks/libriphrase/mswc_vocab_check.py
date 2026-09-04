"""Does the LibriPhrase test vocabulary overlap the head's MSWC training classes?

The sibling of `benchmarks/gsc/mswc_vocab_check.py`, asking the same question of a
much larger and messier vocabulary. Speech Commands has ten fixed single words, so
that check is a ten-row table. LibriPhrase's 500-class run scores 9,030 distinct
texts of one to four words built from 6,913 distinct tokens, and MSWC classes are
single words, so the overlap has to be measured token by token and then weighted
back onto the 49,981 scored trials. A distinct-keyword rate is not the number a
reviewer wants; a trial-weighted rate is.

WHAT IS RECONSTRUCTED, AND WHY
The trained head's checkpoint (`checkpoints/siamese_v3_best.pth`) stores weights,
`config`, `layer`, `backbone`, `epoch` and `auc` - and no class list. The list was
never written down. So this reproduces the two selection rules that produced it,
both consuming one object: a Counter over the `keyword` column of MSWC English
train. Both ran on defaults (`deploy_phase2.ps1`, `deploy_phase3.ps1` pass none).

  training/tts_bank.py (seed 42, --bank-words 1500, --eval-words 150)
      candidates = [w for w, c in counts.most_common()
                    if c >= 8 and w.isascii() and w.isalpha() and len(w) >= 3]
      pool       = candidates[:1650]
      order      = np.random.default_rng(42).permutation(len(pool))
      eval_words = [pool[i] for i in order[:150]]      # TTS'd, never trained on
      bank_words = [pool[i] for i in order[150:]]      # 1,500 synthesized

  training/dataset_v3.py::select_classes (--n-classes 1000, --min-samples 8)
      classes    = [w for w in candidates if w not in eval_set][:1000]

THE ONE THING THAT CANNOT BE RECONSTRUCTED
`dataset_v3.build_cache` does not pass all 150 eval words to `select_classes`. It
passes only those that survived TTS synthesis:

      eval_words = [w for w in manifest["eval_words"]
                    if w in class_to_indices and w in manifest["bank"]]

An eval word whose five clips all tripped the degenerate-synthesis guard is absent
from the bank, is therefore not excluded, and may sit inside the top 1,000. That
bank manifest lived on the rented L4 and is gone. So the true class set is bounded,
not known, and this harness reports both ends:

    strict   - all 150 eval words excluded (the documented code path)
    liberal  - none excluded (every eval word failed synthesis)

The symmetric difference is the ambiguity band, and every verdict that differs
between the two ends is listed by name in `ambiguous_tokens`. Headline numbers use
strict; liberal is carried beside them so the band is visible rather than implied.

STRATA
Each text is placed in one stratum against the class set. A token is *eligible* if
it could have been an MSWC class at all (`isascii`, `isalpha`, `len >= 3`); "a",
"of" and "won't" are structurally ineligible, which is not the same as unseen.

    ineligible  no eligible token at all            e.g. "a as"
    unseen      eligible tokens, none trained
    partial     some but not all eligible tokens trained
    full        every eligible token trained

`mswc_overlap_strata.csv` carries one row per distinct text, so an EER-by-stratum
re-grade joins onto the stored scores on `text` with no recompute.

SELF-CHECK
The candidate reconstruction is asserted against the four values the Speech
Commands run of this same rule already committed to
(`Journal_Paper/experiments/gsc/mswc_vocab_check.json`). If the MSWC splits file
or the filter ever moves, this fails loudly rather than quietly reporting a
different vocabulary from the one behind the published GSC numbers.

Run:
    modal run mswc_vocab_check.py
"""

import json

import modal

app = modal.App("vms-lp-mswc-vocab")
data = modal.Volume.from_name("vms-data", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy<2"))

SPLITS_URL = ("https://huggingface.co/datasets/MLCommons/ml_spoken_words/"
              "resolve/main/data/splits/en/splits.tar.gz")

# Mirrors RUN_ROOT in benchmarks/common/runtime.py. Not imported: runtime.py
# defines the ~8 GB torch image, and this job needs numpy and a CSV reader.
MANIFEST_ROOT = "/data/run/manifests"
OUT_DIR = "/data/libriphrase"

# tts_bank.py / dataset_v3.py argparse defaults; both deploy scripts pass none.
MIN_SAMPLES = 8
MIN_LEN = 3
BANK_WORDS = 1500
EVAL_WORDS = 150
N_CLASSES = 1000
SEED = 42

# Committed by the Speech Commands run of the same reconstruction, 2026-09-04.
EXPECT = {
    "csv_rows": 5266726,
    "distinct_keywords": 38150,
    "candidates_after_filter": 24003,
    "cutoff_candidate_rank": 1094,
}


def _eligible(tok):
    """Could this token have been an MSWC class at all, before any count filter?"""
    return tok.isascii() and tok.isalpha() and len(tok) >= MIN_LEN


def _stratum(text, classes):
    """Place one text, and return the per-token bookkeeping behind the verdict."""
    toks = text.split()
    elig = [t for t in toks if _eligible(t)]
    hit = [t for t in elig if t in classes]
    if not elig:
        s = "ineligible"
    elif len(hit) == len(elig):
        s = "full"
    elif hit:
        s = "partial"
    else:
        s = "unseen"
    return s, len(toks), len(elig), len(hit)


@app.function(image=image, volumes={"/data": data}, timeout=1800)
def check():
    import csv
    import os
    import tarfile
    import urllib.request
    from collections import Counter, defaultdict

    import numpy as np

    # ---- 1. MSWC keyword counts, streamed and discarded (no audio downloaded).
    counts = Counter()
    rows = 0
    req = urllib.request.Request(SPLITS_URL, headers={"User-Agent": "vms/1.0"})
    with urllib.request.urlopen(req) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not member.name.endswith("train.csv"):
                    continue
                f = tar.extractfile(member)
                # Line-by-line decode: a streamed tar member is not seekable, and
                # TextIOWrapper probes for that. This is what ml_spoken_words.py does.
                reader = csv.reader(line.decode("utf-8") for line in f)
                for i, row in enumerate(reader):
                    if i == 0 or len(row) < 2:
                        continue
                    rows += 1
                    if row[1]:
                        counts[row[1]] += 1
                break

    candidates = [w for w, c in counts.most_common()
                  if c >= MIN_SAMPLES and _eligible(w)]
    rank = {w: i for i, w in enumerate(candidates)}

    pool = candidates[:BANK_WORDS + EVAL_WORDS]
    order = np.random.default_rng(SEED).permutation(len(pool))
    eval_words = [pool[i] for i in order[:EVAL_WORDS]]
    eval_set = set(eval_words)
    bank_set = set(pool) - eval_set

    classes_strict = [w for w in candidates if w not in eval_set][:N_CLASSES]
    classes_liberal = candidates[:N_CLASSES]
    strict, liberal = set(classes_strict), set(classes_liberal)
    ambiguous = sorted(strict ^ liberal, key=lambda w: rank[w])
    cutoff_rank = rank[classes_strict[-1]]

    for k, want in EXPECT.items():
        got = {"csv_rows": rows, "distinct_keywords": len(counts),
               "candidates_after_filter": len(candidates),
               "cutoff_candidate_rank": int(cutoff_rank)}[k]
        assert got == want, (
            f"reconstruction drift: {k} is {got}, the Speech Commands run "
            f"committed to {want}. The published GSC vocabulary verdict and this "
            f"one would no longer describe the same training class set.")

    # ---- 2. LibriPhrase vocabularies and the scored trial table.
    with open(f"{MANIFEST_ROOT}/manifest.json") as f:
        man = json.load(f)

    with open(f"{MANIFEST_ROOT}/samples_500.csv") as f:
        trials = list(csv.DictReader(f))

    easy_kw = sorted({r["text"] for r in trials if r["type"] == "diffspk_easyneg"})
    hard_kw = sorted({r["text"] for r in trials if r["type"] == "diffspk_hardneg"})
    assert set(hard_kw) == set(man["t2_keywords"]), "hard keyword set disagrees"
    assert set(easy_kw) | set(hard_kw) == set(man["t1_keywords"]), "t1 disagrees"

    vocabs = {
        "anchor_classes_500": man["anchor_classes_500"],   # the enrolled positives
        "t1_keywords_9030": man["t1_keywords"],            # every text scored
        "lp_easy_keywords_7643": easy_kw,
        "lp_hard_keywords_1969": hard_kw,
        "selection_keywords_809": man["selection_keywords"],
    }

    # ---- 3. Strata, distinct-text weighted and trial weighted.
    def summarise(texts, classes):
        by = Counter()
        for t in texts:
            by[_stratum(t, classes)[0]] += 1
        n = len(texts)
        return {"n": n,
                "counts": {k: by.get(k, 0)
                           for k in ("full", "partial", "unseen", "ineligible")},
                "rate": {k: round(by.get(k, 0) / n, 4)
                         for k in ("full", "partial", "unseen", "ineligible")}}

    out_vocabs = {}
    for name, words in vocabs.items():
        out_vocabs[name] = {"strict": summarise(words, strict),
                            "liberal": summarise(words, liberal)}

    # Per-text detail, computed once against the strict set and reused for trials.
    detail = {}
    for t in vocabs["t1_keywords_9030"]:
        s, n_tok, n_elig, n_hit = _stratum(t, strict)
        sl = _stratum(t, liberal)[0]
        toks = t.split()
        detail[t] = {
            "stratum_strict": s, "stratum_liberal": sl,
            "n_tokens": n_tok, "n_eligible": n_elig, "n_trained": n_hit,
            "n_eval_held_out": sum(1 for x in toks if x in eval_set),
            "n_bank_not_trained": sum(1 for x in toks
                                      if x in bank_set and x not in strict),
        }

    trial_by = defaultdict(Counter)
    for r in trials:
        d = detail[r["text"]]
        split = "lp_easy" if r["type"] == "diffspk_easyneg" else "lp_hard"
        trial_by[split][d["stratum_strict"]] += 1
        trial_by["all"][d["stratum_strict"]] += 1
        trial_by[f"{split}_label{r['label']}"][d["stratum_strict"]] += 1

    trial_out = {}
    for k, c in trial_by.items():
        n = sum(c.values())
        trial_out[k] = {"n": n,
                        "counts": {s: c.get(s, 0) for s in
                                   ("full", "partial", "unseen", "ineligible")},
                        "rate": {s: round(c.get(s, 0) / n, 4) for s in
                                 ("full", "partial", "unseen", "ineligible")}}

    # ---- 4. Token-level view: the 6,913 distinct tokens, and their exposure.
    tokens = sorted({x for t in vocabs["t1_keywords_9030"] for x in t.split()})
    tok_state = Counter()
    for x in tokens:
        if not _eligible(x):
            tok_state["ineligible"] += 1
        elif x in strict:
            tok_state["trained"] += 1
        elif x in eval_set:
            tok_state["eval_held_out"] += 1
        elif x in bank_set:
            tok_state["bank_only"] += 1
        else:
            tok_state["unseen"] += 1

    out = {
        "question": ("does the LibriPhrase test vocabulary overlap the classes the "
                     "attentive head was trained on?"),
        "source": SPLITS_URL,
        "manifest_root": MANIFEST_ROOT,
        "csv_rows": rows,
        "distinct_keywords": len(counts),
        "candidates_after_filter": len(candidates),
        "constants": {"min_samples": MIN_SAMPLES, "min_len": MIN_LEN,
                      "bank_words": BANK_WORDS, "eval_words": EVAL_WORDS,
                      "n_classes": N_CLASSES, "seed": SEED},
        "cutoff_candidate_rank": int(cutoff_rank),
        "class_set_bounds": {
            "strict_n": len(strict), "liberal_n": len(liberal),
            "symmetric_difference": len(ambiguous),
            "eval_words_inside_cutoff": sum(1 for w in eval_words
                                            if rank[w] <= cutoff_rank),
            "ambiguous_tokens": ambiguous,
        },
        "tokens": {"distinct": len(tokens), "by_exposure": dict(tok_state)},
        "vocabularies": out_vocabs,
        "trials": trial_out,
        "caveats": [
            "the trained class list was never written to the checkpoint; it is "
            "reconstructed from tts_bank.py and dataset_v3.py::select_classes",
            "dataset_v3 excludes only eval words present in the TTS bank, and that "
            "bank manifest is gone; strict/liberal bound the resulting class set",
            "counts are from splits/en/train.csv; the loader yields per audio file "
            "present in the archives, so a row with missing audio is counted here "
            "and not there",
            "most_common() ties break on first-encountered order: CSV order here, "
            "audio-archive order in training",
            "overlap is exposure, not leakage: MSWC clips are isolated single words "
            "and LibriPhrase clips are LibriSpeech utterances, with no shared audio",
        ],
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/mswc_vocab_check.json", "w") as f:
        json.dump(out, f, indent=2)

    with open(f"{OUT_DIR}/mswc_overlap_strata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "stratum_strict", "stratum_liberal", "n_tokens",
                    "n_eligible", "n_trained", "n_eval_held_out",
                    "n_bank_not_trained", "in_lp_easy", "in_lp_hard",
                    "is_anchor_class"])
        easy_s, hard_s = set(easy_kw), set(hard_kw)
        anch = set(man["anchor_classes_500"])
        for t in vocabs["t1_keywords_9030"]:
            d = detail[t]
            w.writerow([t, d["stratum_strict"], d["stratum_liberal"],
                        d["n_tokens"], d["n_eligible"], d["n_trained"],
                        d["n_eval_held_out"], d["n_bank_not_trained"],
                        int(t in easy_s), int(t in hard_s), int(t in anch)])
    data.commit()

    def row(label, d):
        c, r = d["counts"], d["rate"]
        return (f"  {label:<26} n={d['n']:>6}  full {c['full']:>6} ({r['full']:.1%})"
                f"  partial {c['partial']:>6} ({r['partial']:.1%})"
                f"  unseen {c['unseen']:>6} ({r['unseen']:.1%})"
                f"  ineligible {c['ineligible']:>5}")

    print(f"\ncandidates {len(candidates)}, cutoff rank {cutoff_rank}, "
          f"class set ambiguous over {len(ambiguous)} words "
          f"({out['class_set_bounds']['eval_words_inside_cutoff']} eval words "
          f"inside the cutoff)")
    print(f"tokens: {dict(tok_state)}")
    print("\nDISTINCT TEXTS (strict class set)")
    for name in vocabs:
        print(row(name, out_vocabs[name]["strict"]))
    print("\nSCORED TRIALS (strict class set)")
    for k in ("all", "lp_easy", "lp_hard", "lp_easy_label1", "lp_easy_label0",
              "lp_hard_label1", "lp_hard_label0"):
        if k in trial_out:
            print(row(k, trial_out[k]))
    print(f"\nwrote {OUT_DIR}/mswc_vocab_check.json and mswc_overlap_strata.csv")
    return out


@app.local_entrypoint()
def main():
    check.remote()
