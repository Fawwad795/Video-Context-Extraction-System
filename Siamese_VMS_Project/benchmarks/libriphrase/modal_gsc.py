"""Stage Google Speech Commands V1 and build its trial table (the CED prong, P5).

Lives beside the LibriPhrase harness because it reuses it wholesale: the same
image, volume, enrollment path (`modal_app.enroll`) and the same two scoring
passes (`modal_score.embed_clips` / `score_shard`, via `--split gsc`). Only data
staging and the trial table are new, so there is no second copy of the scoring
pipeline to keep in sync.

THE PROTOCOL IS INHERITED, NOT INVENTED. Neither CED nor CMCD writes down how a
clip and a keyword become scored trials, but PhonMatchNet - same lineage, public
code - does, in `dataset/google.py`:

    target_list = ['yes','no','up','down','left','right','on','off','stop','go']
    testset_only=True -> read testing_list.txt, keep clips whose parent dir is a target
    for wav in wav_list:
        anchor_text = <parent dir>
        for comparison_text in target_list:        # all ten texts, every clip
            label = 1 if anchor_text == comparison_text else 0

So: the official test list, restricted to the ten words, each clip scored against
all ten texts - one positive and nine negatives apiece. Its `__main__` builds the
test set at batch_size 2048, matching the EER convention we already reproduce.

`src` carries the vocabulary control established 2026-09-04
(`Journal_Paper/experiments/gsc/mswc_vocab_check.json`): `unseen` for the four
words absent from MSWC entirely (no, up, on, go), `seen` for the six that were
training classes. The two groups are reported separately.

Run in order:
    modal run modal_gsc.py::fetch          # ~1.4 GB, cloud-to-cloud
    modal run modal_gsc.py::manifest
    modal run modal_app.py::gsc_enroll     # 10 keywords, anchor + cohort + rivals
    modal run modal_score.py::embed --split gsc
    modal run modal_score.py::score --split gsc --need-margin
    modal run modal_score.py::report_gsc
"""

import json
import os

import modal

from modal_app import RUN_ROOT, data, image

app = modal.App("vms-gsc")

GSC_URL = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
GSC_DIR = "/data/gsc"
GSC_AUDIO = f"{GSC_DIR}/audio"
MANIFEST = f"{RUN_ROOT}/manifests"

# TC-ResNet sec. 3.1 -> CMCD ref [24] -> PhonMatchNet dataset/google.py.
TARGETS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# Absent from MSWC entirely, so unreachable by any training stage.
UNSEEN = {"no", "up", "on", "go"}

# Published V1 statistics, used as the integrity check.
EXPECTED_TEST_LIST = 6835


@app.function(image=image, volumes={"/data": data}, timeout=5400)
def fetch():
    """Stream the canonical archive and extract only what the protocol reads.

    NOT the HuggingFace `google/speech_commands` v0.01 `test` split: that is a
    separate, smaller archive of 3,081 clips, while the official testing_list.txt
    holds 6,835. TC-ResNet - and so CMCD and CED - used the SHA-1 hashed split
    that testing_list.txt materialises.

    Only the ten target directories and the split lists are extracted, ~21k files
    rather than 64,727, because the protocol above never opens the other twenty
    words. A wider trial construction would need a re-fetch.
    """
    import tarfile
    import urllib.request

    os.makedirs(GSC_AUDIO, exist_ok=True)
    keep_dirs = tuple(f"{t}/" for t in TARGETS)
    keep_files = ("testing_list.txt", "validation_list.txt", "LICENSE", "README.md")

    n_wav, n_other = 0, 0
    req = urllib.request.Request(GSC_URL, headers={"User-Agent": "vms/1.0"})
    with urllib.request.urlopen(req) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                name = m.name[2:] if m.name.startswith("./") else m.name
                if name.startswith(keep_dirs):
                    m.name = name
                    tar.extract(m, GSC_AUDIO)
                    n_wav += 1
                elif name in keep_files:
                    m.name = name
                    tar.extract(m, GSC_AUDIO)
                    n_other += 1
    data.commit()

    per_word = {t: len(os.listdir(f"{GSC_AUDIO}/{t}")) for t in TARGETS
                if os.path.isdir(f"{GSC_AUDIO}/{t}")}
    with open(f"{GSC_AUDIO}/testing_list.txt") as f:
        test_list = [ln.strip() for ln in f if ln.strip()]
    out = {"wavs_extracted": n_wav, "aux_files": n_other,
           "per_word": per_word, "test_list_total": len(test_list),
           "test_list_total_expected": EXPECTED_TEST_LIST,
           "test_list_matches_published": len(test_list) == EXPECTED_TEST_LIST}
    print(json.dumps(out, indent=2))
    return out


@app.function(image=image, volumes={"/data": data}, timeout=1800)
def manifest():
    """Build the trial table exactly as PhonMatchNet's loader would enumerate it."""
    import pandas as pd

    with open(f"{GSC_AUDIO}/testing_list.txt") as f:
        test_list = [ln.strip() for ln in f if ln.strip()]

    clips = [p for p in test_list if p.split("/")[0] in set(TARGETS)]
    missing = [p for p in clips if not os.path.exists(f"{GSC_AUDIO}/{p}")]

    rows = []
    for rel in clips:
        anchor_text = rel.split("/")[0]
        for text in TARGETS:
            rows.append({"wav": rel, "text": text,
                         "label": int(anchor_text == text),
                         "type": "closed_10",
                         "src": "unseen" if text in UNSEEN else "seen"})
    df = pd.DataFrame(rows)

    os.makedirs(MANIFEST, exist_ok=True)
    df.to_csv(f"{MANIFEST}/samples_gsc.csv", index=False)
    data.commit()

    per_word = df[df["label"] == 1]["text"].value_counts().to_dict()
    checks = [
        {"check": "test-list total", "got": len(test_list),
         "want": EXPECTED_TEST_LIST, "ok": len(test_list) == EXPECTED_TEST_LIST},
        {"check": "every referenced clip resolves", "got": len(missing),
         "want": 0, "ok": not missing},
        {"check": "rows == clips x 10", "got": len(df),
         "want": len(clips) * 10, "ok": len(df) == len(clips) * 10},
        {"check": "positives == clips", "got": int(df["label"].sum()),
         "want": len(clips), "ok": int(df["label"].sum()) == len(clips)},
        {"check": "one positive text per clip", "got": int(df.groupby("wav")["label"].sum().max()),
         "want": 1, "ok": int(df.groupby("wav")["label"].sum().max()) == 1},
    ]
    out = {"clips_in_ten_words": len(clips), "trials": len(df),
           "positives": int(df["label"].sum()),
           "positives_per_word": per_word,
           "unseen_words": sorted(UNSEEN),
           "missing_sample": missing[:5],
           "reconciliation": checks,
           "all_checks_pass": all(c["ok"] for c in checks)}
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def stage():
    """fetch + manifest in one go."""
    print(json.dumps(fetch.remote(), indent=2))
    print(json.dumps(manifest.remote(), indent=2))
