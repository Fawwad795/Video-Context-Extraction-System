"""Stage the LibriPhrase benchmark data onto a Modal Volume.

Everything large is fetched *inside* Modal (cloud-to-cloud) rather than uploaded
from a laptop: the eval audio is 2.34 GiB and LibriSpeech dev-clean is ~340 MiB.
Only small local assets (code, checkpoint, rival bank) travel from the repo, and
those ride in the image - see modal_app.py.

Run in order:
    modal run modal_fetch.py::fetch_libriphrase
    modal run modal_fetch.py::fetch_devclean
    modal run modal_fetch.py::verify

The broadcast cohort source is OPTIONAL and no longer used: C4 was locked to
LibriSpeech dev-clean on 2026-08-28, so nothing needs uploading from the repo.
Only if you want to reproduce the C4 A/B:
    modal volume put vms-data <repo>/audios/Chunkset_F /cohort_src/broadcast
"""

import modal

app = modal.App("vms-fetch")
data = modal.Volume.from_name("vms-data", create_if_missing=True)

img = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "libsndfile1")
    .pip_install("huggingface_hub", "soundfile", "numpy<2", "pandas", "tqdm")
)

HF_REPO = "charsiu/libriphrase"
CSVS = [f"libriphrase_diffspk_all_{n}word.csv" for n in (1, 2, 3, 4)]
LP_DIR = "/data/libriphrase"
COHORT_DIR = "/data/cohort_src"

# LibriSpeech dev-clean is only the *cohort* source - it is a different split
# from the eval audio (train-other-500), so it cannot leak the test set.
DEVCLEAN_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"

# Size the cohort SOURCE deliberately. core/scoring.py's sample_stream_windows()
# decodes EVERY eligible chunk into memory before drawing its 50 windows, so the
# cost of cohort_builder scales with the source pool, not with the 50 windows it
# needs. A 300-chunk x 7 s pool made cohort_builder take 155 s on Modal against
# ~27 s locally on a 20 x 5 s pool - which would have cost 389 CPU-hours instead
# of 25 across 9,030 keywords. 48 chunks, one per speaker, capped at 6 s gives
# comparable audio volume to the broadcast arm and better speaker diversity than
# 300 utterances drawn from the same 40 speakers.
N_COHORT_CHUNKS = 48
MIN_CHUNK_SECONDS = 3.0
MAX_CHUNK_SECONDS = 6.0


@app.function(image=img, volumes={"/data": data}, timeout=7200)
def fetch_libriphrase():
    """CSVs + the 2.34 GiB eval audio, straight from HuggingFace."""
    import os
    import zipfile

    from huggingface_hub import hf_hub_download

    os.makedirs(LP_DIR, exist_ok=True)
    for f in CSVS:
        p = hf_hub_download(HF_REPO, f, repo_type="dataset", local_dir=LP_DIR)
        print(f"csv  {f}  {os.path.getsize(p) / 1e6:.1f} MB")

    z = hf_hub_download(HF_REPO, "LibriPhrase_evalset.zip", repo_type="dataset",
                        local_dir=LP_DIR)
    print(f"zip  {os.path.getsize(z) / 1e9:.2f} GB - extracting ...")
    audio_root = os.path.join(LP_DIR, "audio")
    os.makedirs(audio_root, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(audio_root)
    os.remove(z)                      # the extracted tree is what we need
    data.commit()
    print("extracted to", audio_root)


@app.function(image=img, volumes={"/data": data}, timeout=3600)
def fetch_devclean():
    """LibriSpeech dev-clean -> a cohort source in the layout cohort_builder wants.

    core/scoring.py's list_chunk_audios() globs '*.wav' and sorts with
    int(basename.split('_')[1].split('.')[0]), so files MUST be named
    <prefix>_<int>.wav. It also needs a transcripts.txt so the keyword-free
    guard can run - LibriSpeech ships exact transcripts, which makes that guard
    stricter here than the Whisper-derived one used on broadcast audio.
    """
    import glob
    import os
    import subprocess
    import tarfile

    import soundfile as sf

    work = "/tmp/ls"
    os.makedirs(work, exist_ok=True)
    tgz = os.path.join(work, "dev-clean.tar.gz")
    subprocess.run(["curl", "-fL", "-o", tgz, DEVCLEAN_URL], check=True)
    with tarfile.open(tgz) as t:
        t.extractall(work)

    # utt_id -> transcript, from the per-chapter .trans.txt files
    text = {}
    for tf in glob.glob(f"{work}/LibriSpeech/dev-clean/*/*/*.trans.txt"):
        for line in open(tf, encoding="utf-8"):
            uid, _, tx = line.strip().partition(" ")
            text[uid] = tx.lower()

    out = os.path.join(COHORT_DIR, "devclean")
    os.makedirs(out, exist_ok=True)
    for stale in glob.glob(os.path.join(out, "*")):
        os.remove(stale)          # rewriting with fewer chunks must not leave
                                  # dev_48..dev_299 behind from an earlier run

    # One utterance per speaker, round-robin, so a small pool still spans the
    # full 40-speaker set rather than a handful of chapters.
    by_spk = {}
    for f in sorted(glob.glob(f"{work}/LibriSpeech/dev-clean/*/*/*.flac")):
        by_spk.setdefault(os.path.basename(f).split("-")[0], []).append(f)

    lines, n, rnd, used_spk = [], 0, 0, set()
    while n < N_COHORT_CHUNKS:
        progressed = False
        for spk in sorted(by_spk):
            if n >= N_COHORT_CHUNKS:
                break
            if rnd >= len(by_spk[spk]):
                continue
            f = by_spk[spk][rnd]
            progressed = True
            uid = os.path.basename(f)[:-5]
            info = sf.info(f)
            if info.duration < MIN_CHUNK_SECONDS or uid not in text:
                continue
            y, sr = sf.read(f, dtype="float32")
            assert sr == 16000, f"unexpected sample rate {sr}"
            y = y[:int(MAX_CHUNK_SECONDS * sr)]      # cap: see N_COHORT_CHUNKS note
            name = f"dev_{n}.wav"
            sf.write(os.path.join(out, name), y, sr)
            lines.append(f"[{name}]\n{text[uid]}\n")
            used_spk.add(spk)
            n += 1
        rnd += 1
        if not progressed:
            break
    print(f"{n} chunks spanning {len(used_spk)} of {len(by_spk)} dev-clean speakers")

    with open(os.path.join(out, "transcripts.txt"), "w", encoding="utf-8") as fh:
        fh.write("Cohort source - LibriSpeech dev-clean (exact transcripts)\n\n")
        fh.write("\n".join(lines))
    data.commit()
    print(f"wrote {n} cohort chunks + transcripts.txt to {out}")


@app.function(image=img, volumes={"/data": data}, timeout=1800)
def verify():
    """Repeat the local integrity check on the volume before spending credits."""
    import glob
    import os

    import pandas as pd

    audio_root = os.path.join(LP_DIR, "audio")
    wavs = glob.glob(os.path.join(audio_root, "**", "*.wav"), recursive=True)
    rel = {os.path.relpath(w, audio_root).replace(os.sep, "/") for w in wavs}
    print(f"wav files on volume: {len(rel):,}")

    total = hits = 0
    for f in CSVS:
        df = pd.read_csv(os.path.join(LP_DIR, f))
        refs = pd.concat([df["anchor"], df["comparison"]]).astype(str).unique()
        h = sum(1 for r in refs if r in rel)
        total += len(refs)
        hits += h
        print(f"  {f:<38} {len(refs):>7,} refs   resolved {h:>7,}")
    print(f"TOTAL {total:,} refs, {hits:,} resolved ({100 * hits / total:.1f}%)")

    ok_audio = (len(rel) == 136461 and hits == total)
    dev = os.path.join(COHORT_DIR, "devclean")
    n_dev = len(glob.glob(os.path.join(dev, "*.wav")))
    ok_dev = n_dev > 0 and os.path.exists(os.path.join(dev, "transcripts.txt"))
    bc = os.path.join(COHORT_DIR, "broadcast")
    n_bc = len(glob.glob(os.path.join(bc, "*.wav")))
    ok_bc = n_bc > 0 and os.path.exists(os.path.join(bc, "transcripts.txt"))

    print(f"\ncohort src devclean : {n_dev} wavs  "
          f"transcripts={'y' if ok_dev else 'N'}   <- the locked C4 source")
    print(f"cohort src broadcast: {n_bc} wavs  transcripts={'y' if ok_bc else 'N'}"
          f"   (OPTIONAL - C4 locked to dev-clean 2026-08-28; only needed to redo the A/B)")
    # broadcast is deliberately excluded from the verdict: requiring it would
    # report NOT READY on a correctly provisioned account.
    print(f"\nVERDICT: {'READY' if (ok_audio and ok_dev) else 'NOT READY'}")
    if not ok_audio:
        print("  expected 136,461 wavs and 100% CSV path resolution")
