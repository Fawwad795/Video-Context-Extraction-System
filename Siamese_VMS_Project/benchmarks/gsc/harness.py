"""Speech Commands benchmark harness - app and enrollment entrypoint.

The shared image and per-keyword enrollment unit come from
`benchmarks/common/runtime.py`; staging and the trial table are in
`modal_gsc.py` beside this file.

    modal run harness.py::gsc_enroll
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal

from common.runtime import (COHORTS, RUN_ROOT, data, _drive, enroll,  # noqa: F401
                            image, runtime, write_results)

app = modal.App("vms-gsc")
app.include(runtime)

# The ten commands, fixed by TC-ResNet sec. 3.1 -> CMCD ref [24] ->
# PhonMatchNet dataset/google.py. See modal_gsc.py for the protocol.
GSC_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
GSC_ROOT = "/data/run_gsc"


@app.local_entrypoint()
def gsc_enroll(cohort: str = "devclean"):
    """CED prong: anchor + cohort + rivals for the ten Speech Commands words.

    Deliberately a SEPARATE project root. Several of the ten ("right", "down",
    "on", "no", ...) also occur as one-word LibriPhrase keywords, and those
    artifacts were built by t1 with need_rivals=False. Sharing /data/run would
    silently reuse a rival-less artifact and RAV would abstain without saying so.

    Cohort source stays LibriSpeech dev-clean, locked 2026-08-28 - outside Speech
    Commands, so scores are not normalised using the test set.
    """
    print(f"GSC: {len(GSC_WORDS)} keywords, cohort={cohort}, root={GSC_ROOT}")
    _drive("gsc_enroll", [{"keyword": k, "cohort": cohort, "need_rivals": True,
                           "root": GSC_ROOT} for k in GSC_WORDS])
