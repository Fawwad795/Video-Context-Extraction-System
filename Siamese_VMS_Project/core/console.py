"""Shared terminal styling for the pipeline scripts.

Every phase prints through these helpers so a full pipeline run reads as a
sequence of uniform blocks:

    ============================================================
     PHASE NAME                                keyword: western
    ============================================================
       key                  value
     > progress line
     + success line
     ! warning line
    ------------------------------------------------------------
     DONE in 42.6s  ->  keywords/western_anchor_wavlm10ft.npz

Colors are used only on a live terminal (and never when NO_COLOR is set),
so logs captured to files stay plain text.
"""

import os
import sys
import time

WIDTH = 62

# Windows consoles may default to a legacy codepage; the pipeline prints
# only ASCII, but transcripts/keywords can carry anything - never crash on
# encoding.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass


def _color_enabled():
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        os.system("")  # enables ANSI escape processing on legacy consoles
    return True


_COLOR = _color_enabled()

_BOLD, _DIM, _GREEN, _YELLOW, _RED, _CYAN = "1", "2", "32", "33", "31", "36"


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def banner(title, subtitle=""):
    """Phase header. `subtitle` is right-aligned (e.g. 'keyword: western')."""
    line = f" {title}"
    if subtitle:
        pad = max(2, WIDTH - len(line) - len(subtitle) - 1)
        line += " " * pad + subtitle
    print()
    print(_c(_CYAN, "=" * WIDTH))
    print(_c(_BOLD, line))
    print(_c(_CYAN, "=" * WIDTH))


def kv(label, value):
    """Aligned 'label  value' detail line."""
    print(f"   {label:<24} {value}")


def step(msg):
    """A progress line: something is about to happen / is happening."""
    print(_c(_DIM, f" > {msg}"))


def item(msg):
    """A plain indented line (per-file / per-voice progress)."""
    print(f"   {msg}")


def ok(msg):
    print(_c(_GREEN, f" + {msg}"))


def warn(msg):
    print(_c(_YELLOW, f" ! {msg}"))


def fail(msg):
    print(_c(_RED, f" x {msg}"))


def rule():
    print(_c(_CYAN, "-" * WIDTH))


def fmt_secs(seconds):
    return f"{seconds / 60:.1f} min" if seconds >= 90 else f"{seconds:.1f}s"


def done(msg, t0=None):
    """Phase footer; pass t0=time.perf_counter() from the start for timing."""
    rule()
    took = f" in {fmt_secs(time.perf_counter() - t0)}" if t0 is not None else ""
    print(_c(_GREEN, _c(_BOLD, f" DONE{took}")) + (f"  ->  {msg}" if msg else ""))
    print()
