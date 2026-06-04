"""Stream 1 Siamese detector (parallel to Stream1_corelation_updated_v2.py).

Launched by the GUI when VMS_DETECTOR=siamese. Delegates to the shared detection loop.
"""
from siamese_vms_detect import run

if __name__ == "__main__":
    run("Stream1")
