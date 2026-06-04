"""Stream 2 Siamese detector (parallel to Stream2_corelation_updated_v2.py).

Launched by the GUI when VMS_DETECTOR=siamese. Delegates to the shared detection loop.
"""
from siamese_vms_detect import run

if __name__ == "__main__":
    run("Stream2")
