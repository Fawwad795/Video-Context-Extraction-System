"""Siamese-network keyword-spotting extension for the Video Monitoring System (VMS).

A learned audio-similarity matcher that replaces the hand-crafted cross-correlation
detector in the original VMS. See ../Extension/Siamese_Network_Extension_Ideation.md
for the full design and roadmap (P0-P5). This package implements that roadmap.
"""

__all__ = ["config", "audio", "datasets", "augment", "tts_bridge"]
