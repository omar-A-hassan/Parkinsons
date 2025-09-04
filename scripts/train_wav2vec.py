#!/usr/bin/env python3
"""Script to train Wav2Vec2 classifier."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parkinsons_voice.train.train_wav2vec import main

if __name__ == "__main__":
    main()