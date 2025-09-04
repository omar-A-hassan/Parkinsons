#!/usr/bin/env python3
"""Script to preprocess audio data for Wav2Vec2 training."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parkinsons_voice.data.preprocessing import main

if __name__ == "__main__":
    main()