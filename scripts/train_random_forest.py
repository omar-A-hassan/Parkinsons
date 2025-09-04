#!/usr/bin/env python3
"""Script to train Random Forest classifier."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parkinsons_voice.train.train_random_forest import main

if __name__ == "__main__":
    main()