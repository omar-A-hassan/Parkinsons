#!/usr/bin/env python3
"""Create dummy test data for CI pipeline."""

import numpy as np
import soundfile as sf
import os

def main():
    """Create dummy audio files for testing."""
    # Create dummy audio files for testing
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    for i in range(3):
        for class_dir in ["HC_AH", "PD_AH"]:
            waveform = 0.3 * np.sin(2 * np.pi * (440 + i * 100) * t)
            sf.write(f"test_data/{class_dir}/sample_{i}.wav", waveform, sample_rate)
    
    print("Created dummy test data files")

if __name__ == "__main__":
    main()