"""Tests for audio preprocessing functionality."""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import librosa
import soundfile as sf

from src.parkinsons_voice.data.preprocessing import AudioPreprocessor


@pytest.fixture
def sample_audio_data():
    """Create sample audio data for testing."""
    # Generate synthetic audio (1 second at 16kHz)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a simple sine wave
    frequency = 440.0  # A4 note
    waveform = 0.3 * np.sin(2 * np.pi * frequency * t)
    return waveform, sample_rate


@pytest.fixture
def temp_audio_file(sample_audio_data):
    """Create temporary audio file for testing."""
    waveform, sample_rate = sample_audio_data
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, waveform, sample_rate)
        yield tmp_file.name
    # Cleanup
    Path(tmp_file.name).unlink(missing_ok=True)


@pytest.fixture
def audio_preprocessor():
    """Create AudioPreprocessor instance for testing."""
    return AudioPreprocessor(
        processor_name="facebook/wav2vec2-base-960h",
        target_sample_rate=16000,
        max_duration=5.0,
        normalize_audio=True,
    )


class TestAudioPreprocessor:
    """Test cases for AudioPreprocessor class."""
    
    def test_init(self):
        """Test AudioPreprocessor initialization."""
        preprocessor = AudioPreprocessor()
        assert preprocessor.target_sample_rate == 16000
        assert preprocessor.max_duration == 10.0
        assert preprocessor.normalize_audio is True
        assert preprocessor.processor is not None
    
    def test_load_audio(self, audio_preprocessor, temp_audio_file):
        """Test audio loading functionality."""
        waveform, sample_rate = audio_preprocessor.load_audio(temp_audio_file)
        
        assert isinstance(waveform, np.ndarray)
        assert isinstance(sample_rate, int)
        assert len(waveform.shape) == 1  # Should be mono
        assert sample_rate > 0
    
    def test_load_audio_invalid_file(self, audio_preprocessor):
        """Test loading invalid audio file."""
        with pytest.raises(ValueError, match="Failed to load audio file"):
            audio_preprocessor.load_audio("nonexistent_file.wav")
    
    def test_resample_audio(self, audio_preprocessor, sample_audio_data):
        """Test audio resampling."""
        waveform, original_sr = sample_audio_data
        
        # Test resampling from different sample rate
        resampled = audio_preprocessor.resample_audio(waveform, original_sr)
        expected_length = int(len(waveform) * audio_preprocessor.target_sample_rate / original_sr)
        
        assert isinstance(resampled, np.ndarray)
        assert abs(len(resampled) - expected_length) <= 1  # Allow for small rounding differences
    
    def test_resample_audio_same_rate(self, audio_preprocessor, sample_audio_data):
        """Test resampling when sample rates are the same."""
        waveform, _ = sample_audio_data
        target_sr = audio_preprocessor.target_sample_rate
        
        resampled = audio_preprocessor.resample_audio(waveform, target_sr)
        np.testing.assert_array_equal(resampled, waveform)
    
    def test_pad_or_truncate_short_audio(self, audio_preprocessor):
        """Test padding short audio."""
        short_waveform = np.random.randn(1000)  # Very short audio
        max_length = int(audio_preprocessor.max_duration * audio_preprocessor.target_sample_rate)
        
        processed = audio_preprocessor.pad_or_truncate(short_waveform)
        
        assert len(processed) == max_length
        # Check that original audio is preserved at the beginning
        np.testing.assert_array_equal(processed[:len(short_waveform)], short_waveform)
        # Check that padding is zeros
        np.testing.assert_array_equal(processed[len(short_waveform):], np.zeros(max_length - len(short_waveform)))
    
    def test_pad_or_truncate_long_audio(self, audio_preprocessor):
        """Test truncating long audio."""
        max_length = int(audio_preprocessor.max_duration * audio_preprocessor.target_sample_rate)
        long_waveform = np.random.randn(max_length * 2)  # Twice as long as max
        
        processed = audio_preprocessor.pad_or_truncate(long_waveform)
        
        assert len(processed) == max_length
        # Check that audio is truncated from center
        start_idx = (len(long_waveform) - max_length) // 2
        expected = long_waveform[start_idx:start_idx + max_length]
        np.testing.assert_array_equal(processed, expected)
    
    def test_pad_or_truncate_exact_length(self, audio_preprocessor):
        """Test audio with exact target length."""
        max_length = int(audio_preprocessor.max_duration * audio_preprocessor.target_sample_rate)
        exact_waveform = np.random.randn(max_length)
        
        processed = audio_preprocessor.pad_or_truncate(exact_waveform)
        
        assert len(processed) == max_length
        np.testing.assert_array_equal(processed, exact_waveform)
    
    def test_normalize_amplitude(self, audio_preprocessor):
        """Test amplitude normalization."""
        # Create waveform with known RMS
        waveform = np.array([1.0, -1.0, 0.5, -0.5])
        normalized = audio_preprocessor.normalize_amplitude(waveform)
        
        assert isinstance(normalized, np.ndarray)
        assert len(normalized) == len(waveform)
        # Check that values are clipped to [-1, 1]
        assert np.all(normalized >= -1.0)
        assert np.all(normalized <= 1.0)
    
    def test_normalize_amplitude_disabled(self):
        """Test when normalization is disabled."""
        preprocessor = AudioPreprocessor(normalize_audio=False)
        waveform = np.array([2.0, -2.0, 1.0, -1.0])  # Values outside [-1, 1]
        
        result = preprocessor.normalize_amplitude(waveform)
        np.testing.assert_array_equal(result, waveform)
    
    def test_preprocess_single_audio(self, audio_preprocessor, temp_audio_file):
        """Test complete preprocessing of single audio file."""
        result = audio_preprocessor.preprocess_single_audio(temp_audio_file)
        
        # Check return structure
        expected_keys = {"input_values", "raw_waveform", "file_path"}
        assert set(result.keys()).issuperset(expected_keys)
        
        # Check types
        assert isinstance(result["input_values"], torch.Tensor)
        assert isinstance(result["raw_waveform"], np.ndarray)
        assert isinstance(result["file_path"], str)
        
        # Check dimensions
        assert result["input_values"].dim() == 1  # Should be 1D tensor
        max_length = int(audio_preprocessor.max_duration * audio_preprocessor.target_sample_rate)
        assert len(result["raw_waveform"]) == max_length
    
    def test_preprocess_single_audio_invalid_file(self, audio_preprocessor):
        """Test preprocessing with invalid file."""
        with pytest.raises(ValueError, match="Failed to load audio file"):
            audio_preprocessor.preprocess_single_audio("nonexistent.wav")
    
    def test_create_dataset_csv_structure(self, audio_preprocessor, tmp_path):
        """Test dataset CSV creation structure."""
        # Create temporary directories and files
        hc_dir = tmp_path / "HC"
        pd_dir = tmp_path / "PD"
        hc_dir.mkdir()
        pd_dir.mkdir()
        
        # Create sample audio files
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        waveform = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        for i, directory in enumerate([hc_dir, pd_dir]):
            for j in range(2):  # Create 2 files per class
                file_path = directory / f"sample_{j}.wav"
                sf.write(str(file_path), waveform, sample_rate)
        
        output_csv = tmp_path / "dataset.csv"
        
        # Create dataset CSV
        df = audio_preprocessor.create_dataset_csv(
            hc_dir=hc_dir,
            pd_dir=pd_dir,
            output_path=output_csv,
        )
        
        # Check DataFrame structure
        expected_columns = {
            "file_path", "file_name", "label", "class_name", 
            "duration", "original_sample_rate"
        }
        assert set(df.columns) == expected_columns
        
        # Check data types
        assert df["label"].dtype in [np.int64, np.int32]
        assert df["duration"].dtype == np.float64
        
        # Check labels
        hc_samples = df[df["class_name"] == "HC"]
        pd_samples = df[df["class_name"] == "PD"]
        assert len(hc_samples) == 2
        assert len(pd_samples) == 2
        assert all(hc_samples["label"] == 0)
        assert all(pd_samples["label"] == 1)
        
        # Check CSV file was created
        assert output_csv.exists()
        
        # Load and verify CSV
        loaded_df = pd.read_csv(output_csv)
        pd.testing.assert_frame_equal(df, loaded_df)


class TestAudioPreprocessorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_short_audio(self, audio_preprocessor, tmp_path):
        """Test with very short audio file."""
        # Create very short audio (0.1 seconds)
        sample_rate = 16000
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        waveform = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        file_path = tmp_path / "short.wav"
        sf.write(str(file_path), waveform, sample_rate)
        
        result = audio_preprocessor.preprocess_single_audio(file_path)
        
        # Should be padded to max length
        max_length = int(audio_preprocessor.max_duration * audio_preprocessor.target_sample_rate)
        assert len(result["raw_waveform"]) == max_length
    
    def test_silent_audio(self, audio_preprocessor, tmp_path):
        """Test with silent audio file."""
        # Create silent audio
        sample_rate = 16000
        duration = 1.0
        waveform = np.zeros(int(sample_rate * duration))
        
        file_path = tmp_path / "silent.wav"
        sf.write(str(file_path), waveform, sample_rate)
        
        result = audio_preprocessor.preprocess_single_audio(file_path)
        
        # Should handle silent audio without errors
        assert isinstance(result["input_values"], torch.Tensor)
        assert not np.isnan(result["raw_waveform"]).any()
    
    def test_empty_directories(self, audio_preprocessor, tmp_path):
        """Test dataset creation with empty directories."""
        hc_dir = tmp_path / "HC"
        pd_dir = tmp_path / "PD"
        hc_dir.mkdir()
        pd_dir.mkdir()
        
        output_csv = tmp_path / "empty_dataset.csv"
        
        df = audio_preprocessor.create_dataset_csv(
            hc_dir=hc_dir,
            pd_dir=pd_dir,
            output_path=output_csv,
        )
        
        # Should create empty DataFrame
        assert len(df) == 0
        # When no files are found, DataFrame might be completely empty
        # This is acceptable behavior