"""Tests for dataset functionality."""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import soundfile as sf

from src.parkinsons_voice.data.dataset import (
    ParkinsonsVoiceDataset, 
    create_train_val_split,
    create_dataloaders
)


@pytest.fixture
def sample_dataset_csv(tmp_path):
    """Create sample dataset CSV for testing."""
    # Create sample audio files
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    waveform = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    data = []
    for i in range(10):  # 5 HC, 5 PD
        label = i % 2  # Alternate between 0 and 1
        class_name = "HC" if label == 0 else "PD"
        
        # Create audio file
        file_path = tmp_path / f"sample_{i}_{class_name}.wav"
        sf.write(str(file_path), waveform, sample_rate)
        
        data.append({
            "file_path": str(file_path),
            "file_name": file_path.name,
            "label": label,
            "class_name": class_name,
            "duration": duration,
            "original_sample_rate": sample_rate,
        })
    
    # Create CSV
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_dataset.csv"
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


class TestParkinsonsVoiceDataset:
    """Test cases for ParkinsonsVoiceDataset class."""
    
    def test_init_valid_csv(self, sample_dataset_csv):
        """Test dataset initialization with valid CSV."""
        dataset = ParkinsonsVoiceDataset(
            csv_path=sample_dataset_csv,
            max_duration=2.0,  # Shorter for faster tests
        )
        
        assert len(dataset) == 10
        assert dataset.preprocessor is not None
    
    def test_init_missing_columns(self, tmp_path):
        """Test initialization with CSV missing required columns."""
        # Create CSV without required columns
        df = pd.DataFrame({"wrong_column": [1, 2, 3]})
        csv_path = tmp_path / "invalid.csv"
        df.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            ParkinsonsVoiceDataset(csv_path=str(csv_path))
    
    def test_init_missing_files(self, tmp_path):
        """Test initialization when audio files don't exist."""
        # Create CSV with non-existent file paths
        data = [
            {"file_path": "/nonexistent/file1.wav", "label": 0},
            {"file_path": "/nonexistent/file2.wav", "label": 1},
        ]
        df = pd.DataFrame(data)
        csv_path = tmp_path / "invalid_files.csv"
        df.to_csv(csv_path, index=False)
        
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            ParkinsonsVoiceDataset(csv_path=str(csv_path))
    
    def test_len(self, sample_dataset_csv):
        """Test dataset length."""
        dataset = ParkinsonsVoiceDataset(sample_dataset_csv, max_duration=2.0)
        assert len(dataset) == 10
    
    def test_getitem(self, sample_dataset_csv):
        """Test getting single item from dataset."""
        dataset = ParkinsonsVoiceDataset(sample_dataset_csv, max_duration=2.0)
        
        # Test getting first item
        sample = dataset[0]
        
        # Check return structure
        expected_keys = {"input_values", "label", "file_path", "file_name"}
        assert set(sample.keys()).issuperset(expected_keys)
        
        # Check types
        assert isinstance(sample["input_values"], torch.Tensor)
        assert isinstance(sample["label"], torch.Tensor)
        assert isinstance(sample["file_path"], str)
        assert isinstance(sample["file_name"], str)
        
        # Check tensor properties
        assert sample["input_values"].dim() == 1  # 1D tensor
        assert sample["label"].dtype == torch.long
        assert sample["label"].item() in [0, 1]  # Valid label
    
    def test_getitem_with_tensor_index(self, sample_dataset_csv):
        """Test getting item with tensor index."""
        dataset = ParkinsonsVoiceDataset(sample_dataset_csv, max_duration=2.0)
        
        # Test with tensor index
        idx = torch.tensor(1)
        sample = dataset[idx]
        
        assert isinstance(sample["input_values"], torch.Tensor)
        assert isinstance(sample["label"], torch.Tensor)
    
    def test_getitem_invalid_audio(self, tmp_path):
        """Test getting item when audio preprocessing fails."""
        # Create CSV with valid structure but create empty files
        data = []
        for i in range(2):
            file_path = tmp_path / f"empty_{i}.wav"
            file_path.write_text("")  # Create empty file (invalid audio)
            
            data.append({
                "file_path": str(file_path),
                "file_name": file_path.name,
                "label": i,
                "class_name": "HC" if i == 0 else "PD",
            })
        
        df = pd.DataFrame(data)
        csv_path = tmp_path / "invalid_audio.csv"
        df.to_csv(csv_path, index=False)
        
        dataset = ParkinsonsVoiceDataset(csv_path=str(csv_path), max_duration=2.0)
        
        with pytest.raises(RuntimeError, match="Failed to preprocess audio"):
            dataset[0]
    
    def test_get_class_distribution(self, sample_dataset_csv):
        """Test class distribution calculation."""
        dataset = ParkinsonsVoiceDataset(sample_dataset_csv, max_duration=2.0)
        
        distribution = dataset.get_class_distribution()
        
        assert isinstance(distribution, dict)
        assert 0 in distribution
        assert 1 in distribution
        assert distribution[0] == 5  # 5 HC samples
        assert distribution[1] == 5  # 5 PD samples
    
    def test_get_duration_stats(self, sample_dataset_csv):
        """Test duration statistics calculation."""
        dataset = ParkinsonsVoiceDataset(sample_dataset_csv, max_duration=2.0)
        
        stats = dataset.get_duration_stats()
        
        assert isinstance(stats, dict)
        expected_keys = {"mean", "std", "min", "max"}
        assert set(stats.keys()) == expected_keys
        
        # All durations should be 1.0 in our test data
        assert abs(stats["mean"] - 1.0) < 1e-6
        assert abs(stats["min"] - 1.0) < 1e-6
        assert abs(stats["max"] - 1.0) < 1e-6
    
    def test_get_duration_stats_no_duration_column(self, tmp_path):
        """Test duration stats when duration column is missing."""
        # Create CSV without duration column
        data = [
            {"file_path": "/path/file1.wav", "label": 0},
            {"file_path": "/path/file2.wav", "label": 1},
        ]
        df = pd.DataFrame(data)
        csv_path = tmp_path / "no_duration.csv"
        df.to_csv(csv_path, index=False)
        
        # Create dataset (this will fail at initialization due to missing files,
        # but we can test the method directly)
        dataset = ParkinsonsVoiceDataset.__new__(ParkinsonsVoiceDataset)
        dataset.df = df
        
        stats = dataset.get_duration_stats()
        assert stats == {}
    
    def test_get_sample_by_class(self, sample_dataset_csv):
        """Test getting samples by class."""
        dataset = ParkinsonsVoiceDataset(sample_dataset_csv, max_duration=2.0)
        
        # Test HC class (label 0)
        hc_samples = dataset.get_sample_by_class(0, n_samples=3)
        assert len(hc_samples) == 3
        assert all(hc_samples["label"] == 0)
        assert all(hc_samples["class_name"] == "HC")
        
        # Test PD class (label 1)
        pd_samples = dataset.get_sample_by_class(1, n_samples=2)
        assert len(pd_samples) == 2
        assert all(pd_samples["label"] == 1)
        assert all(pd_samples["class_name"] == "PD")
    
    def test_custom_transform(self, sample_dataset_csv):
        """Test dataset with custom transform."""
        def dummy_transform(sample):
            sample["transformed"] = True
            return sample
        
        dataset = ParkinsonsVoiceDataset(
            sample_dataset_csv, 
            max_duration=2.0,
            transform=dummy_transform
        )
        
        sample = dataset[0]
        assert "transformed" in sample
        assert sample["transformed"] is True


class TestDatasetUtilities:
    """Test utility functions for dataset handling."""
    
    def test_create_train_val_split(self, sample_dataset_csv):
        """Test train/validation split creation."""
        dataset = ParkinsonsVoiceDataset(sample_dataset_csv, max_duration=2.0)
        
        train_dataset, val_dataset = create_train_val_split(
            dataset, val_split=0.3, stratify=True, random_state=42
        )
        
        # Check split sizes (7 train, 3 val)
        assert len(train_dataset) == 7
        assert len(val_dataset) == 3
        
        # Check that datasets are Subset instances
        assert isinstance(train_dataset, torch.utils.data.Subset)
        assert isinstance(val_dataset, torch.utils.data.Subset)
    
    def test_create_train_val_split_no_stratify(self, sample_dataset_csv):
        """Test train/validation split without stratification."""
        dataset = ParkinsonsVoiceDataset(sample_dataset_csv, max_duration=2.0)
        
        train_dataset, val_dataset = create_train_val_split(
            dataset, val_split=0.2, stratify=False, random_state=42
        )
        
        # Check split sizes (8 train, 2 val)
        assert len(train_dataset) == 8
        assert len(val_dataset) == 2
    
    def test_create_dataloaders(self, sample_dataset_csv):
        """Test DataLoader creation."""
        dataset = ParkinsonsVoiceDataset(sample_dataset_csv, max_duration=2.0)
        train_dataset, val_dataset = create_train_val_split(dataset, val_split=0.2)
        
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, 
            batch_size=2, num_workers=0, pin_memory=False
        )
        
        # Check DataLoader types
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)
        
        # Check DataLoader properties
        assert train_loader.batch_size == 2
        assert val_loader.batch_size == 2
        # Note: DataLoader doesn't have shuffle attribute directly, it's in the sampler
        assert hasattr(train_loader, 'sampler')
        assert hasattr(val_loader, 'sampler')
    
    def test_dataloader_iteration(self, sample_dataset_csv):
        """Test iterating through DataLoaders."""
        dataset = ParkinsonsVoiceDataset(sample_dataset_csv, max_duration=2.0)
        train_dataset, val_dataset = create_train_val_split(dataset, val_split=0.2)
        
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, 
            batch_size=3, num_workers=0, pin_memory=False
        )
        
        # Test training loader
        train_batch = next(iter(train_loader))
        assert "input_values" in train_batch
        assert "label" in train_batch
        assert train_batch["input_values"].shape[0] == 3  # batch size
        assert train_batch["label"].shape[0] == 3
        
        # Test validation loader
        val_batch = next(iter(val_loader))
        assert "input_values" in val_batch
        assert "label" in val_batch
        # Val loader might have smaller batch due to small dataset
        assert val_batch["input_values"].shape[0] <= 3


class TestDatasetEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_single_sample_dataset(self, tmp_path):
        """Test dataset with single sample."""
        # Create single audio file
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        waveform = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        file_path = tmp_path / "single.wav"
        sf.write(str(file_path), waveform, sample_rate)
        
        # Create CSV
        data = [{
            "file_path": str(file_path),
            "file_name": file_path.name,
            "label": 0,
            "class_name": "HC",
            "duration": duration,
            "original_sample_rate": sample_rate,
        }]
        df = pd.DataFrame(data)
        csv_path = tmp_path / "single_sample.csv"
        df.to_csv(csv_path, index=False)
        
        dataset = ParkinsonsVoiceDataset(csv_path=str(csv_path), max_duration=2.0)
        
        assert len(dataset) == 1
        sample = dataset[0]
        assert isinstance(sample["input_values"], torch.Tensor)
        assert sample["label"].item() == 0
    
    def test_large_batch_size(self, sample_dataset_csv):
        """Test DataLoader with batch size larger than dataset."""
        dataset = ParkinsonsVoiceDataset(sample_dataset_csv, max_duration=2.0)
        train_dataset, val_dataset = create_train_val_split(dataset, val_split=0.2)
        
        # Use batch size larger than validation set
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, 
            batch_size=20, num_workers=0, pin_memory=False
        )
        
        # Should handle gracefully - but may be empty if batch size > dataset size
        try:
            train_batch = next(iter(train_loader))
            # Batch sizes should be limited by dataset size
            assert train_batch["input_values"].shape[0] <= len(train_dataset)
        except StopIteration:
            # This can happen if batch size > dataset size with drop_last=True
            pass
        
        try:
            val_batch = next(iter(val_loader))
            assert val_batch["input_values"].shape[0] <= len(val_dataset)
        except StopIteration:
            # This can happen if batch size > dataset size with drop_last=True
            pass
    
    def test_zero_validation_split(self, sample_dataset_csv):
        """Test with zero validation split."""
        dataset = ParkinsonsVoiceDataset(sample_dataset_csv, max_duration=2.0)
        
        train_dataset, val_dataset = create_train_val_split(
            dataset, val_split=0.0, stratify=False
        )
        
        assert len(train_dataset) == len(dataset)
        assert len(val_dataset) == 0