"""Dataset class for Parkinson's voice data compatible with Wav2Vec2."""

from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

from .preprocessing import AudioPreprocessor


class ParkinsonsVoiceDataset(Dataset):
    """PyTorch Dataset for Parkinson's voice detection using Wav2Vec2."""
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        processor_name: str = "facebook/wav2vec2-base-960h",
        target_sample_rate: int = 16000,
        max_duration: float = 10.0,
        normalize_audio: bool = True,
        transform: Optional[Any] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file containing dataset metadata
            processor_name: Wav2Vec2 processor model name
            target_sample_rate: Target sampling rate for audio files
            max_duration: Maximum duration in seconds for audio clips
            normalize_audio: Whether to normalize audio amplitude
            transform: Optional transform to apply to the data
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(
            processor_name=processor_name,
            target_sample_rate=target_sample_rate,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
        )
        
        # Validate dataset
        self._validate_dataset()
    
    def _validate_dataset(self) -> None:
        """Validate that all required columns exist and files are accessible."""
        required_columns = ["file_path", "label"]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check if files exist (sample a few)
        sample_size = min(5, len(self.df))
        sample_files = self.df["file_path"].sample(sample_size)
        
        for file_path in sample_files:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing processed audio data and metadata
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get row data
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        label = int(row["label"])
        
        # Preprocess audio
        try:
            audio_data = self.preprocessor.preprocess_single_audio(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess audio {file_path}: {str(e)}")
        
        # Prepare sample
        sample = {
            "input_values": audio_data["input_values"],
            "label": torch.tensor(label, dtype=torch.long),
            "file_path": file_path,
            "file_name": Path(file_path).name,
        }
        
        # Add attention mask if available
        if audio_data["attention_mask"] is not None:
            sample["attention_mask"] = audio_data["attention_mask"]
        
        # Apply transform if specified
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        return self.df["label"].value_counts().to_dict()
    
    def get_duration_stats(self) -> Dict[str, float]:
        """Get duration statistics for the dataset."""
        if "duration" in self.df.columns:
            return {
                "mean": self.df["duration"].mean(),
                "std": self.df["duration"].std(),
                "min": self.df["duration"].min(),
                "max": self.df["duration"].max(),
            }
        else:
            return {}
    
    def get_sample_by_class(self, class_label: int, n_samples: int = 5) -> pd.DataFrame:
        """
        Get sample files for a specific class.
        
        Args:
            class_label: Class label (0 for HC, 1 for PD)
            n_samples: Number of samples to return
            
        Returns:
            DataFrame with sample files for the specified class
        """
        class_data = self.df[self.df["label"] == class_label]
        return class_data.head(n_samples)


def create_train_val_split(
    dataset: ParkinsonsVoiceDataset,
    val_split: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Create train/validation split from the dataset.
    
    Args:
        dataset: ParkinsonsVoiceDataset instance
        val_split: Fraction of data to use for validation
        stratify: Whether to stratify the split by class
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from sklearn.model_selection import train_test_split
    
    # Get indices
    indices = list(range(len(dataset)))
    
    if val_split <= 0.0:
        # Handle zero validation split
        return dataset, torch.utils.data.Subset(dataset, [])
    
    if stratify:
        labels = [dataset.df.iloc[i]["label"] for i in indices]
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_split,
            stratify=labels,
            random_state=random_state,
        )
    else:
        train_indices, val_indices = train_test_split(
            indices, test_size=val_split, random_state=random_state
        )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    dataset = ParkinsonsVoiceDataset("data/dataset_metadata.csv")
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    print(f"Duration stats: {dataset.get_duration_stats()}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input shape: {sample['input_values'].shape}")
    print(f"Label: {sample['label']}")
    
    # Create train/val split
    train_dataset, val_dataset = create_train_val_split(dataset)
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")