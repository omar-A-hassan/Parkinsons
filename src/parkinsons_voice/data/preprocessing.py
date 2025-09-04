"""Audio preprocessing utilities for Wav2Vec2 compatibility."""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Processor


class AudioPreprocessor:
    """Preprocesses audio files for Wav2Vec2 model input."""
    
    def __init__(
        self,
        processor_name: str = "facebook/wav2vec2-base-960h",
        target_sample_rate: int = 16000,
        max_duration: float = 10.0,
        normalize_audio: bool = True,
    ):
        """
        Initialize the audio preprocessor.
        
        Args:
            processor_name: Wav2Vec2 processor model name
            target_sample_rate: Target sampling rate for audio files
            max_duration: Maximum duration in seconds for audio clips
            normalize_audio: Whether to normalize audio amplitude
        """
        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.normalize_audio = normalize_audio
        self.max_length = int(max_duration * target_sample_rate)
    
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return waveform and sample rate.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            waveform, sample_rate = librosa.load(file_path, sr=None, mono=True)
            return waveform, sample_rate
        except Exception as e:
            raise ValueError(f"Failed to load audio file {file_path}: {str(e)}")
    
    def resample_audio(
        self, waveform: np.ndarray, original_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            waveform: Input audio waveform
            original_sr: Original sample rate
            
        Returns:
            Resampled waveform
        """
        if original_sr != self.target_sample_rate:
            waveform = librosa.resample(
                waveform, orig_sr=original_sr, target_sr=self.target_sample_rate
            )
        return waveform
    
    def pad_or_truncate(self, waveform: np.ndarray) -> np.ndarray:
        """
        Pad or truncate audio to fixed length.
        
        Args:
            waveform: Input waveform
            
        Returns:
            Padded or truncated waveform
        """
        if len(waveform) > self.max_length:
            # Truncate from center
            start_idx = (len(waveform) - self.max_length) // 2
            waveform = waveform[start_idx : start_idx + self.max_length]
        elif len(waveform) < self.max_length:
            # Pad with zeros
            padding = self.max_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode="constant", constant_values=0)
        
        return waveform
    
    def normalize_amplitude(self, waveform: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude.
        
        Args:
            waveform: Input waveform
            
        Returns:
            Normalized waveform
        """
        if self.normalize_audio:
            # RMS normalization
            rms = np.sqrt(np.mean(waveform**2))
            if rms > 0:
                waveform = waveform / rms
            
            # Clip to prevent overflow
            waveform = np.clip(waveform, -1.0, 1.0)
        
        return waveform
    
    def preprocess_single_audio(
        self, file_path: Union[str, Path]
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Preprocess a single audio file for Wav2Vec2 input.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with processed audio data
        """
        # Load audio
        waveform, sample_rate = self.load_audio(file_path)
        
        # Resample if necessary
        waveform = self.resample_audio(waveform, sample_rate)
        
        # Normalize amplitude
        waveform = self.normalize_amplitude(waveform)
        
        # Pad or truncate
        waveform = self.pad_or_truncate(waveform)
        
        # Process with Wav2Vec2 processor
        processed = self.processor(
            waveform,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        
        return {
            "input_values": processed.input_values.squeeze(0),
            "attention_mask": processed.attention_mask.squeeze(0) if hasattr(processed, 'attention_mask') else None,
            "raw_waveform": waveform,
            "file_path": str(file_path),
        }
    
    def create_dataset_csv(
        self, 
        hc_dir: Union[str, Path], 
        pd_dir: Union[str, Path],
        output_path: Union[str, Path],
        demographics_file: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Create a CSV file with dataset metadata.
        
        Args:
            hc_dir: Directory containing healthy control audio files
            pd_dir: Directory containing Parkinson's disease audio files
            output_path: Path to save the CSV file
            demographics_file: Optional demographics file path
            
        Returns:
            DataFrame with dataset metadata
        """
        data = []
        
        # Process HC files
        hc_files = list(Path(hc_dir).glob("*.wav"))
        for file_path in hc_files:
            try:
                waveform, sr = self.load_audio(file_path)
                duration = len(waveform) / sr
                data.append({
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "label": 0,  # Healthy Control
                    "class_name": "HC",
                    "duration": duration,
                    "original_sample_rate": sr,
                })
            except Exception as e:
                warnings.warn(f"Skipping file {file_path}: {str(e)}")
        
        # Process PD files
        pd_files = list(Path(pd_dir).glob("*.wav"))
        for file_path in pd_files:
            try:
                waveform, sr = self.load_audio(file_path)
                duration = len(waveform) / sr
                data.append({
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "label": 1,  # Parkinson's Disease
                    "class_name": "PD",
                    "duration": duration,
                    "original_sample_rate": sr,
                })
            except Exception as e:
                warnings.warn(f"Skipping file {file_path}: {str(e)}")
        
        df = pd.DataFrame(data)
        
        # Add demographics if available
        if demographics_file and os.path.exists(demographics_file):
            try:
                # This would need to be customized based on the actual demographics file format
                print(f"Demographics file found at {demographics_file}")
                print("Note: Demographics integration requires manual inspection of the file format")
            except Exception as e:
                warnings.warn(f"Could not load demographics: {str(e)}")
        
        # Save CSV
        df.to_csv(output_path, index=False)
        
        # Print statistics
        print(f"Dataset created with {len(df)} samples:")
        if len(df) > 0 and 'label' in df.columns:
            print(f"  - HC samples: {len(df[df['label'] == 0])}")
            print(f"  - PD samples: {len(df[df['label'] == 1])}")
            if 'duration' in df.columns:
                print(f"  - Average duration: {df['duration'].mean():.2f}s")
                print(f"  - Duration range: {df['duration'].min():.2f}s - {df['duration'].max():.2f}s")
        print(f"Dataset saved to: {output_path}")
        
        return df


def main() -> None:
    """Main preprocessing script."""
    import sys
    from pathlib import Path
    
    # Add config utils to path for importing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from utils.config import config_loader, ModelConfig
        
        # Load configuration
        model_config = ModelConfig.from_config(config_loader)
        data_config = config_loader.load_config("data_config")
        
        # Initialize preprocessor with config values
        preprocessor = AudioPreprocessor(
            processor_name=model_config.model_name,
            target_sample_rate=model_config.target_sample_rate,
            max_duration=model_config.max_duration,
            normalize_audio=model_config.normalize_audio,
        )
        
        # Define paths from config
        data_dir = Path(data_config["paths"]["data_dir"])
        hc_dir = Path(data_config["paths"]["hc_dir"])
        pd_dir = Path(data_config["paths"]["pd_dir"])
        demographics_file = Path(data_config["paths"]["demographics_file"])
        output_csv = Path(data_config["paths"]["dataset_csv"])
        
    except ImportError:
        print("Config system not available, using defaults...")
        # Fallback to defaults
        preprocessor = AudioPreprocessor(
            processor_name="facebook/wav2vec2-base-960h",
            target_sample_rate=16000,
            max_duration=10.0,
            normalize_audio=True,
        )
        
        # Define default paths
        data_dir = Path("data")
        hc_dir = data_dir / "HC_AH"
        pd_dir = data_dir / "PD_AH"
        demographics_file = data_dir / "Demographics_age_sex.xlsx"
        output_csv = data_dir / "dataset_metadata.csv"
    
    # Create dataset CSV
    if hc_dir.exists() and pd_dir.exists():
        df = preprocessor.create_dataset_csv(
            hc_dir=hc_dir,
            pd_dir=pd_dir,
            output_path=output_csv,
            demographics_file=demographics_file if demographics_file.exists() else None,
        )
        print(f"Preprocessing completed successfully!")
    else:
        print(f"Data directories not found: {hc_dir}, {pd_dir}")


if __name__ == "__main__":
    main()