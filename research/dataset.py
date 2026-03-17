import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

class VoiceDataset(Dataset):
    """
    Dataset for loading audio files and their corresponding GRBAS ratings.
    Simplified for Minimal Reproducible Example (MRE).
    """
    def __init__(self, metadata_df, audio_dir, feature_extractor, max_length=1024):
        self.metadata = metadata_df
        self.audio_dir = audio_dir
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.grbas_cols = ['Grade', 'Roughness', 'Breathiness', 'Asthenia', 'Strain']

    def __len__(self):
        return len(self.metadata)

    def preprocess_audio(self, audio_path):
        """
        Standardized preprocessing: 16kHz, Mel-spectrogram, fixed length.
        """
        # Load audio (target_sr=16000 as per study)
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Extract features using AST Feature Extractor
        inputs = self.feature_extractor(
            speech, 
            sampling_rate=sr, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt"
        )
        return inputs.input_values.squeeze(0)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['audio_file_name'])
        
        # Preprocess
        input_values = self.preprocess_audio(audio_path)
        
        # Get ground truth labels
        labels = torch.tensor([row[col] for col in self.grbas_cols], dtype=torch.float)
        
        return {
            "input_values": input_values,
            "labels": labels,
            "filename": row['audio_file_name']
        }

def get_feature_extractor():
    """Returns the standard AST feature extractor."""
    return AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
