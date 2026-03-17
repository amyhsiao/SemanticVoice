import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import ASTModel, ASTConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Constants
TARGET_SR = 16000
N_MEL = 128
TARGET_LENGTH = 1024
PROCRUSTES_SCALER_PATH = "models/procrustes_scaler_params.json"
ROTATION_MATRIX_PATH = "models/rotation_matrix_final.npy"
AUDIO_ORIGIN_PATH = "models/audio_origin_all.npy"
PROCRUSTES_SCALE_PATH = "models/procrustes_scale.npy"
TEXT_ORIGIN_PATH = "models/text_origin.npy"
ANCHOR_DESCRIPTORS_PATH = "models/anchor_descriptors.json"

class ASTPredictor:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading AST Model on {self.device}...")
        
        # 1. Load Feature Extractor (Base AST)
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(self.device)
        print(f"DEBUG: cls_token before load: {self.ast.embeddings.cls_token[0,0,:5].detach().cpu().numpy()}")
        
        # 2. Load Your Fine-Tuned Weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                clean_state = {k.replace('ast.', ''): v for k, v in checkpoint.items() if k.startswith('ast.')}
                if not clean_state:
                    clean_state = checkpoint
                
                msg = self.ast.load_state_dict(clean_state, strict=True)
                print(f"Model loading info: {msg}")
                # Verify a weight that was definitely fine-tuned (last layer)
                l11_weight = self.ast.encoder.layer[11].output.dense.weight[0, :5].detach().cpu().numpy()
                print(f"DEBUG: Layer 11 weight sample: {l11_weight}")
        except Exception as e:
            print(f"Warning: Could not load custom weights fully: {e}")

        self.ast.eval()

        # 3. Load Sentence Transformer for Semantic Comparison
        self.sent_model = SentenceTransformer('all-mpnet-base-v2').to(self.device)
        
        # 4. Load Procrustes Alignment Parameters
        self.R = np.load(ROTATION_MATRIX_PATH)
        self.audio_origin = np.load(AUDIO_ORIGIN_PATH)
        self.scale = np.load(PROCRUSTES_SCALE_PATH)[0]
        self.text_origin = np.load(TEXT_ORIGIN_PATH)
        
        print(f"DEBUG: audio_origin mean: {np.mean(self.audio_origin):.6f}")
        print(f"DEBUG: text_origin mean: {np.mean(self.text_origin):.6f}")
        print(f"DEBUG: scale: {self.scale:.6f}")

        with open(ANCHOR_DESCRIPTORS_PATH, 'r') as f:
            self.anchor_descriptors = json.load(f)

        # 5. Load Standard Scaler Parameters (Z-score stats from Grade 0)
        with open(PROCRUSTES_SCALER_PATH, 'r') as f:
            scaler_params = json.load(f)
        
        self.adjectives = list(scaler_params['mean'].keys())
        self.scaler_mean = np.array([scaler_params['mean'][adj] for adj in self.adjectives])
        self.scaler_std = np.array([scaler_params['std'][adj] for adj in self.adjectives])

        # 6. Pre-encode Adjectives
        self.adj_embs = self._prepare_adjective_embeddings()
        print(f"DEBUG: adj_embs[0] norm: {np.linalg.norm(self.adj_embs[0]):.6f}")

    def _prepare_adjective_embeddings(self):
        """Create mean embeddings for target adjectives, using refined descriptors for GRBAS."""
        templates = [
            "The patient's voice is {}.",
            "The audio demonstrates {} quality.",
            "A {} voice.",
            "Auditory perception: {}.",
            "The voice sounds {}.",
        ]
        
        # Map for refined descriptors (severe case for main GRBAS terms)
        refined_map = {k.lower(): self.anchor_descriptors['severe'][k] for k in self.anchor_descriptors['severe']}
        for k in self.anchor_descriptors['severe']:
            refined_map[k] = self.anchor_descriptors['severe'][k]

        adjective_embeddings = []
        print(f"Encoding {len(self.adjectives)} adjectives with Procrustes space logic...")
        
        for adj in self.adjectives:
            if adj in refined_map:
                all_sentences = []
                for desc in refined_map[adj]:
                    all_sentences.extend([template.format(desc) for template in templates])
                # Match research: mean of RAW embeddings (not normalized ones)
                mean_embedding = np.mean(self.sent_model.encode(all_sentences), axis=0)
            else:
                sentences = [template.format(adj) for template in templates]
                embeddings = self.sent_model.encode(sentences)
                mean_embedding = np.mean(embeddings, axis=0)
            adjective_embeddings.append(mean_embedding)
        
        return np.array(adjective_embeddings)

    def preprocess(self, audio_path):
        # 1. Load Audio
        audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        
        # 2. RMS Normalization
        rms = np.sqrt(np.mean(audio**2))
        print(f"DEBUG: audio original RMS: {rms:.6f}")
        if rms > 1e-6:
            audio = audio / rms
            
        # 3. Explicit Type Casting (Match Colab)
        audio = audio.astype(np.float32)

        # 4. Length Standardization (Time Domain)
        # IMPORTANT: Match research pipeline exactly. 
        # Research saved 1875 frames (~30s) then training cropped to 1024.
        # Decibel normalization (ref=np.max) MUST happen on the 1875 window.
        n_fft = 1024
        hop_length = 256
        research_frames = 1875
        max_audio_samples = (research_frames - 1) * hop_length + n_fft
        
        if len(audio) > max_audio_samples:
            # Match research pipeline 'test' mode: Center-crop
            start_idx = (len(audio) - max_audio_samples) // 2
            audio = audio[start_idx:start_idx + max_audio_samples]
        elif len(audio) < max_audio_samples:
            padding = max_audio_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')

        # 5. Compute Mel Spectrogram (1875 frames)
        mel_spectrogram_power = librosa.feature.melspectrogram(
            y=audio, 
            sr=TARGET_SR, 
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=N_MEL
        )
        print(f"DEBUG: mel_spectrogram_power max: {np.max(mel_spectrogram_power):.6e}")

        # 6. Log Scale Conversion
        mel_db = librosa.power_to_db(mel_spectrogram_power, ref=np.max)

        # 7. Ensure research frame length (1875) for normalization consistency
        current_frames = mel_db.shape[1]
        if current_frames > research_frames:
            mel_db = mel_db[:, :research_frames]
        elif current_frames < research_frames:
            padding_frames = research_frames - current_frames
            mel_db = np.pad(mel_db, ((0, 0), (0, padding_frames)), mode='constant', constant_values=-100.0)

        # 8. Global Normalization (calculated from 1875-frame data)
        mel_db = (mel_db - (-60.2558)) / 17.5056
        
        # 9. Crop to model's TARGET_LENGTH (1024)
        if mel_db.shape[1] > TARGET_LENGTH:
            mel_db = mel_db[:, :TARGET_LENGTH]
        
        return torch.tensor(mel_db).float().unsqueeze(0).to(self.device)

    def predict(self, audio_path):
        input_tensor = self.preprocess(audio_path)
        
        # 1. Get Audio Embedding
        with torch.no_grad():
            outputs = self.ast(input_values=input_tensor)
            audio_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        print(f"DEBUG: audio_emb norm: {np.linalg.norm(audio_emb):.6f}")
        print(f"DEBUG: audio_emb[0,:10]: {audio_emb[0,:10]}")

        # 2. Procrustes Alignment
        # Center -> Rotate & Scale -> Shift to Text Space
        centered = audio_emb - self.audio_origin
        rotated = np.dot(centered, self.R) * self.scale
        aligned_emb = rotated + self.text_origin

        print(f"DEBUG: aligned_emb norm: {np.linalg.norm(aligned_emb):.6f}")
        print(f"DEBUG: aligned_emb[0,:10]: {aligned_emb[0,:10]}")

        # 3. Compute Adjective Similarity
        sim_scores = cosine_similarity(aligned_emb, self.adj_embs)[0]
        
        # Diagnostic: Print raw scores for main parameters
        print(f"DEBUG: Adjectives order: {self.adjectives[:5]}")
        diag_indices = [i for i, adj in enumerate(self.adjectives) if adj in ['Grade', 'Roughness']]
        for idx in diag_indices:
            print(f"DEBUG: Raw {self.adjectives[idx]} score: {sim_scores[idx]:.6f}, mean: {self.scaler_mean[idx]:.6f}")
        
        # 4. Apply Z-score Scaling (based on Grade 0 stats)
        scaled_scores = (sim_scores - self.scaler_mean) / self.scaler_std
        
        return {
            "adjectives": dict(zip(self.adjectives, [float(s) for s in scaled_scores]))
        }