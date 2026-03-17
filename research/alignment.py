import numpy as np
import json
from sentence_transformers import SentenceTransformer
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity

class ProcrustesAligner:
    """
    Core implementation of the Zero-Shot Procrustes Alignment.
    Maps Audio Embeddings (768-d) to Clinical Text Embeddings (768-d).
    """
    def __init__(self, text_model_name='all-mpnet-base-v2'):
        self.text_model = SentenceTransformer(text_model_name)
        self.grbas_parameters = ["Grade", "Roughness", "Breathiness", "Asthenia", "Strain"]
        # Parameters that need to be flipped based on clinical descriptors
        # (Normalization: Healthy-to-Severe direction)
        self.flip_params = ["Grade", "Roughness", "Asthenia", "Strain"]
        
        # Final rotation, scale, and origins
        self.R = None
        self.scale = None
        self.audio_origin = None
        self.text_origin = None

    def create_text_anchors(self):
        """
        Creates refined clinical text anchors for GRBAS parameters.
        Logic from 'checkpoint 6' using medically-precise descriptors.
        """
        templates = [
            "The patient's voice is {}.",
            "The audio demonstrates {} quality.",
            "A {} voice.",
            "Auditory perception: {}.",
            "The voice sounds {}.",
            "Evidence of {} in the vocal signal.",
        ]

        # Clinical descriptors optimized during research
        severe_descriptors = {
            "Grade": ["severely disordered", "very hoarse", "severely dysphonic", "extremely abnormal"],
            "Roughness": ["severely rough", "very gravelly", "harsh and grating", "raspy", "vocal fry"],
            "Breathiness": ["severely breathy", "very airy", "whispery", "turbulent airflow"],
            "Asthenia": ["severely asthenic", "very weak", "feeble", "hypophonic"],
            "Strain": ["severely strained", "very thin", "nasal", "jerky", "stuttering", "choppy", "hyperfunctional"],
        }
        normal_descriptors = ["normal", "healthy", "clear", "stable", "natural"]

        text_anchors = []
        
        # Healthy Origin (Text Space)
        normal_sentences = []
        for desc in normal_descriptors:
            normal_sentences.extend([template.format(desc) for template in templates])
        self.text_origin = np.mean(self.text_model.encode(normal_sentences), axis=0)

        for param in self.grbas_parameters:
            # Create Severe Centroid
            high_embeddings = []
            for desc in severe_descriptors[param]:
                sentences = [template.format(desc) for template in templates]
                high_embeddings.append(np.mean(self.text_model.encode(sentences), axis=0))
            v_high = np.mean(high_embeddings, axis=0)

            # Define Axis direction
            if param in self.flip_params:
                v_text_axis = self.text_origin - v_high
            else:
                v_text_axis = v_high - self.text_origin
            
            text_anchors.append(v_text_axis)

        return np.array(text_anchors)

    def calculate_audio_anchors(self, embeddings, labels_df):
        """
        Calculates direction vectors in the audio space based on ground truth.
        """
        audio_anchors = []
        
        # Audio Origin: Centroid of all Grade 0 (Healthy) samples
        grade0_mask = labels_df['Grade'] == 0.0
        self.audio_origin = np.mean(embeddings[grade0_mask], axis=0)

        for param in self.grbas_parameters:
            # High (Severe) Group: score >= 2
            # Low (Normal) Group: score == 0
            high_mask = labels_df[param] >= 2.0
            low_mask = labels_df[param] == 0.0
            
            mu_high = np.mean(embeddings[high_mask], axis=0)
            mu_low = np.mean(embeddings[low_mask], axis=0)
            
            # The direction vector in audio space
            audio_anchors.append(mu_high - mu_low)
            
        return np.array(audio_anchors)

    def fit(self, audio_embeddings, labels_df):
        """
        Trains the Procrustes Alignment by finding the rotation matrix R and scale.
        """
        text_anchors = self.create_text_anchors()
        audio_anchors = self.calculate_audio_anchors(audio_embeddings, labels_df)
        
        # Solve for R, scale using scipy's orthogonal_procrustes
        self.R, self.scale = orthogonal_procrustes(audio_anchors, text_anchors)
        return self.R, self.scale

    def transform(self, test_embeddings):
        """
        Applies the alignment: Translation -> Rotation/Scaling -> Translation.
        """
        # 1. Shift audio embeddings to healthy origin (centered)
        centered = test_embeddings - self.audio_origin
        
        # 2. Rotate and Scale into text space
        rotated = np.dot(centered, self.R) * self.scale
        
        # 3. Shift to text healthy origin
        aligned = rotated + self.text_origin
        return aligned

    def generate_scores(self, aligned_embeddings, target_adjectives):
        """
        Calculates Zero-Shot similarity scores for target adjectives.
        """
        # Encode target clinical adjectives
        adj_embeddings = self.text_model.encode(target_adjectives)
        
        # Compute Cosine Similarity
        scores = cosine_similarity(aligned_embeddings, adj_embeddings)
        return scores
