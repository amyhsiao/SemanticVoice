import os
import torch
import pandas as pd
import numpy as np
from dataset import VoiceDataset, get_feature_extractor
from model import load_model
from alignment import ProcrustesAligner
from evaluate import zscore_normalize, calculate_correlations, print_metrics

def run_pipeline(audio_dir, metadata_path, model_checkpoint=None):
    """
    Main execution pipeline for the Minimal Reproducible Example.
    """
    print("Step 1: Loading Data and Models...")
    labels_df = pd.read_csv(metadata_path)
    feature_extractor = get_feature_extractor()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load AST model (pretrained or fine-tuned)
    model = load_model(model_checkpoint, device=device)
    dataset = VoiceDataset(labels_df, audio_dir, feature_extractor)
    
    print("Step 2: Extracting Audio Embeddings (AST CLS tokens)...")
    # In a full run, we would iterate through the DataLoader. 
    # For MRE simplicity, we simulate/describe the embedding extraction:
    # (Using pre-extracted embeddings from the research if available for faster reproduction)
    
    all_embeddings = []
    # Simplified extraction loop
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            input_values = batch['input_values'].unsqueeze(0).to(device)
            outputs = model(input_values)
            all_embeddings.append(outputs['embeddings'].cpu().numpy())
    
    all_embeddings = np.vstack(all_embeddings)
    
    print(f"Step 3: Performing Procrustes Alignment (Zero-Shot Mapping)...")
    aligner = ProcrustesAligner()
    
    # 3a. Fit (Find R and scale using training logic)
    # Using all data here for MRE demonstration of the alignment logic
    R, scale = aligner.fit(all_embeddings, labels_df)
    
    # 3b. Transform (Apply the alignment)
    aligned_embeddings = aligner.transform(all_embeddings)
    
    print("Step 4: Generating Clinical Scores and Validating...")
    # Generate scores for the 5 GRBAS parameters
    target_adjectives = ["Grade", "Roughness", "Breathiness", "Asthenia", "Strain"]
    sim_scores = aligner.generate_scores(aligned_embeddings, target_adjectives)
    
    # Prepare results DataFrame
    scores_df = pd.DataFrame(sim_scores, columns=target_adjectives)
    scores_df['filename'] = labels_df['audio_file_name'].values
    
    # Step 5: Healthy-Voice Z-Score Normalization
    scaled_scores_df = zscore_normalize(scores_df, labels_df)
    
    # Step 6: Convergent Validity (Spearman Correlation)
    results = calculate_correlations(scaled_scores_df, labels_df)
    
    print("\nPROCRUSTES ALIGNMENT RESULTS (ZERO-SHOT)")
    print_metrics(results)
    
    return scaled_scores_df

if __name__ == "__main__":
    # Example paths (Update these for your local environment)
    AUDIO_DIR = "../data/audio" 
    METADATA_PATH = "../filenames/fold_all_train.csv"
    CHECKPOINT = None # Path to regression_best_grbas_ast_model_fold_all.pt
    
    if os.path.exists(METADATA_PATH):
        run_pipeline(AUDIO_DIR, METADATA_PATH, CHECKPOINT)
    else:
        print(f"Metadata file not found at {METADATA_PATH}. Please check paths.")
