import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

def zscore_normalize(scores_df, ground_truth_df):
    """
    Normalizes Procrustes scores based on Grade 0 (Healthy) samples.
    """
    # Merge to identify Grade 0 samples
    merged = pd.merge(scores_df, ground_truth_df[['audio_file_name', 'Grade']], 
                      left_on='filename', right_on='audio_file_name')
    
    grade0_df = merged[merged['Grade'] == 0]
    
    # Adjective columns
    adj_cols = [col for col in scores_df.columns if col != 'filename']
    
    # Calculate stats from healthy voices
    stats_mean = grade0_df[adj_cols].mean()
    stats_std = grade0_df[adj_cols].std().replace(0, 1.0)
    
    # Apply Scaling
    scaled_df = scores_df.copy()
    for adj in adj_cols:
        scaled_df[adj] = (scores_df[adj] - stats_mean[adj]) / stats_std[adj]
        
    return scaled_df

def calculate_correlations(predictions_df, labels_df):
    """
    Calculates Spearman correlations between aligned scores and human ratings.
    """
    merged = pd.merge(predictions_df, labels_df, left_on='filename', right_on='audio_file_name', suffixes=('_pred', '_truth'))
    
    grbas_params = ["Grade", "Roughness", "Breathiness", "Asthenia", "Strain"]
    results = []
    
    for param in grbas_params:
        # Match case if necessary (e.g., 'Grade' vs 'grade')
        pred_col = param
        truth_col = param + "_truth"
        
        if pred_col in merged.columns and truth_col in merged.columns:
            corr, p_val = spearmanr(merged[pred_col], merged[truth_col])
            mae = mean_absolute_error(merged[truth_col], merged[pred_col])
            
            results.append({
                "Parameter": param,
                "Spearman_rho": corr,
                "p-value": p_val,
                "MAE": mae
            })
            
    return pd.DataFrame(results)

def print_metrics(results_df):
    """Utility to print results in a clean table."""
    print("\n" + "="*60)
    print(f"{'GRBAS Parameter':<15} | {'Correlation (rs)':^18} | {'p-value':^10}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"{row['Parameter']:<15} | {row['Spearman_rho']:^18.4f} | {row['p-value']:.2e}")
    print("="*60 + "\n")
