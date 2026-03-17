# Procrustes Alignment for Zero-Shot Clinical Voice Assessment (GRBAS)

This repository contains the **Minimal Reproducible Example (MRE)** for the methodology presented in our research paper. Our approach uses **Procrustes Alignment** to map high-dimensional audio embeddings (from an Audio Spectrogram Transformer) into a semantic medical text space for zero-shot clinical voice pathology scoring.

## Methodology Overview

We propose a cross-modal alignment framework that bridges the gap between deep learning acoustic features and human clinical perception (the GRBAS scale):
1.  **Audio Space:** 768-dimensional embeddings extracted from the `[CLS]` token of a fine-tuned **Audio Spectrogram Transformer (AST)**.
2.  **Text Space:** 768-dimensional embeddings from a **SentenceTransformer (`all-mpnet-base-v2`)** representing medically-precise vocal descriptors.
3.  **Alignment:** An **Orthogonal Procrustes** rotation that aligns the audio feature directions with the clinical text axes, centered on a "healthy-voice" origin.
4.  **Zero-Shot Scoring:** Calculating the cosine similarity between aligned patient audio and target clinical adjectives (Grade, Roughness, Breathiness, Asthenia, Strain), and the adjectives.

## Repository Structure

The code is organized into modular components for clarity and reusability:

-   `dataset.py`: Handles audio preprocessing (16kHz, Mel-spectrograms) and feature extraction parameters.
-   `model.py`: Defines the AST architecture and the logic for extracting 768-d audio embeddings.
-   `alignment.py`: **The Core Engine.** Contains the medically-precise text anchors, the Procrustes rotation solver, and the healthy-voice origin shifting logic.
-   `evaluate.py`: Implements Z-score normalization (relative to healthy controls) and Spearman Rank Correlation metrics.
-   `run_mre.py`: The main orchestration script to run the entire pipeline from end to end.

## Getting Started

### 1. Installation

Ensure you have Python 3.8+ and install the required dependencies:

```bash
pip install torch transformers librosa sentence-transformers scipy pandas numpy scikit-learn
```

### 2. Data Preparation

To run the MRE, you will need:
-   **Audio Files:** Path to your patient recordings (WAV format recommended).
-   **Metadata CSV:** A file (e.g., `labels.csv`) containing at least the following columns:
    -   `audio_file_name`: Filename of the recording.
    -   `Grade`, `Roughness`, `Breathiness`, `Asthenia`, `Strain`: Human expert ratings (0-3 scale).

### 3. Running the Example

Update the paths in `run_mre.py` and execute the script:

```bash
python run_mre.py
```

The script will:
1.  Preprocess the audio and extract AST embeddings.
2.  Generate clinical text anchors.
3.  Calculate and apply the Procrustes rotation.
4.  Normalize the scores and output the **Spearman Correlation** coefficients against the ground truth.

## Citation

If you use this code or methodology in your research, please cite our manuscript:

> placeholder

---
**Note:** This MRE is designed for scientific transparency. For the datasets or other data, please refer to the data availability statement in the original manuscript.
