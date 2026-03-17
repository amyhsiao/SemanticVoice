# Audio Analyzer: Voice Quality Assessment System

An advanced automated voice quality assessment system that combines traditional acoustic analysis with state-of-the-art deep learning for clinical and research applications. This tool provides comprehensive metrics for voice analysis, focusing on the GRBAS (Grade, Roughness, Breathiness, Asthenia, Strain) scale, and extending to 62 voice-describing adjectives.

## Research Disclosure & Disclaimer

**IMPORTANT:** This software is intended strictly for research and educational purposes. It is **not** a medical device and has not been cleared or approved by any regulatory authority (such as the FDA or EMA) for clinical use. The analysis results provided by this system should not be used for medical diagnosis, clinical decision-making, or as a substitute for professional medical advice. All data and scores are experimental and should be interpreted within a research context.

---

## Key Features

### 1. Dual-Engine Analysis
- **Acoustic Engine (Praat):** Extracts high-precision classical metrics including:
  - **Fundamental Frequency (F0):** Mean and standard deviation.
  - **Perturbation Measures:** Jitter (local) and Shimmer (local_dB).
  - **Noise-to-Harmonics:** HNR (Harmonics-to-Noise Ratio) and NHR (Noise-to-Harmonics Ratio).
  - **Cepstral Analysis:** CPPS (Cepstral Peak Prominence - Smoothed) for dysphonia severity.
- **Deep Learning Engine (AST):** Utilizes a fine-tuned **Audio Spectrogram Transformer (MIT/ast)** to map audio features into a semantic space.

### 2. Procrustes Alignment
A novel approach that aligns audio embeddings with medically precise text-based descriptors. 
- **Cross-Modal Mapping:** Uses Procrustes rotation to bridge the gap between acoustic features and clinical terminology.
- **Semantic Scoring:** Provides Z-score normalized similarity scores for the GRBAS parameters.

### 3. Integrated Audio Pipeline
- **Quality Control:** Automatic checks for signal-to-noise ratio (SNR), duration, and audio clipping.
- **Speech Detection:** Automatic timestamping of speech segments.
- **Preprocessing:** 16kHz resampling, RMS normalization, and Mel Spectrogram generation (128 bins).

---

## Project Structure

```text
├── core/
│   ├── ast_processor.py      # AST Model & Procrustes alignment logic
│   ├── praat_processor.py    # Parselmouth/Praat feature extraction
│   ├── quality_control.py    # Audio validation and QC checks
│   ├── audio_utils.py        # Speech detection and FFmpeg utilities
├── models/
│   ├── *.pt                  # Fine-tuned AST model weights
│   ├── *.npy                 # Procrustes rotation matrices and origins
│   └── *.json                # Anchor descriptors and scaler parameters
├── static/                   # Web interface (HTML/CSS/JS)
├── main.py                   # FastAPI Application entry point
└── requirements.txt          # Project dependencies
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- **FFmpeg** (Required for audio processing)
  ```bash
  # macOS
  brew install ffmpeg
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  ```

### Local Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/audio-analyzer.git
   cd audio-analyzer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

To start the FastAPI server with auto-reload enabled:

```bash
source venv/bin/activate
uvicorn main:app --reload
```

The application will be available at: `http://127.0.0.1:8000`

---

## Technical Details
- **Sample Rate:** 16,000 Hz (Mono)
- **Model Architecture:** Audio Spectrogram Transformer (AST)
- **Feature Space:** 768-dimensional embeddings aligned to clinical descriptors.
- **Normalization:** Z-score scaling based on healthy voice (Grade 0) statistics.
