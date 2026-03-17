import parselmouth
import numpy as np
import os

def analyze_praat_features(wav_file_path: str):
    """
    Analyzes a single .wav file using Parselmouth and returns a dictionary of metrics.
    """
    if not os.path.exists(wav_file_path):
        return {"error": "File not found"}

    try:
        snd = parselmouth.Sound(wav_file_path)
    except Exception as e:
        return {"error": f"Error loading sound: {str(e)}"}

    pitch_floor = 75.0
    pitch_ceiling = 600.0
    results = {}

    try:
        # 1. F0 Statistics
        pitch = snd.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
        results['mean_F0_Hz'] = float(parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz"))
        results['std_dev_F0_Hz'] = float(parselmouth.praat.call(pitch, "Get standard deviation", 0, 0, "Hertz"))

        # 2. Jitter/Shimmer
        point_process = parselmouth.praat.call(pitch, "To PointProcess")
        results['jitter_local'] = float(parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)) * 100
        results['shimmer_local_db'] = float(parselmouth.praat.call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

        # 3. HNR
        harmonicity = snd.to_harmonicity(time_step=0.01, minimum_pitch=pitch_floor, silence_threshold=0.1, periods_per_window=1.0)
        hnr_mean_dB = float(parselmouth.praat.call(harmonicity, "Get mean", 0, 0))
        results['hnr_mean_dB'] = hnr_mean_dB
        intensity = snd.to_intensity(minimum_pitch=pitch_floor, time_step=0.01)

        if hnr_mean_dB == -np.inf:
            nhr_ratio = np.nan
        else:
            nhr_ratio = 1 / (10**(hnr_mean_dB / 10))
        results['nhr_ratio'] = nhr_ratio

        # 4. CPPS (Cepstral Peak Prominence)
        spectrum = snd.to_spectrum()
        cepstrum = parselmouth.praat.call(spectrum, "To PowerCepstrum")
        results['cpps_mean'] = float(parselmouth.praat.call(cepstrum, "Get peak prominence", pitch_floor, pitch_ceiling, "parabolic", 0.001, 0.05, "straight", "robust slow"))

        return results

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}