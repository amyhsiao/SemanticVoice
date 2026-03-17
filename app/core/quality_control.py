import librosa
import numpy as np

MIN_DURATION = 0.5
MAX_DURATION = 30  # Max duration of 30s
CLIPPING_THRESHOLD_RATIO = 0.01 # More than 1% of samples are clipped
CLIPPING_AMPLITUDE_THRESHOLD = 0.98 # Amplitude considered as clipping
SILENCE_THRESHOLD_DB = -50 # Adjusted for typical speech
SNR_THRESHOLD_DB = 10 # A reasonable default for speech

def run_quality_checks(audio_path: str, max_duration: float = MAX_DURATION):
    """
    Runs all quality checks on the audio file and yields the result of each check.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        yield {"check": "File Read", "status": "failed", "message": f"Could not read audio file: {e}"}
        return

    # 1. Silence Check (most basic)
    check_name = "Basic Silence Check"
    if np.max(np.abs(y)) < 0.001:
        yield {"check": check_name, "status": "failed", "message": "Audio appears to be silent. No speech detected."}
        return
    yield {"check": check_name, "status": "passed", "message": "Basic silence check passed."}

    # 2. Duration Check
    check_name = "Duration Check"
    duration = librosa.get_duration(y=y, sr=sr)
    if duration < MIN_DURATION:
        yield {"check": check_name, "status": "failed", "message": f"Audio is too short. Minimum duration is {MIN_DURATION}s."}
        return
    if duration > max_duration:
        yield {"check": check_name, "status": "failed", "message": f"Audio is too long. Maximum duration is {max_duration}s."}
        return
    yield {"check": check_name, "status": "passed", "message": "Duration check passed."}

    # 3. Quietness Check
    check_name = "Quietness Check"
    try:
        rms = librosa.feature.rms(y=y)
        # Filter out zero RMS values to avoid issues with log of zero
        rms_filtered = rms[rms > 0]
        if rms_filtered.size == 0:
            yield {"check": check_name, "status": "failed", "message": "Audio is completely silent or contains no energy."}
            return
        
        avg_db = librosa.amplitude_to_db(np.mean(rms_filtered), ref=1.0)
        
        if avg_db < SILENCE_THRESHOLD_DB:
            yield {"check": check_name, "status": "failed", "message": f"Audio is too quiet. Average level is {avg_db:.2f} dB, which is below the {SILENCE_THRESHOLD_DB} dB threshold."}
            return
        yield {"check": check_name, "status": "passed", "message": "Quietness check passed."}
    except Exception as e:
        yield {"check": check_name, "status": "failed", "message": f"An error occurred during quietness check: {e}"}
        return


    # 4. Clipping Detection
    check_name = "Clipping Detection"
    num_clipping_samples = np.sum(np.abs(y) >= CLIPPING_AMPLITUDE_THRESHOLD)
    clipping_ratio = num_clipping_samples / len(y)
    if clipping_ratio > CLIPPING_THRESHOLD_RATIO:
         yield {"check": check_name, "status": "failed", "message": f"Audio is clipping or distorted. Please reduce your recording level."}
         return
    yield {"check": check_name, "status": "passed", "message": "Clipping detection passed."}

    # 5. SNR Estimation (Simplified) - TODO: This is unreliable, needs a better implementation.
    # ... (code remains commented out) ...

