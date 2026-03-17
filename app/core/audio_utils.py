import librosa
import numpy as np

def get_speech_timestamps(audio_path: str, top_db: int = 30):
    """
    Detects the start and end of speech in an audio file.
    Returns timestamps in seconds.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # librosa.effects.trim returns the indices of the non-silent part
        non_silent_indices = librosa.effects.trim(y, top_db=top_db)[1]
        
        # If the array is empty, it means the audio is all silence.
        if non_silent_indices.size == 0:
            return {"start": 0, "end": librosa.get_duration(y=y, sr=sr)}

        start_index, end_index = non_silent_indices[0], non_silent_indices[1]
        
        # Convert indices to time
        start_time = librosa.samples_to_time(start_index, sr=sr)
        end_time = librosa.samples_to_time(end_index, sr=sr)
        
        return {"start": start_time, "end": end_time}
    except Exception as e:
        # If any librosa error occurs, return the full duration
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=False) # Try again without forcing mono
            duration = librosa.get_duration(y=y, sr=sr)
        except Exception:
            duration = 0 # Cannot determine duration
        return {"start": 0, "end": duration, "error": str(e)}
