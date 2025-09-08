import numpy as np
import librosa

def extract_features(file):
    """
    Extract MFCC features from an audio file.
    Returns mean MFCC feature vector.
    """
    audio, sr = librosa.load(file, res_type='kaiser_fast', duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)
