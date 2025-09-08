import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from src.feature_extraction import extract_features

# Map emotion IDs from filename to labels
EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fear",
    7: "disgust",
    8: "surprise"
}

def load_dataset(data_path):
    features, labels = [], []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                emotion_id = int(file.split("-")[2])
                label = EMOTION_MAP.get(emotion_id, "unknown")
                features.append(extract_features(file_path))
                labels.append(label)

    X = np.array(features)
    y = np.array(labels)

    le = LabelEncoder()
    y = to_categorical(le.fit_transform(y))

    return X, y, le

def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
