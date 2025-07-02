import os
import numpy as np
import pandas as pd
import librosa
import joblib
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from tqdm import tqdm
import hashlib
import pickle
from typing import List


# Constants
DATA_DIR = "data/"
SAMPLE_RATE = 22050  # Reduced from default 44.1kHz for faster processing
CACHE_DIR = "feature_cache/"
N_WORKERS = os.cpu_count() # Number of parallel threads for feature extraction
N_COMPONENTS = 30  # Number of PCA components
RANDOM_STATE = 10000 # For reproducibility 
TEST_SIZE = 0.2

# Emotion mapping
EMOTION_MAP = {
    1: "neutral", 2: "calm", 3: "happy",
    4: "sad", 5: "angry", 6: "fearful",
    7: "disgust", 8: "surprised"
}

# Create cache directory if not exists
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(file_path: str) -> str:
    """Generate unique cache path for audio file"""
    file_hash = hashlib.md5(file_path.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{file_hash}.pkl")

def extract_features(file_path: str) -> np.ndarray:
    """
    Extract high-impact features for emotion recognition with caching.
    Focuses on features that maximize F1-score, even if computationally expensive.
    Returns 45-dimensional feature vector.
    """
    cache_path = get_cache_path(file_path)
    
    # Return cached features if available
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    try:
        # Load audio with optimal parameters for emotion recognition
        y, sr = librosa.load(file_path, sr=22050, duration=3.0)  # Focus on first 3 seconds
        
        # Fundamental audio features
        S = np.abs(librosa.stft(y))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        
        # 1. Enhanced MFCCs with derivatives (21 features)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_deltas = librosa.feature.delta(mfccs)
        mfcc_delta_deltas = librosa.feature.delta(mfccs, order=2)
        
        # 2. Pitch and harmonic features (5 features)
        pitch, _, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
        pitch = np.nan_to_num(pitch, nan=0.0)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        hnr = harmonic / (percussive + 1e-6)  # Harmonic-to-noise ratio
        
        # 3. Spectral descriptors (7 features)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)
        contrast = librosa.feature.spectral_contrast(S=S_db, sr=sr)
        
        # 4. Temporal features (5 features)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        
        # 5. Emotion-specific features (7 features)
        chroma = librosa.feature.chroma_stft(S=S_db, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        poly_features = librosa.feature.poly_features(S=S_db, sr=sr)
        
        # Create comprehensive feature vector
        features = np.concatenate([
            # MFCCs and derivatives
            np.mean(mfccs, axis=1),
            np.mean(mfcc_deltas, axis=1),
            np.mean(mfcc_delta_deltas, axis=1),
            
            # Pitch and harmonic
            [np.mean(pitch), np.std(pitch), np.max(pitch)],
            [np.mean(hnr), np.max(harmonic)],
            
            # Spectral
            [np.mean(centroid), np.mean(bandwidth), np.mean(rolloff)],
            [np.mean(flatness)],
            np.mean(contrast, axis=1),
            
            # Temporal
            [np.mean(zcr), np.mean(rms), np.std(rms)],
            [np.mean(tempogram)],
            
            # Emotion-specific
            np.mean(chroma, axis=1),
            np.mean(tonnetz, axis=1),
            np.mean(poly_features, axis=1)
        ])
        
        # Cache the features
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
            
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return np.zeros(45)  # Return zero array if error occurs

def extract_features_parallel(file_paths: List[str]) -> np.ndarray:
    """Parallel feature extraction with progress bar"""
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        features = list(tqdm(executor.map(extract_features, file_paths),
                           total=len(file_paths),
                           desc="Extracting features"))
    return np.vstack(features)

def load_dataset() -> pd.DataFrame:
    """Load dataset with emotion labels"""
    file_paths = []
    emotions = []
    
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".wav"):
                try:
                    label = int(file.split("-")[2])
                    file_paths.append(os.path.join(root, file))
                    emotions.append(EMOTION_MAP[label])
                except (IndexError, ValueError):
                    continue
    
    return pd.DataFrame({"Path": file_paths, "Emotion": emotions})

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> HistGradientBoostingClassifier:
    """Train a faster gradient boosting model"""
    model = HistGradientBoostingClassifier(
        max_iter=100,
        early_stopping=True,
        random_state=RANDOM_STATE,
        validation_fraction=0.2,
        verbose=1
    )
    model.fit(X_train, y_train)
    return model

def main():
    print("Loading dataset...")
    df = load_dataset().sample(frac=1, random_state=RANDOM_STATE)
    
    print(f"\nFound {len(df)} audio files")
    print("Emotion distribution:")
    print(df["Emotion"].value_counts())
    
    print("\nExtracting features (parallel processing)...")
    X = extract_features_parallel(df["Path"].tolist())
    y = df["Emotion"].values
    
    print("\nApplying dimensionality reduction...")
    pca = PCA(n_components=N_COMPONENTS)
    X_reduced = pca.fit_transform(X)
    
    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print("\nSaving model artifacts...")
    joblib.dump(model, "emotion_model.pkl")
    joblib.dump(pca, "pca_transformer.pkl")
    print("Done!")
#################################################### đọc 38, 178, 180, 195, 196
if __name__ == "__main__":
    main()