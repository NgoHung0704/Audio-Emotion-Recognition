import os
import numpy as np
import pandas as pd
import librosa
import joblib
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import hashlib
import pickle



# Constants
DATA_DIR = "data/"

CACHE_DIR = "feature_cache/"
N_WORKERS = os.cpu_count() # Number of parallel threads for feature extraction

RANDOM_STATE = 10000 # For reproducibility 
TEST_SIZE = 0.2

# Emotion mapping
EMOTION_MAP = {
    1: "neutral", 2: "calm", 3: "happy",
    4: "sad", 5: "angry", 6: "fearful",
    7: "disgust", 8: "surprised"
}

# Create a cache directory if not exists
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(file_path: str) -> str:
    """Generate unique cache path for audio file"""
    file_hash = hashlib.md5(file_path.encode()).hexdigest() #???
    return os.path.join(CACHE_DIR, f"{file_hash}.pkl") #pickle???