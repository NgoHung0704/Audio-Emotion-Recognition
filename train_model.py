import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

DATA_DIR = "data/"
FEATURE_SIZE = 34  # Adjust based on the number of features extracted


# Functions
# Visualize the distribution of emotions
def plot_emotion_distribution(df):
    plt.title("Distribution of Emotions")
    sns.countplot(x="Emotions", data=df, palette="Set2", legend=False)
    plt.xlabel("Emotions")
    plt.ylabel("Count")
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()


# take a wav file as an example and mel frequency ceptral coefficients
def create_waveplot(file_path):
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


def create_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", fmax=8000)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-frequency spectrogram")
    plt.colorbar()
    plt.show()


def create_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.mfcc(y=y, sr=sr, n_mels=128, fmax=8000)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", fmax=8000)
    plt.colorbar(format="%+2.0f dB")
    plt.title("MFCC Frequency Spectrogram")
    plt.colorbar()
    plt.show()


# Extract features from audio files
def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13 MFCCs
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # 12 chroma features
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr
    )  # 7 spectral contrast features
    rms = librosa.feature.rms(y=y)  # Root Mean Square Energy
    zcr = librosa.feature.zero_crossing_rate(y)  # Zero Crossing Rate
    features = np.hstack(
        (
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(contrast, axis=1),
            np.mean(rms, axis=1),
            np.mean(zcr, axis=1),
        )
    )

    return features  # (34 features in total, adjust as necessary based on your feature extraction method)


# =========================================================#
if __name__ == "__main__":
    file_path = []
    file_emotion = []

    # Check if the data directory exists
    for dir in os.listdir(DATA_DIR):
        for file in os.listdir(DATA_DIR + dir):
            if file.endswith(".wav"):
                label = file.split("-")[2]  # Extract emotion label from filename
                file_path.append(DATA_DIR + dir + "/" + file)
                file_emotion.append(int(label))

    emotion_df = pd.DataFrame(file_emotion, columns=["Emotions"])
    path_df = pd.DataFrame(file_path, columns=["file_path"])

    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    Ravdess_df["Emotions"] = Ravdess_df["Emotions"].replace(
        {
            1: "neutral",
            2: "calm",
            3: "happy",
            4: "sad",
            5: "angry",
            6: "fearful",
            7: "disgust",
            8: "surprised",
        },
    )
    Ravdess_df = Ravdess_df.sample(frac=1).reset_index(
        drop=True
    )  # Shuffle the DataFrame

    # Ravdess_df.head()

    # plot_emotion_distribution(Ravdess_df)

    # # Create waveplot and spectrogram for a sample audio file
    # # Example path for happy emotion
    # path = Ravdess_df["file_path"][Ravdess_df.Emotions == "happy"].iloc[0]
    # create_waveplot(path)
    # create_spectrogram(path)
    # create_mfcc(path)

    # Prepare the dataset
    n = Ravdess_df.shape[0]

    X = np.zeros(
        (n, FEATURE_SIZE)
    )  # Adjusted to match the number of features extracted
    y = np.empty(n, dtype=str)
    for index, row in Ravdess_df.iterrows():
        print(f"Processing file: {row['file_path']}")
        features = extract_features(row["file_path"])
        X[index] = features
        y[index] = row["Emotions"]
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    # Save the model and scaler
    joblib.dump(model, "emotion_recognition_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    # Load the model and scaler for future use
    # model = joblib.load("emotion_recognition_model.pkl")
    # scaler = joblib.load("scaler.pkl")
    # Example usage of the loaded model
    # def predict_emotion(file_path):
    #     features = extract_features(file_path)
    #     features = scaler.transform([features])
    #     prediction = model.predict(features)
    #     return prediction[0]
    # Example prediction
    # emotion = predict_emotion("path_to_audio_file.wav")
    # print(f"Predicted emotion: {emotion}")
    # Note: Uncomment the example usage and prediction code to test with a specific audio file.
    # The code above is complete and ready to run. It includes data loading, feature extraction, model training, evaluation, and saving the model.
    # Make sure to have the necessary libraries installed and the audio files available in the specified directory.
