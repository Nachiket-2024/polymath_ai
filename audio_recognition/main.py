import numpy as np
import librosa
import os
import pickle
from tqdm import tqdm
from pydub import AudioSegment
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tempfile

# Function to convert audio to WAV format
def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")  # Temporary WAV file
    audio.export(temp_wav.name, format="wav")
    return temp_wav.name  # Return converted file path

# Function to extract MFCC features from an audio file
def extract_features(file_path, max_len=50):
    # Convert non-WAV files to WAV
    if not file_path.endswith(".wav"):
        file_path = convert_to_wav(file_path)
    
    y, sr = librosa.load(file_path, sr=22050)  # Load audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Extract MFCCs
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.flatten()  # Flatten to make it a single vector

# Load dataset (folder structure: dataset/speaker_name/audio_file)
dataset_path = "dataset/"
speakers = []
features = []

print("Extracting features from audio files...")

for speaker in tqdm(os.listdir(dataset_path), desc="Processing Speakers"):
    speaker_path = os.path.join(dataset_path, speaker)
    if os.path.isdir(speaker_path):
        for audio_file in tqdm(os.listdir(speaker_path), desc=f"Processing {speaker}", leave=False):
            file_path = os.path.join(speaker_path, audio_file)
            feature = extract_features(file_path)
            features.append(feature)
            speakers.append(speaker)

# Encode speaker labels
le = LabelEncoder()
labels = le.fit_transform(speakers)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train SVM model
print("\nTraining model...")
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
print("Model trained successfully!")

# Save model & label encoder
with open("speaker_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model and label encoder saved.")

# Function to recognize speaker
def recognize_speaker(file_path):
    with open("speaker_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    
    print("\nExtracting features from test audio...")
    feature = extract_features(file_path).reshape(1, -1)
    
    print("Predicting speaker...")
    prediction = model.predict(feature)
    speaker_name = le.inverse_transform(prediction)[0]
    
    return speaker_name

# Test with a new audio file (supports wav, weba, mp3, flac, etc.)
test_audio = "test_audio.weba"
predicted_speaker = recognize_speaker(test_audio)
print(f"\n🎤 Predicted Speaker: {predicted_speaker}")
