import os
import pickle
import numpy as np
import librosa
import tempfile
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model and label encoder
with open("speaker_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Function to convert audio to WAV format
def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(temp_wav.name, format="wav")
    return temp_wav.name

# Function to extract MFCC features
def extract_features(file_path, max_len=50):
    if not file_path.endswith(".wav"):
        file_path = convert_to_wav(file_path)
    
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.flatten()

# Flask routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    feature = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(feature)
    speaker_name = le.inverse_transform(prediction)[0]
    
    return jsonify({"predicted_speaker": speaker_name})

if __name__ == '__main__':
    app.run(debug=True)
