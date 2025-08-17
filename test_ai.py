# ---------------------------- External Imports ----------------------------
import os
import torch
import soundfile as sf
import numpy as np
from transformers import BertTokenizer

# ---------------------------- Internal Imports ----------------------------
from ai_core.learning.self_evolving_ai import SelfEvolvingAI

# ---------------------------- Step 1: Ensure Dummy Embeddings ----------------------------
embedding_path = "ai_core/embeddings/dummy_embeddings.txt"
os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
if not os.path.exists(embedding_path):
    with open(embedding_path, "w", encoding="utf-8") as f:
        f.write("test 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0\n")
        f.write("hello 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1\n")
    print(f"Created dummy text embeddings at {embedding_path}")
else:
    print(f"Dummy embeddings already exist at {embedding_path}")

# ---------------------------- Step 2: Ensure Dummy Audio ----------------------------
audio_dir = "audio_recog/uploads"
os.makedirs(audio_dir, exist_ok=True)
dummy_wav = os.path.join(audio_dir, "test_audio.wav")
if not os.path.exists(dummy_wav):
    sr = 48000
    t = np.linspace(0, 1, sr)
    wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(dummy_wav, wave, sr)
    print(f"Created dummy WAV file at {dummy_wav}")
else:
    print(f"WAV file already exists: {dummy_wav}")

# ---------------------------- Step 3: Load Audio ----------------------------
data, sr = sf.read(dummy_wav)
print(f"Loaded audio: {dummy_wav}, {len(data)} samples, {sr} Hz")
# Use dummy MFCC embedding to simulate audio features
mfcc_embedding = torch.rand(1, 13)
print("MFCC embedding shape:", mfcc_embedding.shape)

# ---------------------------- Step 4: Initialize Model ----------------------------
model = SelfEvolvingAI(device='cpu')

# ---------------------------- Step 5: Prepare Dummy Text Input ----------------------------
text_input = ["test", "hello", "world"]
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize and return PyTorch tensors
text_input_dict = tokenizer(
    text_input,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
# Convert from BatchEncoding to plain dictionary for AICore
text_input_dict = dict(text_input_dict)

# ---------------------------- Step 6: Forward Pass ----------------------------
output, dopamine = model.core(text_input=text_input_dict, audio_input=mfcc_embedding)

# ---------------------------- Step 7: Print Results ----------------------------
print("Model output embedding shape:", output.shape)
print("Dopamine signal:", dopamine)
