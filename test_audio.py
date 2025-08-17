# ---------------------------- External Imports ----------------------------
from transformers import BertTokenizer

# ---------------------------- Internal Imports ----------------------------
from ai_core.learning.self_evolving_ai import SelfEvolvingAI

# ---------------------------- Config ----------------------------
audio_file = "audio_recognition/dataset/other/exercise_bike.wav"
device = 'cpu'

# ---------------------------- Initialize Model ----------------------------
model = SelfEvolvingAI(device=device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ---------------------------- Step 1: Load/Transcribe Audio ----------------------------
# For now, we assume the transcription is available in a text variable
# Replace this with actual transcription if you integrate Whisper
transcription = "exercise bike sound"  # Example placeholder text

# ---------------------------- Step 2: Tokenize for BERT ----------------------------
text_input_dict = tokenizer(
    [transcription],  # single-item batch
    padding=True,
    truncation=True,
    return_tensors="pt"
)
text_input_dict = dict(text_input_dict)  # Ensure plain dict

# ---------------------------- Step 3: Forward Pass ----------------------------
embeddings, dopamine = model.core(text_input=text_input_dict, audio_input=None)

# ---------------------------- Step 4: Print Results ----------------------------
print("Audio file:", audio_file)
print("Transcription:", transcription)
print("Embedding shape:", embeddings.shape)
print("Dopamine signal:", dopamine)
