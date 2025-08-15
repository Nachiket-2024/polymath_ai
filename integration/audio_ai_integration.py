# ---------------------------- External Imports ----------------------------
import torch
import numpy as np
import whisper
from googletrans import Translator
import ffmpeg
import tempfile

# ---------------------------- Internal Imports ----------------------------
from ..ai_core.learning.self_evolving_ai import SelfEvolvingAI

# ---------------------------- AudioAIIntegration Class ----------------------------
class AudioAIIntegration:
    """
    Integrates Whisper transcription and translation with SelfEvolvingAI.
    Continuously adapts the AI model based on audio embeddings.
    """

    def __init__(self, device='cpu', dopamine_threshold=0.6, lr=0.001, fast_steps=5, max_len=50):
        """
        Initialize the audio-AI integration.
        Args:
            device (str): 'cpu' or 'cuda'
            dopamine_threshold (float): Threshold below which AI adapts
            lr (float): Learning rate for meta-learner
            fast_steps (int): Number of quick adaptation steps
            max_len (int): Length of audio embedding vector
        """
        self.device = device
        self.max_len = max_len

        # Initialize SelfEvolvingAI
        self.ai_model = SelfEvolvingAI(
            device=device,
            dopamine_threshold=dopamine_threshold,
            lr=lr,
            fast_steps=fast_steps
        ).to(device)

        # Load Whisper model
        self.whisper_model = whisper.load_model("base")
        self.translator = Translator()

    # ---------------------------- Audio Utilities ----------------------------
    def convert_to_wav(self, input_file):
        """Converts audio to WAV format using ffmpeg."""
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        ffmpeg.input(input_file).output(temp_wav.name).run(quiet=True, overwrite_output=True)
        return temp_wav.name

    def extract_audio_embedding(self, audio_file):
        """Simple embedding: uses MFCCs or a placeholder tensor."""
        try:
            # Convert audio to WAV if needed
            if not audio_file.endswith(".wav"):
                audio_file = self.convert_to_wav(audio_file)

            import librosa
            y, sr = librosa.load(audio_file, sr=22050)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            if mfcc.shape[1] < self.max_len:
                mfcc = np.pad(mfcc, ((0, 0), (0, self.max_len - mfcc.shape[1])), mode='constant')
            else:
                mfcc = mfcc[:, :self.max_len]
            embedding = torch.tensor(mfcc.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
            return embedding
        except Exception as e:
            print(f"Error extracting audio embedding: {e}")
            return torch.zeros(1, 20*self.max_len).to(self.device)

    def transcribe_audio(self, audio_file):
        """Transcribe audio using Whisper."""
        result = self.whisper_model.transcribe(audio_file, language="en")
        return result['text']

    def translate_text(self, text, target_language='hi'):
        """Translate text to target language."""
        return self.translator.translate(text, dest=target_language).text

    # ---------------------------- AI Adaptation ----------------------------
    def create_task_loader(self, embedding):
        """Wrap embedding into a DataLoader for meta-learning."""
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(embedding, torch.zeros(embedding.size(0), dtype=torch.long))
        loader = DataLoader(dataset, batch_size=1)
        return loader

    def process_audio(self, audio_file):
        """Process an audio file: extract embedding, adapt AI, get dopamine."""
        embedding = self.extract_audio_embedding(audio_file)
        task_loader = self.create_task_loader(embedding)
        embeddings, dopamine = self.ai_model(embedding, task_loader=task_loader)
        return embeddings, dopamine

    # ---------------------------- Main Interactive Flow ----------------------------
    def run(self):
        """Interactive loop to select audio file, transcribe, translate, and adapt AI."""
        import tkinter as tk
        from tkinter import filedialog
        import numpy as np

        root = tk.Tk()
        root.withdraw()

        audio_file = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.wma *.mkv *.opus")]
        )

        if not audio_file:
            print("No file selected. Exiting...")
            return

        print("Transcribing audio...")
        transcription = self.transcribe_audio(audio_file)
        print("Transcription:", transcription)

        print("Translating transcription to Hindi...")
        translation = self.translate_text(transcription, 'hi')
        print("Translation (Hindi):", translation)

        print("Processing AI adaptation...")
        _, dopamine = self.process_audio(audio_file)
        print(f"AI Dopamine signal after adaptation: {dopamine:.2f}")

# ---------------------------- Main Entry ----------------------------
if __name__ == "__main__":
    integrator = AudioAIIntegration(device='cpu')
    integrator.run()
