# ---------------------------- External Imports ----------------------------
from transformers import AutoModel
import torch
import torch.nn as nn

# ---------------------------- MultimodalFusion Class ----------------------------
class MultimodalFusion(nn.Module):
    """
    Integrates embeddings from text, audio, and vision modalities into a fused representation.
    """

    def __init__(self, text_model_name="bert-base-uncased", 
                 audio_model=None, vision_model=None, 
                 fusion_weights=None):
        """
        Initialize the multimodal fusion module.

        Args:
            text_model_name (str): Pretrained text model name.
            audio_model (nn.Module): Pretrained audio model.
            vision_model (nn.Module): Pretrained vision model.
            fusion_weights (list or tuple): Weights for [text, audio, vision] fusion.
        """
        super().__init__()

        # Load pretrained text model
        self.text_model = AutoModel.from_pretrained(text_model_name)

        # Expect pretrained audio and vision models passed in
        self.audio_model = audio_model
        self.vision_model = vision_model

        # Default equal weights if none provided
        if fusion_weights is None:
            fusion_weights = [1.0, 1.0, 1.0]
        self.register_buffer("fusion_weights", torch.tensor(fusion_weights))

    def forward(self, text_input, audio_input=None, vision_input=None):
        """
        Forward pass for multimodal fusion.

        Args:
            text_input (torch.Tensor or dict): Either a precomputed embedding tensor
                or a tokenized BERT input dict (input_ids, attention_mask, etc.).
            audio_input (torch.Tensor): Audio input tensor (optional).
            vision_input (torch.Tensor): Vision input tensor (optional).

        Returns:
            torch.Tensor: Fused embedding vector.
        """
        # ---------------- Step 1: Handle text embeddings ----------------
        if isinstance(text_input, torch.Tensor):
            # Already an embedding tensor
            text_emb = text_input
        elif isinstance(text_input, dict):
            # Tokenized input for BERT
            text_emb = self.text_model(**text_input).pooler_output
        else:
            raise ValueError("text_input must be a torch.Tensor or tokenized dict for BERT")

        # ---------------- Step 2: Handle audio embeddings ----------------
        if self.audio_model is not None and audio_input is not None:
            audio_emb = self.audio_model(audio_input).embeddings
        else:
            audio_emb = torch.zeros_like(text_emb)

        # ---------------- Step 3: Handle vision embeddings ----------------
        if self.vision_model is not None and vision_input is not None:
            vision_emb = self.vision_model(vision_input).features
        else:
            vision_emb = torch.zeros_like(text_emb)

        # ---------------- Step 4: Weighted sum fusion ----------------
        fused = (
            self.fusion_weights[0] * text_emb +
            self.fusion_weights[1] * audio_emb +
            self.fusion_weights[2] * vision_emb
        )

        return fused
