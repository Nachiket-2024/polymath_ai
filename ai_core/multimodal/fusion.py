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
            text_input (torch.Tensor): Input IDs or embeddings for text.
            audio_input (torch.Tensor): Audio input tensor (optional).
            vision_input (torch.Tensor): Vision input tensor (optional).

        Returns:
            torch.Tensor: Fused embedding vector.
        """
        # Get text embeddings (using pooler output)
        text_emb = self.text_model(**text_input).pooler_output  # (batch_size, hidden_dim)

        # Process audio and vision if provided, else zero tensor
        audio_emb = (
            self.audio_model(audio_input).embeddings if self.audio_model and audio_input is not None
            else torch.zeros_like(text_emb)
        )
        vision_emb = (
            self.vision_model(vision_input).features if self.vision_model and vision_input is not None
            else torch.zeros_like(text_emb)
        )

        # Weighted sum fusion
        fused = (
            self.fusion_weights[0] * text_emb +
            self.fusion_weights[1] * audio_emb +
            self.fusion_weights[2] * vision_emb
        )

        return fused
