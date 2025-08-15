# ---------------------------- External Imports ----------------------------
import torch

# ---------------------------- Biology Domain ----------------------------
def get_data(device='cpu'):
    """
    Returns a placeholder tensor representing biology knowledge/embedding.
    Replace with real embedding/data retrieval logic.
    """
    return torch.full((1, 512), 2.0).to(device)
