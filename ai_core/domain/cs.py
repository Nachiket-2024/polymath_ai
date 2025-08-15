# ---------------------------- External Imports ----------------------------
import torch

# ---------------------------- Computer Science Domain ----------------------------
def get_data(device='cpu'):
    """
    Returns a placeholder tensor representing CS knowledge/embedding.
    Replace with real embedding/data retrieval logic.
    """
    return torch.full((1, 512), 4.0).to(device)
