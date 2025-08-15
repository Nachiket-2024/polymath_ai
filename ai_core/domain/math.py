# ---------------------------- External Imports ----------------------------
import torch

# ---------------------------- Math Domain ----------------------------
def get_data(device='cpu'):
    """
    Returns a placeholder tensor representing math knowledge/embedding.
    Replace with real embedding/data retrieval logic.
    """
    return torch.full((1, 512), 3.0).to(device)
