# ---------------------------- External Imports ----------------------------
import torch

# ---------------------------- Utility Functions ----------------------------
def get_domain_data(domain_name, device='cpu'):
    """
    Retrieve example data for a given domain.
    Placeholder method; implement actual data retrieval logic.

    Args:
        domain_name (str): Name of the domain
        device (str): Device to place tensor on

    Returns:
        torch.Tensor: Example tensor for domain
    """
    return torch.zeros(1, 512).to(device)


def combine_data(data1, data2):
    """
    Combine two datasets or embeddings for cross-domain reasoning.

    Args:
        data1, data2 (torch.Tensor): Input tensors to combine

    Returns:
        torch.Tensor: Combined tensor
    """
    return data1 + data2
