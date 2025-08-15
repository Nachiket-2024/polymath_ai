# ---------------------------- External Imports ----------------------------
import torch

# ---------------------------- Internal Imports ----------------------------
from . import physics
from . import biology
from . import math
from . import cs

# ---------------------------- DomainManager Class ----------------------------
class DomainManager:
    """
    Manages multiple knowledge domains and retrieves their embeddings/data.
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.domains = {
            'physics': physics,
            'biology': biology,
            'math': math,
            'cs': cs
        }

    def get_domain_data(self, domain_name):
        """
        Retrieve data tensor for the specified domain.

        Args:
            domain_name (str): Name of the domain (e.g., 'physics')

        Returns:
            torch.Tensor: Domain embedding tensor
        """
        domain = self.domains.get(domain_name)
        if domain is None:
            raise ValueError(f"Domain '{domain_name}' not found.")
        return domain.get_data(device=self.device)

    def combine_domains(self, domain_list):
        """
        Combine data from multiple domains into a single tensor.

        Args:
            domain_list (list of str): List of domain names to combine

        Returns:
            torch.Tensor: Combined tensor
        """
        combined = torch.zeros(1, 512).to(self.device)
        for name in domain_list:
            combined += self.get_domain_data(name)
        return combined
