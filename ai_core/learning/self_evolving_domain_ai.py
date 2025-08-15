# ---------------------------- External Imports ----------------------------
import torch
import torch.nn as nn

# ---------------------------- Internal Imports ----------------------------
from .self_evolving_ai import SelfEvolvingAI
from ..domain.domain_manager import DomainManager

# ---------------------------- SelfEvolvingDomainAI Class ----------------------------
class SelfEvolvingDomainAI(nn.Module):
    """
    Extends SelfEvolvingAI to integrate domain-specific knowledge via DomainManager.
    Allows multi-domain reasoning and adaptation.
    """

    def __init__(self, device='cpu', dopamine_threshold=0.6, lr=0.001, fast_steps=5):
        super().__init__()
        self.device = device

        # Core self-evolving AI
        self.self_evolving_ai = SelfEvolvingAI(device=device,
                                              dopamine_threshold=dopamine_threshold,
                                              lr=lr,
                                              fast_steps=fast_steps).to(device)

        # Domain manager for cross-domain embeddings
        self.domain_manager = DomainManager(device=device)

    def forward(self, domains=None, task_loader=None, audio_input=None, vision_input=None, graph_data=None):
        """
        Forward pass with domain integration.

        Args:
            domains (list of str): List of domain names to include (e.g., ['physics', 'math'])
            task_loader: Optional DataLoader for adaptation tasks
            audio_input: Optional audio input
            vision_input: Optional vision input
            graph_data: Optional graph data for GNN

        Returns:
            tuple: (embedding tensor, dopamine signal)
        """
        if domains is None:
            domains = []

        # Step 1: Combine selected domain embeddings
        if domains:
            domain_emb = self.domain_manager.combine_domains(domains)
        else:
            # Default zero tensor if no domains provided
            domain_emb = torch.zeros(1, 512).to(self.device)

        # Step 2: Pass combined domain embeddings to SelfEvolvingAI
        embeddings, dopamine = self.self_evolving_ai(
            text_input=domain_emb,
            audio_input=audio_input,
            vision_input=vision_input,
            graph_data=graph_data,
            task_loader=task_loader
        )

        return embeddings, dopamine
