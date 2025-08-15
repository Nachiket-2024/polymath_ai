# ---------------------------- External Imports ----------------------------
import torch.nn as nn

# ---------------------------- Internal Imports ----------------------------
from .embeddings.concept_embedder import ConceptEmbedder
from .attention.multi_head_attention import MultiHeadAttention
from .gnn.gnn_model import GNNModel
from .learning.meta_learning import MetaLearner
from .learning.reward_system import RewardSystem
from .multimodal.fusion import MultimodalFusion
from .memory_networks.memory_network import MemoryNetwork
from .domain.domain_manager import DomainManager

# ---------------------------- AICore Class ----------------------------
class AICore(nn.Module):
    """
    Main AI model integrating embeddings, attention, GNN, memory, multimodal fusion,
    meta-learning, reward system, and multi-domain knowledge.
    """

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        # Initialize core modules
        self.embedder = ConceptEmbedder().to(device)
        self.attention = MultiHeadAttention(num_heads=8, d_model=512).to(device)
        self.gnn = GNNModel(in_channels=512, hidden_dim=256, output_dim=128).to(device)
        self.memory = MemoryNetwork(size=200)
        self.fusion = MultimodalFusion().to(device)
        self.reward_system = RewardSystem()
        self.meta_learner = MetaLearner(model=self, lr=0.001, fast_steps=5)

        # Initialize domain manager
        self.domain_manager = DomainManager(device=device)

    def forward(self, text_input, audio_input=None, vision_input=None,
                graph_data=None, domain_list=None):
        """
        Forward pass integrating all components and domain knowledge.

        Args:
            text_input: Tokenized text input for embedding
            audio_input: Audio input tensor (optional)
            vision_input: Vision input tensor (optional)
            graph_data: Data object for GNN (optional)
            domain_list: List of domain names to integrate (optional)

        Returns:
            tuple: (embedding tensor, dopamine signal scalar)
        """
        # Step 1: Embed text
        text_emb = self.embedder(text_input)

        # Step 2: Fuse multimodal embeddings
        fused_emb = self.fusion(text_input, audio_input, vision_input)

        # Step 3: Integrate domain knowledge if provided
        if domain_list:
            domain_emb = self.domain_manager.combine_domains(domain_list)
            fused_emb += domain_emb

        # Step 4: Apply multi-head attention
        attn_output = self.attention(
            fused_emb.unsqueeze(1),  # query
            fused_emb.unsqueeze(1),  # key
            fused_emb.unsqueeze(1)   # value
        ).squeeze(1)

        # Step 5: Pass through GNN if graph data provided
        if graph_data is not None:
            gnn_emb = self.gnn(graph_data.x, graph_data.edge_index)
        else:
            gnn_emb = attn_output

        # Step 6: Store embeddings in memory
        self.memory.store(gnn_emb.detach().cpu().numpy())

        # Step 7: Calculate reward (placeholder metrics)
        accuracy, efficiency, novelty, exploration = 0.8, 0.7, 0.6, 0.5
        reward = self.reward_system.reward(accuracy, efficiency, novelty, exploration)
        dopamine = self.reward_system.dopamine_signal(reward)

        return gnn_emb, dopamine
