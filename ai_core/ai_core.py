# ---------------------------- External Imports ----------------------------
import torch
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
    AI model integrating embeddings, attention, GNN, memory, multimodal fusion,
    meta-learning, reward system, and domain reasoning.
    """

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        # Core modules
        self.embedder = ConceptEmbedder().to(device)
        self.attention = MultiHeadAttention(num_heads=8, d_model=512).to(device)
        self.gnn = GNNModel(in_channels=512, hidden_dim=256, output_dim=128).to(device)
        self.memory = MemoryNetwork(size=200)
        self.fusion = MultimodalFusion().to(device)
        self.reward_system = RewardSystem()
        self.domain_manager = DomainManager(device=device)  # New

        # Optional meta-learner
        self.meta_learner = MetaLearner(model=self, lr=0.001, fast_steps=5)

    def forward(self, text_input=None, domains=None, audio_input=None, vision_input=None, graph_data=None, task_loader=None):
        """
        Forward pass integrating text, domains, multimodal, GNN, memory, reward, and adaptation.

        Args:
            text_input: Optional tokenized text input.
            domains: Optional list of domain names for domain reasoning.
            audio_input: Optional audio input tensor.
            vision_input: Optional vision input tensor.
            graph_data: Optional GNN data.
            task_loader: Optional DataLoader for adaptation tasks.

        Returns:
            tuple: (embedding tensor, dopamine scalar)
        """
        # Step 1: Embed concepts from text or domain embeddings
        if domains:
            text_emb = self.domain_manager.combine_domains(domains)
        else:
            text_emb = self.embedder(text_input) if text_input is not None else torch.zeros(1, 512).to(self.device)

        # Step 2: Fuse multimodal embeddings
        fused_emb = self.fusion(text_emb, audio_input, vision_input)

        # Step 3: Apply multi-head attention
        attn_output = self.attention(
            fused_emb.unsqueeze(1),
            fused_emb.unsqueeze(1),
            fused_emb.unsqueeze(1)
        ).squeeze(1)

        # Step 4: GNN processing if graph data is available
        gnn_emb = self.gnn(graph_data.x, graph_data.edge_index) if graph_data is not None else attn_output

        # Step 5: Store in memory
        self.memory.store(gnn_emb.detach().cpu().numpy())

        # Step 6: Compute placeholder reward
        accuracy, efficiency, novelty, exploration = 0.8, 0.7, 0.6, 0.5
        reward = self.reward_system.reward(accuracy, efficiency, novelty, exploration)
        dopamine = self.reward_system.dopamine_signal(reward)

        # Step 7: Optional adaptation via task_loader
        if task_loader is not None and hasattr(self, 'meta_learner'):
            self.meta_learner.adapt(task_loader)
            # Recompute embeddings
            if domains:
                text_emb = self.domain_manager.combine_domains(domains)
            else:
                text_emb = self.embedder(text_input) if text_input is not None else torch.zeros(1, 512).to(self.device)
            fused_emb = self.fusion(text_emb, audio_input, vision_input)
            attn_output = self.attention(fused_emb.unsqueeze(1), fused_emb.unsqueeze(1), fused_emb.unsqueeze(1)).squeeze(1)
            gnn_emb = self.gnn(graph_data.x, graph_data.edge_index) if graph_data is not None else attn_output
            self.memory.store(gnn_emb.detach().cpu().numpy())
            reward = self.reward_system.reward(accuracy, efficiency, novelty, exploration)
            dopamine = self.reward_system.dopamine_signal(reward)

        return gnn_emb, dopamine
