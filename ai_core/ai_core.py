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

    def __init__(self, device='cpu', embedding_path="ai_core/embeddings/dummy_embeddings.txt"):
        super().__init__()
        self.device = device

        # Core modules
        self.embedder = ConceptEmbedder(embedding_path=embedding_path)
        self.fusion = MultimodalFusion().to(device)

        # Use BERT hidden size (768) for attention and GNN
        self.d_model = 768
        self.attention = MultiHeadAttention(num_heads=8, d_model=self.d_model).to(device)
        self.gnn = GNNModel(self.d_model, 256, 128).to(device)
        self.memory = MemoryNetwork(size=200)
        self.reward_system = RewardSystem()
        self.domain_manager = DomainManager(device=device)

        # Optional meta-learner
        self.meta_learner = MetaLearner(model=self, lr=0.001, fast_steps=5)

        # Linear projection in case ConceptEmbedder outputs smaller embedding
        self.embed_proj = nn.Linear(512, self.d_model)  # 512 → 768

    def _get_text_embedding(self, text_input, domains):
        """
        Helper to compute text embeddings, either from domain, ConceptEmbedder, or BERT.
        """
        if domains:
            text_emb = self.domain_manager.combine_domains(domains)
            if text_emb.shape[1] != self.d_model:
                text_emb = self.embed_proj(text_emb)
        else:
            if text_input is None:
                text_emb = torch.zeros(1, self.d_model).to(self.device)
            elif isinstance(text_input, dict):
                # Tokenized input for BERT
                text_emb = self.fusion(text_input)  # fusion handles BERT dict
            elif isinstance(text_input, list):
                text_emb_np = self.embedder.embed_concept(text_input)
                text_emb = torch.tensor(text_emb_np, dtype=torch.float32).unsqueeze(0).to(self.device)
                text_emb = self.embed_proj(text_emb)
            elif isinstance(text_input, torch.Tensor):
                words = [str(tok.item()) for tok in text_input.view(-1)]
                text_emb_np = self.embedder.embed_concept(words)
                text_emb = torch.tensor(text_emb_np, dtype=torch.float32).unsqueeze(0).to(self.device)
                text_emb = self.embed_proj(text_emb)
            else:
                raise ValueError("text_input must be a tensor, list of strings, or dict")
        return text_emb

    def forward(self, text_input=None, domains=None, audio_input=None, vision_input=None, graph_data=None, task_loader=None):
        """
        Forward pass integrating text, domains, multimodal, GNN, memory, reward, and adaptation.
        """

        # ---------------- Step 1: Get text embedding ----------------
        text_emb = self._get_text_embedding(text_input, domains)

        # ---------------- Step 2: Fuse multimodal embeddings ----------------
        fused_emb = self.fusion(text_emb, audio_input, vision_input)

        # ---------------- Step 3: Apply multi-head attention ----------------
        attn_output = self.attention(
            fused_emb.unsqueeze(1),
            fused_emb.unsqueeze(1),
            fused_emb.unsqueeze(1)
        ).squeeze(1)

        # ---------------- Step 4: GNN processing if graph data is available ----------------
        gnn_emb = self.gnn(graph_data.x, graph_data.edge_index) if graph_data is not None else attn_output

        # ---------------- Step 5: Store in memory ----------------
        self.memory.store(gnn_emb.detach().cpu().numpy())

        # ---------------- Step 6: Compute placeholder reward ----------------
        accuracy, efficiency, novelty, exploration = 0.8, 0.7, 0.6, 0.5
        reward = self.reward_system.reward(accuracy, efficiency, novelty, exploration)
        dopamine = self.reward_system.dopamine_signal(reward)

        # ---------------- Step 7: Optional adaptation via task_loader ----------------
        if task_loader is not None and hasattr(self, 'meta_learner'):
            self.meta_learner.adapt(task_loader)
            # Recompute embeddings after adaptation
            text_emb = self._get_text_embedding(text_input, domains)
            fused_emb = self.fusion(text_emb, audio_input, vision_input)
            attn_output = self.attention(
                fused_emb.unsqueeze(1),
                fused_emb.unsqueeze(1),
                fused_emb.unsqueeze(1)
            ).squeeze(1)
            gnn_emb = self.gnn(graph_data.x, graph_data.edge_index) if graph_data is not None else attn_output
            self.memory.store(gnn_emb.detach().cpu().numpy())
            reward = self.reward_system.reward(accuracy, efficiency, novelty, exploration)
            dopamine = self.reward_system.dopamine_signal(reward)

        return gnn_emb, dopamine
