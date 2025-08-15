# ---------------------------- External Imports ----------------------------
import torch.nn as nn

# ---------------------------- Internal Imports ----------------------------
from .reward_system import RewardSystem
from .meta_learning import MetaLearner
from ..ai_core import AICore


# ---------------------------- SelfEvolvingAI Class ----------------------------
class SelfEvolvingAI(nn.Module):
    """
    Wraps an AICore model with self-evolving capabilities using dopamine-based
    reward and meta-learning adaptation.
    """

    def __init__(self, device='cpu', dopamine_threshold=0.6, lr=0.001, fast_steps=5):
        """
        Initialize the self-evolving AI.

        Args:
            device (str): 'cpu' or 'cuda'.
            dopamine_threshold (float): Threshold below which the model adapts.
            lr (float): Learning rate for MetaLearner.
            fast_steps (int): Number of quick adaptation steps per task.
        """
        super().__init__()
        self.device = device
        self.dopamine_threshold = dopamine_threshold

        # Core AI model
        self.core = AICore(device=device).to(device)

        # Reward system
        self.reward_system = RewardSystem()

        # Meta-learner for self-adaptation
        self.meta_learner = MetaLearner(model=self.core, lr=lr, fast_steps=fast_steps)

    def forward(self, text_input, audio_input=None, vision_input=None, graph_data=None, task_loader=None):
        """
        Forward pass with self-evolution logic.

        Args:
            text_input: Tokenized text input.
            audio_input: Optional audio input.
            vision_input: Optional vision input.
            graph_data: Optional graph data for GNN.
            task_loader: Optional DataLoader for adaptation tasks.

        Returns:
            tuple: (embedding tensor, dopamine signal)
        """
        # Step 1: Get core AI output
        embeddings, _ = self.core(text_input, audio_input, vision_input, graph_data)

        # Step 2: Compute placeholder performance metrics (can be replaced by actual evaluation)
        accuracy, efficiency, novelty, exploration = 0.8, 0.7, 0.6, 0.5
        reward = self.reward_system.reward(accuracy, efficiency, novelty, exploration)
        dopamine = self.reward_system.dopamine_signal(reward)

        # Step 3: Adapt model using MetaLearner if dopamine below threshold
        if dopamine < self.dopamine_threshold and task_loader is not None:
            self.meta_learner.adapt(task_loader)
            # Optionally recompute embeddings after adaptation
            embeddings, _ = self.core(text_input, audio_input, vision_input, graph_data)
            # Recompute dopamine after adaptation
            reward = self.reward_system.reward(accuracy, efficiency, novelty, exploration)
            dopamine = self.reward_system.dopamine_signal(reward)

        return embeddings, dopamine
