# ---------------------------- External Imports ----------------------------
import numpy as np

# ---------------------------- RewardSystem Class ----------------------------
class RewardSystem:
    """
    Mimics a biological reward system calculating internal reward ("dopamine signal")
    based on accuracy, efficiency, novelty, and exploration.
    """

    def __init__(self, weights=None):
        """
        Initialize with weights for each reward component.

        Args:
            weights (list or tuple): Weights [w1, w2, w3, w4] for accuracy, efficiency,
                                    novelty, exploration respectively.
                                    Defaults to [0.5, 0.3, 0.1, 0.1].
        """
        if weights is None:
            weights = [0.5, 0.3, 0.1, 0.1]
        self.weights = weights

    def reward(self, accuracy, efficiency, novelty, exploration):
        """
        Calculate the weighted reward value.

        Args:
            accuracy (float): Accuracy metric (0 to 1).
            efficiency (float): Efficiency metric (0 to 1).
            novelty (float): Novelty metric (0 to 1).
            exploration (float): Exploration metric (0 to 1).

        Returns:
            float: Weighted reward score.
        """
        w1, w2, w3, w4 = self.weights
        return w1 * accuracy + w2 * efficiency + w3 * novelty + w4 * exploration

    def dopamine_signal(self, reward_value):
        """
        Compute the dopamine signal by applying a sigmoid function to the reward.

        Args:
            reward_value (float): Raw reward score.

        Returns:
            float: Dopamine signal between 0 and 1.
        """
        return 1 / (1 + np.exp(-reward_value))
