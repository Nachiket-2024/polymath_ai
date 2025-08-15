# ---------------------------- External Imports ----------------------------
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------- MetaLearner Class ----------------------------
class MetaLearner:
    """
    Implements a basic meta-learning loop for continuous learning and adaptation.
    """

    def __init__(self, model: nn.Module, lr: float = 0.001, fast_steps: int = 5):
        """
        Initialize meta-learner.

        Args:
            model (nn.Module): Model to be meta-trained.
            lr (float): Learning rate for inner updates.
            fast_steps (int): Number of quick adaptation steps per task.
        """
        self.model = model
        self.lr = lr
        self.fast_steps = fast_steps
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()  # Can be customized

    def adapt(self, task_loader):
        """
        Perform adaptation steps on a single task dataset.

        Args:
            task_loader (iterable): DataLoader or iterable providing task data batches.
        """
        self.model.train()
        for step, (inputs, targets) in enumerate(task_loader):
            if step >= self.fast_steps:
                break
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

    def meta_train(self, tasks):
        """
        Perform meta-training over multiple tasks.

        Args:
            tasks (list): List of task datasets or DataLoaders.
        """
        for task_loader in tasks:
            self.adapt(task_loader)
            # Optionally, evaluate performance on new tasks here

    def evaluate(self, eval_loader):
        """
        Evaluate the adapted model on evaluation data.

        Args:
            eval_loader (iterable): Evaluation data loader.

        Returns:
            float: Average loss or metric.
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in eval_loader:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        return total_loss / total_samples if total_samples > 0 else 0
