# ---------------------------- External Imports ----------------------------
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ---------------------------- GNNModel Class ----------------------------
class GNNModel(torch.nn.Module):
    """
    A simple Graph Convolutional Network (GCN) for processing knowledge graphs.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the GCN model.

        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of output embeddings.
        """
        super().__init__()
        # First GCN layer
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # Second GCN layer
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN.

        Args:
            x (torch.Tensor): Node feature matrix (num_nodes x input_dim).
            edge_index (torch.LongTensor): Graph connectivity (2 x num_edges).

        Returns:
            torch.Tensor: Node embeddings after GCN layers.
        """
        # First layer + ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Second layer
        x = self.conv2(x, edge_index)
        return x
