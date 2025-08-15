# ---------------------------- External Imports ----------------------------
import torch
import torch.nn as nn

# ---------------------------- MultiHeadAttention Class ----------------------------
class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention mechanism.
    """

    def __init__(self, num_heads, d_model):
        """
        Initialize the multi-head attention module.

        Args:
            num_heads (int): Number of attention heads.
            d_model (int): Dimensionality of the input and output.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        # Linear layers to project input to queries, keys, and values
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Final linear layer after concatenation of heads
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        """
        Forward pass for multi-head attention.

        Args:
            query, key, value (torch.Tensor): Input tensors of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size = query.shape[0]

        # Project inputs to queries, keys, and values
        Q = self.W_Q(query)  # (batch_size, seq_len, d_model)
        K = self.W_K(key)
        V = self.W_V(value)

        # Split into multiple heads and reshape
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores scaled by sqrt(head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)  # Normalize scores

        # Weighted sum of values
        context = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        out = self.fc_out(context)

        return out
