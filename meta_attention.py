import torch
import torch.nn as nn
import math


class MetaAttention(nn.Module):
    """
    MetaAttention Layer
    
    Attends over reasoning paths instead of individual tokens.
    Forms a global reasoning query from the model's internal state and
    uses it to aggregate information across the sequence.
    
    This enables the model to:
    - Analyze its own reasoning patterns
    - Understand what aspects of the input were important
    - Maintain global context for decision-making
    
    Args:
        d_model (int): Dimension of model
        num_heads (int): Number of attention heads (default: 4)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Multi-head attention projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, d_model)
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, H: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through MetaAttention.
        
        Args:
            H (Tensor): Hidden states [batch_size, seq_length, d_model]
            mask (Tensor, optional): Attention mask [batch_size, seq_length]
        
        Returns:
            Tensor: Meta-attention output [batch_size, 1, d_model]
        """
        batch_size = H.shape[0]
        
        # Layer normalization for stability
        H_norm = self.layer_norm(H)
        
        # Create global reasoning query from mean of sequence
        # This represents what the model should focus on for reasoning
        global_query = H_norm.mean(dim=1, keepdim=True)  # [batch_size, 1, d_model]
        
        # Project to query, key, value
        Q = self.query(global_query)    # [batch_size, 1, d_model]
        K = self.key(H_norm)             # [batch_size, seq_length, d_model]
        V = self.value(H_norm)           # [batch_size, seq_length, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # [batch_size, num_heads, 1, seq_length]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_length]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [batch_size, num_heads, 1, head_dim]
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, 1, self.d_model)
        
        # Output projection
        output = self.fc_out(context)  # [batch_size, 1, d_model]
        
        return output
