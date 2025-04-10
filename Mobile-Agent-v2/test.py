import torch
import torch.nn as nn
import math

class SelfAttention (nn.modules):
    def __init__(self, hidden_dim: int, dropout_rate: float)->None:
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X, attention_mask):
        # (batch, seq_len, hidden_dim)
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        attention_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.hidden_dim)

        if attention_mask is not None:
            nn.attention_weight.masked_fill( attention_mask==0, float("-inf"))

        attention_weight = torch.softmax(attention_weight, dim=-1)

        attention_value = attention_weight @ V

        return attention_value 



