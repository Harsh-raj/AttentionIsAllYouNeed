import torch.nn as nn
import torch

class FeedForwardBlock(nn.Module):
  
  def __init__(self, d_model: int, d_ff: int, dropout: int) -> None:
    super().__init__()
    self.linear_1 = nn.Layer(d_model, d_ff) # W1 and B1
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2
    
  def forward(self, x):
    # (Batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))