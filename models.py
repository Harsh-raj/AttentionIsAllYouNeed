import torch
import torch.nn as nn
  
class LayerNormalization(nn.Module):
  def __init__(self, eps: float=10**-6) -> None:
    super().__init__()
    self.eps = eps
    
    self.alpha = nn.Parameter(torch.ones(1)) #Multiplied
    self.bias = nn.Parameter(torch.zeros(1)) #Added
    #alpha and bias are learnable parameters
    
  def forward(self, x):
    # x: (batch, seq_len, hidden_size)
    # Keep the dimension for broadcasting
    mean = x.mean(dim = -1, keepdim=True)# (batch, seq_len, 1)
    # Keep the dimension for broadcasting
    std = x.std(dim = -1, keepdim = True)# (batch, seq_len, 1)
    # eps is to prevent dividing by zero or when std is very small
    return self.alpha * (x - mean) / (std + self.eps) + self.bias
