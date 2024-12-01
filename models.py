import torch
import torch.nn as nn

class ResidualConnection(nn.module):
  def __init__(self, dropout: float) -> None:
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization()

  def forward(self, x, sublayer):
    return x+self.dropout(sublayer(self.norm(x))) #some implementations have normalization applied on sublayer which is opposite of this one but we will stick to this implementation.