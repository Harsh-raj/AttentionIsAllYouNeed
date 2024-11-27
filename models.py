import torch
import torch.nn as nn

class Decoder(nn.Module):
  
  def __init__(self, layers: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers
    self.norms = LayerNormalization()
    
  def forward(self, x, encoder_output, src_mask, target_mask):
    for layer in self.layers:
      x = layer(x, encoder_output, src_mask, target_mask)
    return self.norm(x)