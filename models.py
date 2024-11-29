import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
  
  def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
    super().__init__()
    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
  def forward(self, x, encoder_output, src_mask, target_mask):
    x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
    x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
    x = self.residual_connection[2](x, self.feed_forward_block)
    return x