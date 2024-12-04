import torch
import torch.nn as nn

class Transformer(nn.Module):
  
  def __init__(self, encoder: Encoder, decoder: Decoder, source_embed: InputEmbeddings, target_embed: InputEmbeddings, source_pos: PositionalEncoding, target_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.source_embed = source_embed
    self.target_embed = target_embed
    self.source_pos = source_pos
    self.target_pos = target_pos
    self.projection_layer = projection_layer
    
  def encoder(self, src, src_mask):
    src = self.source_embed(src)
    src = self.source_pos(src)
    return self.encoder(src, src_mask)
  
  def decoder(self, encoder_output, source_mask, target, target_mask):
    target = self.target_embed(target)
    target = self.target_pos(target)
    return self.decoder(target, encoder_output, source_mask, target_mask)
  
  def project(self, x):
    return self.projection_layer(x)