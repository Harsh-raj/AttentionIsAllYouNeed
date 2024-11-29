import torch
import torch.nn as nn

class BuildTransformer:
  def __init__(source_vocab_size: int, target_vocab_size: int, source_seq_len: int, target_seq_len: int, d_model:int = 512, N: int = 6, h: int = 8, dropout: float=0.1, d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    source_embed = InputEmbeddings(d_model, source_vocab_size)
    target_embed = InputEmbeddings(d_model, target_vocab_size)
    
    # Create the positional emcoding layer
    source_pos = PositionalEncoding(d_model, source_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
      encoder_self_attention_blocks = MultiHeadAttentionBlock(d_model, h, dropout)
      feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
      encoder_block = EncoderBlock(encoder_self_attention_blocks, feed_forward_block, dropout)
      encoder_block.append(encoder_block)
      
    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
      decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
      decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
      feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
      decoder_blocks = decoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
      decoder_blocks.append(decoder_blocks)
      
    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = projectionLayer(d_model, target_vocab_size)
    
    # Create the transformer
    transformer = transformer(encoder, decoder, source_embed, target_embed, source_pos, target_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
        
    return transformer