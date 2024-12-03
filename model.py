import torch.nn as nn
import torch
import math

class MultiHeadAttentionBlock(nn.Module):

  def __init__(self, d_model: int, h: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.h = h
    assert d_model % h == 0, "d_model is not divisible by h"
    
    self.d_k = d_model // h
    self.w_q = nn.Linear(d_model, d_model, bias=False) #wq
    self.w_k = nn.Linear(d_model, d_model, bias=False) #wk
    self.w_v = nn.Linear(d_model, d_model, bias=False) #wv
    
    self.w_o = nn.Linear(d_model, d_model, bias=False) #wo
    self.dropout = nn.Dropout(dropout)
  
  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout):
    d_k = query.shape[-1]
    
    # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
      attention_scores.masked_fill_(mask == 0, -1e9)
    
    attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)
    
    if dropout is not None:
      attention_scores = dropout(attention_scores)
      
    return (attention_scores @ value), attention_scores
    
     
  def forward(self, q, k, v, mask):
    query = self.w_q(q) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
    key = self.w_k(k) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
    value = self.w_v(v) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
    
    # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
    key = query.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
    
    x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
    
    # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
    
    # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
    return self.w_o(x)
    