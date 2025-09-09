import os
from os.path import exists
import torch
import torch.nn as nn
import torch.Tensor as Tensor
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils import *
''' d_model 和 hidden_size 混用'''

#%% 
# Definition of MHA and attention computation
def attention(
    query:Tensor,
    key:Tensor,
    value:Tensor,
    mask:Tensor = None,
    dropout:nn.Module = None
):
    """
    Compute Scaled-Dot-Product-Attention.
        Args:
            query:      query, shape: [batch_size, n_heads, sql_len, d_k]
            key:        key,   shape: [batch_size, n_heads, sql_len, d_k]
            value:      value, shape: [batch_size, n_heads, sql_len, d_k]
            mask:       mask
            dropout:    nn.Dropout module instance
        
        Returns:
            [1]:        attention result, shape: [batch_size, n_heads, sql_len, d_k]
            [p_atten]   probalibity matrix
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(mask == 0, 1e-9)# mask为0表示被去掉，注意力得分为无穷小，取指数以后为负无穷，概率为0
    p_atten = scores.softmax(dim = -1)
    if dropout is not None:
        scores = dropout(p_atten)
    return torch.matmul(scores, value), p_atten
    
class MultiHeadAttention(nn.Mudule):
    def __init__(self, 
                 d_model:int = 512,
                 n_heads:int = 8,
                 dropout:float = 0.1):
        """
        multi-head attention
            Args:
                d_model:      dimension of embeddings
                n_heads:      number of self-attention heads
                dropout:      probability of dropout
        """
        assert d_model % n_heads ==0, "d_model should be "
        super(MultiHeadAttention, self).__init()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = self.d_model/self.n_heads
        self.dropout = nn.Dropout(p = dropout) if dropout > 0 else None
        # linears are list of the Wq Wk Wv and Wo
        self.linears = clone_Modules(nn.Linear(d_model, d_model), 4)
        self.atten_prob = None # probability matirx
        
    def forward(self,
                query:Tensor,
                key:Tensor,
                value:Tensor,
                mask:Tensor = None):
        n_batch = query.size(0)
        
        # after this, we get QKV, shape: [n_batch, n_heads, sql_len, d_k]
        # zip will enumerate the first 3 matrices in the linear layers
        query, key, value = [
            lin(x).view(n_batch, -1, self.n_heads, self.d_k)
            .transpose(-2,-1)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        
        x, self.atten_prob = attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            dropout=self.dropout
        )
        
        x = x.transpose(1,2).contiguous().view(n_batch, -1, self.d_model)
        
        # del query, key, value
        
        return x
        
        
#%% 
# Implementation of EncoderLayer, DecoderLayer, LayerNorm, FFN, PositionEmbedding, 
#                   EncoderBlock, DecoderBlock
#                   EncoderDecoder and make model