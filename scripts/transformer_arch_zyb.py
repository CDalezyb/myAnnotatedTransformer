import os
from os.path import exists
import torch
import torch.nn as nn
from torch import Tensor
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

""" d_model 和 hidden_size 混用"""

# %%
# Definition of MHA and attention computation


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor = None,
    dropout: nn.Module = None,
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
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # mask为0表示被去掉，注意力得分为无穷小，取指数以后为负无穷，概率为0
        scores.masked_fill_(mask == 0, 1e-9)
    p_atten = scores.softmax(dim=-1)
    if dropout is not None:
        scores = dropout(p_atten)
    return torch.matmul(scores, value), p_atten


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        """
        multi-head attention
            Args:
                d_model:      dimension of embeddings
                n_heads:      number of self-attention heads
                dropout:      probability of dropout
        """
        assert d_model % n_heads == 0, "d_model should be "
        super(MultiHeadAttention, self).__init()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model / n_heads
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        # linears are list of the Wq Wk Wv and Wo
        self.linears = clone_Modules(nn.Linear(d_model, d_model), 4)
        self.atten_prob = None  # probability matirx

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor = None,
    ):
        n_batch = query.size(0)

        # after this, we get QKV, shape: [n_batch, n_heads, sql_len, d_k]
        # zip will enumerate the first 3 matrices in self.linears
        query, key, value = [
            lin(x).view(n_batch, -1, self.n_heads, self.d_k).transpose(-2, -1)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.atten_prob = attention(
            query=query, key=key, value=value, mask=mask, dropout=self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.d_model)

        # del query, key, value

        return x, self.atten_prob


# %%
# Implementations of
#                       Embedding, PositionalEncoder, LayerNorm, PositionwiseFeedForward,
#                       EncoderLayer, DecoderLayer,
#                       Encoder, Decoder
#                       EncoderDecoder full arch and make_model method to construct the full network
class Embedding(nn.Module):
    """
    Args:
        d_model:        hidden_size
        d_vocab:        dimension of src and tgt vocab dictionary
    """

    def __init__(self, d_model: int, d_vocab: int):
        super(Embedding, self).__init__()
        # nn.Embedding的第一个参数为 自然语言词典大小
        # nn.Embedding的第二个参数为 隐藏层维度
        self.lut = nn.Embedding(d_vocab, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor):
        # ensures the embedding vectors have a similar magnitude to the positional encodings
        # they will be combined with, preventing one from dominating the other in the model's attention mechanisms
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoder(nn.Module):
    """
    Args:
        NOT FINISHED YET.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        pass

    def forward(self, x: Tensor):
        return x


class LayerNorm(nn.module):
    def __init__(self, hidden_size: int, affine: bool = True, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zereos(hidden_size))

    def froward(self, x: Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(std + self.eps)
        if self.affine:
            x_norm = self.weight * x_norm + self.bias
        return x_norm


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_ffn = d_ffn
        self.w1 = nn.Linear(d_model, d_ffn)
        self.w2 = nn.Linear(d_ffn, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        # w1 is followed by relu
        return self.w2(self.drop(self.w1(x).relu()))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        """
        Args:
            hidden_size/d_model:      dimension of embeddings
            n_heads:                   number of heads
            d_ffn:                      dimension of feed-forward network
            dropout:                     probability of dropout occurring
        """
        super(EncoderLayer, self).__init()
        self.self_atten = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_attention = LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ffn, dropout=dropout)
        self.norm_ffn = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None):
        """
        Args:
            src:          positionally embedded sequences   (batch_size, seq_length, d_model)
            src_mask:     mask for the sequences            (batch_size, 1, 1, seq_length)
        Returns:
            src:          sequences after self-attention    (batch_size, seq_length, d_model)
        """
        _x, _ = self.self_atten(x, x, x, mask)

        x = self.norm_attention(
            x + self.drop(_x)
        )  # x + Dropout( attention(x, x, x, mask) )

        # now x is attention output
        return self.norm_ffn(x + self.drop(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        """
        Args:
            d_model:      dimension of embeddings
            n_heads:      number of heads
            d_ffn:        dimension of feed-forward network
            dropout:      probability of dropout occurring
        """
        super(DecoderLayer, self).__init()

        self.masked_self_atten = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_masked_attention = LayerNorm(d_model)

        self.cross_atten = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_cross_attention = LayerNorm(d_model)

        self.norm_ffn = LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ffn, dropout=dropout)

        self.drop = nn.Dropout(
            dropout
        )  # Dropout Module has no Parameters and can be reused

    def forward(self, tgt: Tensor, src: Tensor, tgt_mask: Tensor, src_mask: Tensor):
        """
        Args:
            tgt:          embedded sequences                                            (batch_size, trg_seq_length, d_model)
            src:          embedded sequences from encoder                               (batch_size, src_seq_length, d_model)
            tgt_mask:     mask for the sequences                                        (batch_size, 1, trg_seq_length, trg_seq_length)
            src_mask:     mask for the sequences(padding and invalid char)              (batch_size, 1, 1, src_seq_length)

        Returns:
            tgt:          sequences after self-attention    (batch_size, trg_seq_length, d_model)
        """
        _tmp, _ = self.masked_self_atten(
            tgt, tgt, tgt, tgt_mask
        )  # casual mask of output tokens
        tgt = self.norm_masked_attention(tgt + self.drop(_tmp))

        _tmp = self.cross_atten(tgt, src, src, src_mask)
        tgt = self.norm_cross_attention(tgt + self.drop(_tmp))

        _tmp = self.ffn(tgt)
        return self.norm_ffn(tgt + self.drop(_tmp))


# %%
# Definition of
#               Encoder, Decoder
class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model:      dimension of embeddings
            n_layers:     number of encoder layers
            n_heads:      number of heads
            d_ffn:        dimension of feed-forward network
            dropout:      probability of dropout occurring
        """
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)]
        )

    def forward(self, src: Tensor, src_mask: Tensor):
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        return src


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model:      dimension of embeddings
            n_layers:     number of encoder layers
            n_heads:      number of heads
            d_ffn:        dimension of feed-forward network
            dropout:      probability of dropout occurring
        """
        super(Encoder, self).__init__()
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        tgt: Tensor,
        tgt_mask: Tensor,
        encoder_out: Tensor,
        src_mask: Tensor,
    ):
        for layer in self.decoder_layers:
            src = layer(tgt, encoder_out, tgt_mask, src_mask)
        return src


# %%
# Definition of
#               EncoderDecoder full arch and make_model method to construct the full network
class Generator(nn.Module):
    def __init__(self, d_model: int, d_tgt_vocab: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, d_tgt_vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ffn: int = 2048,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        # 输入 embedding，with src_mask
        self.src_embed = nn.Sequential(
            Embedding(d_model, src_vocab), PositionalEncoder(d_model, dropout)
        )
        self.encoder = Encoder(d_model, n_heads, d_ffn, n_layers, dropout)

        # 输出 embedding，with tgt__mask
        self.tgt_embed = nn.Sequential(Embedding(d_model, tgt_vocab))
        self.decoder = Decoder(d_model, n_heads, d_ffn, n_layers, dropout)

        # project to dimension of tgt_vocab and softmax to verb
        self.generator = Generator(d_model, tgt_vocab)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ):

        return self.generator(
            self.decoder(
                encoder_out=self.encoder(src, src_mask),
                src_mask=src_mask,
                tgt=tgt,
                tgt_mask=tgt_mask,
            )
        )

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, encoder_out, src_mask, tgt, tgt_mask):
        return self.decoder(
            tgt=self.tgt_embed(tgt),
            tgt_mask=tgt_mask,
            encoder_out=encoder_out,
            src_mask=src_mask,
        )


def make_model(
    src_vocab,
    tgt_vocab,
    N=6,
    d_model=512,
    d_ffn=2048,
    h=8,
    dropout=0.1,
):
    model = EncoderDecoder(src_vocab, tgt_vocab, d_model, h, d_ffn, N, dropout)

    # initialize the model
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
