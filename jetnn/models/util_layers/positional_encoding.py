import math

import torch
from torch import nn, Tensor
from torch.nn import Embedding


class PositionalEncodingOriginal(nn.Module):
    def __init__(self, emb_size: int, dropout: float, max_token_length: int = 5000):
        super(PositionalEncodingOriginal, self).__init__()

        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_token_length).reshape(max_token_length, 1)

        pe = torch.zeros((max_token_length, emb_size))
        pe[:, 0::2] = torch.sin(pos * den)
        pe[:, 1::2] = torch.cos(pos * den)
        pe = pe.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe)

    def forward(self, token_embedding: Tensor):
        output = token_embedding + self.pe[:, : token_embedding.size(1), :]
        return self.dropout(output)


class PositionalEncodingWithEmbedding(nn.Module):
    def __init__(self, emb_size, dropout, max_sequence_len=5000):
        super(PositionalEncodingWithEmbedding, self).__init__()
        self._positional_encoding = Embedding(max_sequence_len, emb_size)
        self._dropout = nn.Dropout(dropout)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, token_embedding):
        positions = torch.arange(token_embedding.size(1)).to(self._device)
        return token_embedding + self._positional_encoding(positions)
