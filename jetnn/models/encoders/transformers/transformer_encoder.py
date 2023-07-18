from omegaconf import DictConfig
from torch import nn, Tensor
from torch.nn import Linear
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from jetnn.data_processing.plain_code_method.labeled_plain_code import (
    BatchedLabeledCodeTokens,
)
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary
from jetnn.models.util_layers.positional_encoding import PositionalEncodingWithEmbedding
from jetnn.models.util_layers.embedding import TokenEmbedding


class MethodNameTransformerEncoder(nn.Module):
    def __init__(self, config: DictConfig, vocab: Vocabulary):
        super().__init__()
        self._vocab_size = len(vocab)
        self._pad_token = vocab.pad_id()

        self._embedding = TokenEmbedding(self._vocab_size, config.d_model)
        self._positional_encoding = PositionalEncodingWithEmbedding(config.d_model, config.dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self._encoder = TransformerEncoder(encoder_layer, config.num_layers)
        self._linear = Linear(config.d_model, len(vocab))

    def forward(self, batch: BatchedLabeledCodeTokens) -> Tensor:
        src_sequence = batch.code_tokens.permute(1, 0)
        src_key_padding_mask = src_sequence == self._pad_token
        return self._encoder(
            self._positional_encoding(self._embedding(src_sequence)),
            src_key_padding_mask=src_key_padding_mask,
        )
