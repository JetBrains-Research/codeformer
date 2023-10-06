import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.nn import Linear
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from jetnn.data_processing.plain_code_ast_method.labeled_plain_code_ast import (
    BatchedLabeledCodeAstTokens,
)
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary
from jetnn.models.util_layers.positional_encoding import PositionalEncodingWithEmbedding
from jetnn.models.util_layers.embedding import TokenEmbedding


class MethodNameMyTransformerEncoder(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        vocab: Vocabulary,
        max_subsequence_size: int,
    ):
        super().__init__()
        self._vocab_size = len(vocab)
        self._pad_token = vocab.pad_id()
        self._max_subsequence_size = max_subsequence_size + 2
        self._embedding = TokenEmbedding(self._vocab_size, config.d_model)
        self._positional_encoding = PositionalEncodingWithEmbedding(
            config.d_model, config.dropout
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self._encoder_1 = TransformerEncoder(encoder_layer, config.num_layers)
        self._linear_1 = Linear(self._max_subsequence_size, 1)
        self._encoder_2 = TransformerEncoder(encoder_layer, config.num_layers)

    def _pad_to_match_linear_layer(self, x, batch_split):
        max_splits = torch.max(batch_split)
        num_batches = len(batch_split)
        result = torch.zeros(
            (num_batches, max_splits, self._max_subsequence_size, x.shape[2])
        ).to(self._device)
        p_sum = 0
        for i, split in enumerate(batch_split):
            result[i][:split, : x.shape[1]] = x[p_sum : p_sum + split]
            p_sum += split
        return result

    def forward(self, batch: BatchedLabeledCodeAstTokens) -> Tensor:
        src_sequence = batch.code_tokens
        src_key_padding_mask = src_sequence == self._pad_token
        x = self._positional_encoding(self._embedding(src_sequence))
        x = self._encoder_1(x, src_key_padding_mask=src_key_padding_mask)
        x = self._pad_to_match_linear_layer(x, batch.batch_split)
        x = x.permute(0, 1, 3, 2)
        x = self._linear_1(x).squeeze(dim=3)
        x = self._encoder_2(x)
        return x
