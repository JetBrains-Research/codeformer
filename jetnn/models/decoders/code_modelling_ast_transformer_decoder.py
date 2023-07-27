import math

import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.nn import Linear
from torch.nn.modules.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    Transformer,
)

from jetnn.data_processing.vocabularies.vocabulary import Vocabulary
from jetnn.models.util_layers.positional_encoding import PositionalEncodingWithEmbedding
from jetnn.models.util_layers.embedding import TokenEmbedding


class CodeModellingAstTransformerDecoder(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        vocab: Vocabulary,
        max_subsequence_size: int,
    ):
        super().__init__()
        self._vocab_size = len(vocab)
        self._pad_token = vocab.pad_id()
        self._max_subsequence_size = max_subsequence_size
        if config.use_begin_end_tokens:
            self._max_subsequence_size += 2
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
        self._tokens_encoder = TransformerEncoder(encoder_layer, config.num_layers)
        self._context_encoder = TransformerEncoder(encoder_layer, config.num_layers)
        self._sequence_linear = Linear(self._max_subsequence_size, 1)
        self._decoder = TransformerEncoder(encoder_layer, config.num_layers)
        self._vocab_linear = Linear(config.d_model, len(vocab))

    def _pad_to_match_linear_layer(self, x, batch_split):
        max_splits = torch.max(batch_split)
        num_splits = len(batch_split)
        result = torch.zeros(
            (num_splits, max_splits, self._max_subsequence_size, x.shape[2])
        ).to(self._device)
        p_sum = 0
        for i, split in enumerate(batch_split):
            result[i][:split, : x.shape[1]] = x[p_sum : p_sum + split]
            p_sum += split
        return result

    def _generate_context_for_decoder(self, context, batch_split, sequence_size):
        start_context = torch.zeros((sequence_size, context.size(-1))).to(self._device)
        split_sum = 0
        for batch_num, split in enumerate(batch_split):
            start_context[split_sum + 1 : split_sum + split] = context[batch_num][
                : split - 1
            ]
            split_sum += split
        return start_context.unsqueeze(dim=1)

    def forward(self, batch, step) -> Tensor:
        src_sequence = batch.code_tokens
        src_key_padding_mask = src_sequence == self._pad_token
        tokens_embedding = self._positional_encoding(self._embedding(src_sequence))
        context = self._tokens_encoder(
            tokens_embedding, src_key_padding_mask=src_key_padding_mask
        )
        context = self._pad_to_match_linear_layer(context, batch.batch_split)
        context = context.permute(0, 1, 3, 2)
        context = self._sequence_linear(context).squeeze(dim=3)

        src_mask = Transformer.generate_square_subsequent_mask(context.size(1)).to(
            self._device
        )
        context = self._context_encoder(context, mask=src_mask)

        start_context = self._generate_context_for_decoder(
            context, batch.batch_split, src_sequence.size(0)
        )
        mask = (
            Transformer.generate_square_subsequent_mask(src_sequence.size(1) + 1)
        ).to(self._device)

        tokens_with_context = torch.cat((start_context, tokens_embedding), dim=1)
        context_padding_mask = torch.zeros((src_key_padding_mask.size(0), 1)).to(
            self._device
        )
        src_key_padding_mask = torch.cat(
            (context_padding_mask, src_key_padding_mask), dim=1
        )

        x = self._decoder(
            tokens_with_context, mask=mask, src_key_padding_mask=src_key_padding_mask
        )
        x = self._vocab_linear(x)
        return x.permute(1, 0, 2)
