import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from transformers import LongformerModel, LongformerConfig

from jetnn.data_processing.plain_code_method.labeled_plain_code import (
    BatchedLabeledCodeTokens,
)
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary


class MethodNameLongformerEncoder(nn.Module):
    def __init__(self, config: DictConfig, vocab: Vocabulary):
        super().__init__()
        self._vocab_size = len(vocab)
        self._pad_token = vocab.pad_id()
        longformer_config = LongformerConfig(
            vocab_size=self._vocab_size,
            hidden_size=config.d_model,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.nhead,
            hidden_dropout_prob=config.dropout,
            attention_window=config.attention_window,
        )
        self._encoder = LongformerModel(longformer_config)

    def forward(self, batch: BatchedLabeledCodeTokens) -> Tensor:
        src_sequence = batch.code_tokens.permute(1, 0)
        src_key_padding_mask = src_sequence == self._pad_token
        return self._encoder(
            src_sequence, attention_mask=src_key_padding_mask
        ).last_hidden_state
