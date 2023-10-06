from omegaconf import DictConfig
from torch import nn, Tensor
from transformers import T5EncoderModel, T5Config

from jetnn.data_processing.plain_code_method.labeled_plain_code import (
    BatchedLabeledCodeTokens,
)
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary


class MethodNameT5Encoder(nn.Module):
    def __init__(self, config: DictConfig, vocab: Vocabulary):
        super().__init__()
        self._vocab_size = len(vocab)
        self._pad_token = vocab.pad_id()

        t5_config = T5Config(
            vocab_size=self._vocab_size,
            d_model=config.d_model,
            d_ff=config.dim_feedforward,
            num_heads=config.nhead,
            num_layers=config.num_layers,
            dropout_rate=config.dropout,
        )
        self._encoder = T5EncoderModel(t5_config)

    def forward(self, batch: BatchedLabeledCodeTokens) -> Tensor:
        src_sequence = batch.code_tokens.permute(1, 0)
        src_key_padding_mask = src_sequence == self._pad_token
        src_key_padding_mask = src_key_padding_mask.long()
        return self._encoder(
            input_ids=src_sequence, attention_mask=src_key_padding_mask
        ).last_hidden_state
