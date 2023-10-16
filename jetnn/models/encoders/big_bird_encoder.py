from omegaconf import DictConfig
from torch import nn, Tensor
from transformers import BigBirdModel, BigBirdConfig

from jetnn.data_processing.plain_code_method.labeled_plain_code import (
    BatchedLabeledCodeTokens,
)
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary


class BigBirdEncoder(nn.Module):
    def __init__(self, config: DictConfig, vocab: Vocabulary):
        super().__init__()
        self._vocab_size = len(vocab)
        self._pad_token = vocab.pad_id()

        big_bird_config = BigBirdConfig(
            vocab_size=self._vocab_size,
            hidden_size=config.d_model,
            num_attention_heads=config.nhead,
            num_hidden_layers=config.num_layers,
            hidden_dropout_prob=config.dropout,
            attention_type=config.attention_type,
            block_size=config.block_size,
            num_random_blocks=config.num_random_blocks,
        )
        self._encoder = BigBirdModel(big_bird_config)

    def forward(self, batch: BatchedLabeledCodeTokens) -> Tensor:
        src_sequence = batch.code_tokens.permute(1, 0)
        src_key_padding_mask = src_sequence == self._pad_token
        src_key_padding_mask = src_key_padding_mask.long()
        return self._encoder(
            src_sequence, attention_mask=src_key_padding_mask
        ).last_hidden_state
