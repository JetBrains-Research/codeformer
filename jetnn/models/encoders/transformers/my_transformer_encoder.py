import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.nn import Linear
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from transformers import BigBirdModel, BigBirdConfig
from jetnn.data_processing.plain_code_ast_method.labeled_plain_code_ast import (
    BatchedLabeledCodeTokens,
)
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary
from jetnn.models.decoders.transformer_decoder import TokenEmbedding, PositionalEncoding


class MethodNameMyTransformerEncoder(nn.Module):
    def __init__(
            self, config: DictConfig, vocab: Vocabulary, max_subsequence_size: int, big_bird_config: DictConfig
    ):
        super().__init__()
        self._vocab_size = len(vocab)
        self._pad_token = vocab.pad_id()
        self._max_subsequence_size = max_subsequence_size
        self._embedding = TokenEmbedding(self._vocab_size, config.d_model)
        self._positional_encoding = PositionalEncoding(config.d_model, config.dropout)
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        # big_bird_config = BigBirdConfig(
        #     vocab_size=self._vocab_size,
        #     hidden_size=big_bird_config.d_model,
        #     num_attention_heads=big_bird_config.nhead,
        #     num_hidden_layers=big_bird_config.num_layers,
        #     hidden_dropout_prob=big_bird_config.dropout,
        #     attention_type=big_bird_config.attention_type,
        #     block_size=big_bird_config.block_size,
        #     num_random_blocks=big_bird_config.num_random_blocks,
        # )
        self._encoder_1 = TransformerEncoder(encoder_layer, config.num_layers)
        # self._encoder_1 = BigBirdModel(big_bird_config)
        self._linear_1 = Linear(self._max_subsequence_size, 1)
        # self._encoder_2 = BigBirdModel(big_bird_config)
        self._encoder_2 = TransformerEncoder(encoder_layer, config.num_layers)

    def _pad_to_match_linear_layer(self, x, batch_split):
        max_splits = torch.max(batch_split)
        num_splits = len(batch_split)
        result = torch.zeros((num_splits, max_splits, self._max_subsequence_size, x.shape[2])).to(self._device)
        p_sum = 0
        for i, split in enumerate(batch_split):
            result[i][:split, :x.shape[1]] = x[p_sum: p_sum + split]
            p_sum += split
        return result

    def forward(self, batch: BatchedLabeledCodeTokens) -> Tensor:
        src_sequence = batch.code_tokens
        src_key_padding_mask = src_sequence == self._pad_token
        x = self._positional_encoding(self._embedding(src_sequence))
        x = self._encoder_1(x, src_key_padding_mask=src_key_padding_mask)
        # x = self._encoder_1(inputs_embeds=x, attention_mask=src_key_padding_mask).last_hidden_state
        x = self._pad_to_match_linear_layer(x, batch.batch_split)
        x = x.permute(0, 1, 3, 2)
        x = self._linear_1(x).squeeze(dim=3)
        # x = self._encoder_2(inputs_embeds=x).last_hidden_state
        x = self._encoder_2(x)
        return x
