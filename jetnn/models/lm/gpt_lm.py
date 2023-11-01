from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary
from transformers import GPT2Config, GPT2LMHeadModel
from jetnn.data_processing.base_data_classes import (
    BatchedData
)

class GPTLM(nn.Module):
    def __init__(self, gpt_config: DictConfig, data_config: DictConfig, vocab: Vocabulary):
        super().__init__()

        vocab_size = len(vocab)

        self._vocab = vocab
        self._pad_id = vocab.pad_id()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        token_encoder_config = GPT2Config(**gpt_config,
                                          vocab_size=vocab_size,
                                          n_positions=data_config.max_text_tokens,)
        
        self._decoder = GPT2LMHeadModel(token_encoder_config)

    def forward(self, batch: BatchedData) -> Tensor:
        attention_mask = (batch.text_tokens == self._pad_id)
        result = self._decoder(input_ids=batch.text_tokens, attention_mask=attention_mask, labels=batch.text_tokens)
        return result.logits, batch.text_tokens, result.loss
