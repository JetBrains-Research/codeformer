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

from jetnn.data_processing.plain_code_ast_method.labeled_plain_code_ast import (
    BatchedLabeledCodeAstTokens,
)
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary
from jetnn.models.util_layers.positional_encoding import PositionalEncodingWithEmbedding
from jetnn.models.util_layers.embedding import TokenEmbedding


class CodeModellingTransformerDecoder(nn.Module):
    def __init__(
            self, config: DictConfig, vocab: Vocabulary, max_subsequence_size: int,
    ):
        super().__init__()
        self._vocab_size = len(vocab)
        self._pad_token = vocab.pad_id()
        
    def forward(self, batch: BatchedLabeledCodeAstTokens, step) -> Tensor:
        pass
