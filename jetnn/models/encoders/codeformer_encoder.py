import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.nn import Linear
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from jetnn.data_processing.base_data_classes import (
    BatchedData,
)
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary
from jetnn.models.util_layers.positional_encoding import PositionalEncodingWithEmbedding
from jetnn.models.util_layers.embedding import TokenEmbedding

from jetnn.models.utils.codeformer_utils import (
    generate_batches_from_splits,
    pad_to_match_linear_layer
)

class CodeformerEncoder(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        vocab: Vocabulary,
        max_chunk_size: int,
    ):
        super().__init__()
        self._vocab_size = len(vocab)
        self._pad_id = vocab.pad_id()
        self._bos_id = vocab.bos_id()
        self._eos_id = vocab.eos_id()
        self._context_size = max_chunk_size + 2
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
        self._linear_1 = Linear(self._context_size, 1)
        self._encoder_2 = TransformerEncoder(encoder_layer, config.num_layers)

    def forward(self, batch: BatchedData) -> Tensor:
        splits = generate_batches_from_splits(
            input_ids=batch.text_tokens, 
            splits_size=batch.batch_split, 
            context_size=self._context_size,
            pad_id=self._pad_id, 
            bos_id=self._bos_id, 
            eos_id=self._eos_id,
            device=self._device,
        )
        attention_mask = (splits == self._pad_id)
        x = self._positional_encoding(self._embedding(splits))
        x = self._encoder_1(x, src_key_padding_mask=attention_mask)
        x = pad_to_match_linear_layer(
            x=x,
            split_sizes=batch.batch_split,
            context_size=self._context_size
        )
        x = x.permute(0, 1, 3, 2)
        x = self._linear_1(x).squeeze(dim=3)
        x = self._encoder_2(x)
        return x
