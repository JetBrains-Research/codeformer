import math

import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.nn import Embedding, Linear
from torch.nn.modules.transformer import (
    TransformerDecoderLayer,
    Transformer,
    TransformerDecoder,
)

from jetnn.data_processing.plain_code_method.labeled_plain_code import (
    BatchedLabeledCodeTokens,
)
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, max_token_length: int = 5000):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_token_length).reshape(max_token_length, 1)

        pe = torch.zeros((max_token_length, emb_size))
        pe[:, 0::2] = torch.sin(pos * den)
        pe[:, 1::2] = torch.cos(pos * den)
        pe = pe.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe)

    def forward(self, token_embedding: Tensor):
        output = token_embedding + self.pe[:, : token_embedding.size(1), :]
        return self.dropout(output)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class MethodNameTransformerDecoder(nn.Module):
    def __init__(self, config: DictConfig, vocab: Vocabulary):
        super().__init__()
        self._vocab_size = len(vocab)
        self._pad_token = vocab.pad_id()
        self._sos_token = vocab.bos_id()
        self._eos_token = vocab.eos_id()

        self._embedding = TokenEmbedding(self._vocab_size, config.d_model)
        self._positional_encoding = PositionalEncoding(config.d_model, config.dropout)
        decoder_layer = TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self._decoder = TransformerDecoder(decoder_layer, config.num_layers)
        self._linear = Linear(config.d_model, len(vocab))

    def decode(
        self, target_sequence: Tensor, batched_encoder_output: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        tgt_key_padding_mask = target_sequence == self._pad_token
        embedded = self._embedding(target_sequence)
        positionally_encoded = self._positional_encoding(embedded)
        decoded = self._decoder(
            tgt=positionally_encoded,
            memory=batched_encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self._linear(decoded)

    def forward(
        self, batched_encoder_output: Tensor, batch: BatchedLabeledCodeTokens, step: str
    ) -> Tensor:
        device = batched_encoder_output.device
        target_sequence = batch.label_tokens.permute(1, 0)
        batch_size = target_sequence.shape[0]
        output_size = target_sequence.shape[1]

        if step != "test":
            tgt_mask = (Transformer.generate_square_subsequent_mask(output_size)).to(
                device
            )

            output = self.decode(target_sequence, batched_encoder_output, tgt_mask)
        else:
            with torch.no_grad():
                output = torch.zeros((batch_size, output_size, self._vocab_size)).to(
                    device
                )

                target_sequence = torch.zeros((batch_size, 1)).to(device)
                target_sequence[:, 0] = self._sos_token
                is_ended = torch.zeros(batch_size, dtype=torch.bool).to(device)

                for i in range(output_size):
                    tgt_mask = (Transformer.generate_square_subsequent_mask(i + 1)).to(
                        device
                    )
                    logits = self.decode(
                        target_sequence, batched_encoder_output, tgt_mask
                    )

                    prediction = logits.argmax(-1)[:, i]
                    target_sequence = torch.cat(
                        (target_sequence, prediction.unsqueeze(1)), dim=1
                    )
                    output[:, i, :] = logits[:, i, :]

                    is_ended = torch.logical_or(
                        is_ended, (prediction == self._eos_token)
                    )
                    if torch.count_nonzero(is_ended).item() == batch_size:
                        break

        return output.permute(1, 0, 2)
