import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.nn import Linear
from transformers import BigBirdModel, BigBirdConfig

from jetnn.data_processing.base_data_classes import (
    BatchedData,
)
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary


class BigBirdDecoder(nn.Module):
    def __init__(self, config: DictConfig, vocab: Vocabulary):
        super().__init__()
        self._vocab_size = len(vocab)
        self._pad_token = vocab.pad_id()
        self._sos_token = vocab.bos_id()
        self._eos_token = vocab.eos_id()

        big_bird_config = BigBirdConfig(
            vocab_size=self._vocab_size,
            hidden_size=config.d_model,
            num_attention_heads=config.nhead,
            num_hidden_layers=config.num_layers,
            hidden_dropout_prob=config.dropout,
            attention_type="original_full",
            add_cross_attention=True,
            is_decoder=True,
        )
        self._decoder = BigBirdModel(big_bird_config)

        self._linear = Linear(config.d_model, self._vocab_size)

    def decode(self, target_sequence: Tensor, batched_encoder_output: Tensor) -> Tensor:
        tgt_key_padding_mask = target_sequence == self._pad_token

        decoded = self._decoder(
            input_ids=target_sequence,
            attention_mask=tgt_key_padding_mask,
            encoder_hidden_states=batched_encoder_output,
        )
        return self._linear(decoded.last_hidden_state)

    def forward(
        self, batched_encoder_output: Tensor, batch: BatchedData, step: str
    ) -> Tensor:
        device = batched_encoder_output.device
        target_sequence = batch.label_tokens.permute(1, 0)
        batch_size = target_sequence.shape[0]
        output_size = target_sequence.shape[1]

        if step != "test":

            output = self.decode(target_sequence, batched_encoder_output)
        else:
            with torch.no_grad():
                output = torch.zeros((batch_size, output_size, self._vocab_size)).to(
                    device
                )

                target_sequence = torch.zeros((batch_size, 1)).to(device)
                target_sequence[:, 0] = self._sos_token
                is_ended = torch.zeros(batch_size, dtype=torch.bool).to(device)

                for i in range(output_size):
                    logits = self.decode(target_sequence, batched_encoder_output)

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
