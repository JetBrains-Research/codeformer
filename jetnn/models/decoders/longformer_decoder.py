import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.nn import Linear
from transformers import LEDModel, LEDConfig

from jetnn.data_processing.plain_code_method.labeled_plain_code import (
    BatchedLabeledCodeTokens,
)
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary


class MethodNameBigBirdDecoder(nn.Module):
    def __init__(self, config: DictConfig, vocab: Vocabulary):
        super().__init__()
        self._vocab_size = len(vocab)
        self._pad_token = vocab.pad_id()
        self._sos_token = vocab.bos_id()
        self._eos_token = vocab.eos_id()

        longformer_config = LEDConfig(
            vocab_size=self._vocab_size,
            d_model=config.d_model,
            encoder_layers=0,
            decoder_layers=config.num_layers,
            encoder_attention_heads=0,
            decoder_attention_heads=config.nhead,
            hidden_dropout_prob=config.dropout,
        )
        self._decoder = LEDModel(longformer_config)

        self._linear = Linear(config.d_model, self._vocab_size)

    def decode(self, target_sequence: Tensor, batched_encoder_output: Tensor) -> Tensor:
        tgt_key_padding_mask = target_sequence == self._pad_token

        decoded = self._decoder(
            inputs_embeds=batched_encoder_output,
            decoder_input_ids=target_sequence,
            decoder_attention_mask=tgt_key_padding_mask,
        )
        return self._linear(decoded.last_hidden_state)

    def forward(
        self, batched_encoder_output: Tensor, batch: BatchedLabeledCodeTokens, step: str
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
