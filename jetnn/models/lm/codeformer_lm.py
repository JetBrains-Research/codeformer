from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, Tensor
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
from jetnn.data_processing.base_data_classes import (
    BatchedData
)
from jetnn.models.utils.codeformer_utils import (
    generate_batches_from_splits_without_last_chunk,
    pad_to_match_linear_layer,
    concat_with_bos_embedding,
    generate_chunk_level_attention_mask,
    generate_context_for_last_decoder,
)

class CodeformerLM(nn.Module):
    def __init__(self, codeformer_config: DictConfig, data_config: DictConfig, vocab: Vocabulary):
        super().__init__()

        vocab_size = len(vocab)
        
        self._vocab = vocab
        self._pad_id = vocab.pad_id()
        self._bos_id = vocab.bos_id()
        self._eos_id = vocab.eos_id()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._token_encoder_context_size = data_config.max_chunk_size + 2 # +2 for begin and end tokens
        token_encoder_config = GPT2Config(**codeformer_config.token_encoder_config,
                                          vocab_size=vocab_size,
                                          n_positions=self._token_encoder_context_size,)
        
        split_decoder_context_size = data_config.max_chunks_number + 1 # +1 for begin token
        split_decoder_config = GPT2Config(**codeformer_config.split_decoder_config,
                                          vocab_size=vocab_size,
                                          n_positions=split_decoder_context_size,)
        
        decoder_context_size = data_config.max_chunk_size + 3 # +3 for begin, end and context tokens
        decoder_config = GPT2Config(**codeformer_config.decoder_config,
                                          vocab_size=vocab_size,
                                          n_positions=decoder_context_size,)
        
        self._token_encoder = GPT2Model(token_encoder_config)
        self._chunk_decoder = GPT2Model(split_decoder_config)
        self._decoder = GPT2LMHeadModel(decoder_config)
        self._chunk_accumulator_linear = nn.Linear(self._token_encoder_context_size, 1)

    # keep in mind that sum(split_sizes[i]) may be less that len(input_ids[i]) for any i
    # TODO: add samples for last chunk!
    def forward(self, batch: BatchedData) -> Tensor:
        splits, last_chunk_tokens = generate_batches_from_splits_without_last_chunk(
            input_ids=batch.text_tokens, 
            splits_size=batch.batch_split, 
            context_size=self._token_encoder_context_size,
            pad_id=self._pad_id, 
            bos_id=self._bos_id, 
            eos_id=self._eos_id,
            device=self._device,
        )
        attention_mask = (splits == self._pad_id)

        context_hidden_states = self._token_encoder(
            input_ids=splits, attention_mask=attention_mask
        ).hidden_states
        input_embeddings, context = context_hidden_states[0], context_hidden_states[-1]

        cropped_context = pad_to_match_linear_layer(
            x=context,
            split_sizes=batch.batch_split,
            context_size=self._token_encoder_context_size,
            device=self._device,
            last_chunk_omitted=True,
        )
        cropped_context = cropped_context.permute(0, 1, 3, 2)
        cropped_context = self._chunk_accumulator_linear(cropped_context).squeeze(dim=3)
        cropped_context = concat_with_bos_embedding(cropped_context, input_embeddings[0][0])
        
        attention_mask = generate_chunk_level_attention_mask(
            split_sizes=batch.batch_split,
            device=self._device,
        )
        cropped_context = self._chunk_decoder(inputs_embeds=cropped_context, attention_mask=attention_mask).last_hidden_state

        context, labels, cross_attention, cross_attention_mask = generate_context_for_last_decoder(
            input_ids=splits,
            input_embeddings=input_embeddings,
            chunk_representations=cropped_context,
            splits_size=batch.batch_split,
            pad_id=self._pad_id,
            device=self._device,
        )

        result = self._decoder(inputs_embeds=context)

        return result.logits, labels
