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
    pad_to_match_linear_layer
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
        
        decoder_context_size = data_config.max_chunk_size + 2 # +2 for begin and context tokens
        decoder_config = GPT2Config(**codeformer_config.decoder_config,
                                          vocab_size=vocab_size,
                                          n_positions=decoder_context_size,)
        
        self._token_encoder = GPT2Model(token_encoder_config)
        self._split_decoder = GPT2Model(split_decoder_config)
        self._decoder = GPT2LMHeadModel(decoder_config)
        self._split_accumulator_linear = nn.Linear(self._token_encoder_context_size, 1)
    
    # def _generate_input_and_labels_for_decoder(self, input_embeddings: torch.Tensor, labels: torch.Tensor, context: torch.Tensor, split_sizes: torch.Tensor):
    #     start_token_embedding = input_embeddings[0, 0, :]
    #     input_embeddings = input_embeddings[:, 1:-1, :] # we would like to exclude bos and eos representations
    #     splits_by_batch = split_sizes.count_nonzero(dim=1).to(self._device)
    #     empty_context = torch.zeros(input_embeddings.size(-1)).to(self._device)
    #     num_splits = torch.sum(splits_by_batch)
    #     decoder_input = torch.zeros((num_splits, input_embeddings.size(-2) + 2, input_embeddings.size(-1))).to(self._device) # +2, as before, needed for start and context tokens
    #     decoder_input[:, 0] = start_token_embedding
    #     processed_labels = torch.zeros((num_splits, input_embeddings.size(-2) + 2), dtype=torch.long).to(self._device)
    #     processed_labels[:, : 2] = self._special_tokens.labels_pad_id # don't want to compute loss for bos and context tokens
    #     split_idx = 0
    #     for batch_num in range(context.size(0)):
    #         split_sum = 0
    #         for idx in range(splits_by_batch[batch_num]):
    #             if idx == 0:
    #                 decoder_input[split_idx][1] = empty_context
    #             else:
    #                 decoder_input[split_idx][1] = context[batch_num][idx - 1]
    #             split_size = split_sizes[batch_num][idx - 1]
    #             decoder_input[split_idx][2: 2 + split_size] = input_embeddings[batch_num][:split_size]
    #             processed_labels[split_idx][2: 2 + split_size] = labels[batch_num][split_sum: split_sum + split_size]
    #             split_sum += split_sizes[batch_num][idx]
    #             split_idx += 1
    #     return decoder_input, processed_labels


    # keep in mind that sum(split_sizes[i]) may be less that len(input_ids[i]) for any i
    def forward(self, batch: BatchedData) -> Tensor:
        print("batch.text_tokens", batch.text_tokens.size())
        print("batch.batch_split", batch.batch_split.size())
        splits = generate_batches_from_splits_without_last_chunk(
            input_ids=batch.text_tokens, 
            splits_size=batch.batch_split, 
            context_size=self._token_encoder_context_size,
            pad_id=self._pad_id, 
            bos_id=self._bos_id, 
            eos_id=self._eos_id,
            device=self._device,
        )
        print("splits", splits.size())
        attention_mask = (splits == self._pad_id)

        context_hidden_states = self._token_encoder(
            input_ids=splits, attention_mask=attention_mask
        ).hidden_states
        input_embeddings, context = context_hidden_states[0], context_hidden_states[-1]
        print("context", context.size())
        # cropped_context = pad_to_match_linear_layer(
        #     x=context,
        #     split_sizes=batch.batch_split,
        #     context_size=self._token_encoder_context_size
        # )
        # print("cropped_context before linear", cropped_context.size())
        # cropped_context = cropped_context.permute(0, 1, 3, 2)
        # cropped_context = self._split_accumulator_linear(cropped_context).squeeze(dim=3)
        # print("cropped_context after linear", cropped_context.size())
        # cropped_context = self._split_decoder(inputs_embeds=cropped_context).last_hidden_state
        # print("cropped_context after decoder", cropped_context.size())
        # context, processed_labels = self._generate_input_and_labels_for_decoder(input_embeddings, labels, context, split_sizes)
        # print("context", context.size())
        # print("processed_labels", processed_labels.size())
        # lm_casual_output = self._decoder(inputs_embeds=context, labels=processed_labels)
        # return lm_casual_output.logits, lm_casual_output.loss, processed_labels
        return None
