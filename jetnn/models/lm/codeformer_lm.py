from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
from jetnn.data_processing.tasks.language_modeling import (
    BatchedTextTokens
)

class CodeformerLM(nn.Module):
    def __init__(self, codeformer_config: DictConfig, vocab):
        super().__init__()
        vocab_size = len(vocab)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._special_tokens = codeformer_config.tokenizer_special_tokens
        self._splits = codeformer_config.splits

        self._token_encoder_context_size = codeformer_config.splits.max_split_size
        if self._splits.add_begin_end_tokens_in_splits:
            self._token_encoder_context_size += 2
        token_encoder_config = GPT2Config(**codeformer_config.token_encoder_config,
                                          vocab_size=vocab_size,
                                          n_positions=self._token_encoder_context_size)
        
        split_encoder_config = GPT2Config(**codeformer_config.split_encoder_config,
                                          vocab_size=vocab_size,
                                          n_positions=self._splits.max_splits_number)
        
        # we add 2 here because we concatenate the current split tokens with context token and begin token
        decoder_context_size = self._splits.max_split_size + 2
        decoder_config = GPT2Config(**codeformer_config.decoder_config,
                                          vocab_size=vocab_size,
                                          n_positions=decoder_context_size)
        
        self._token_encoder = GPT2Model(token_encoder_config)
        self._split_encoder = GPT2Model(split_encoder_config)
        self._decoder = GPT2LMHeadModel(decoder_config)
        self._split_accumulator_linear = nn.Linear(self._token_encoder_context_size, 1)

    def _generate_samples_from_splits(self, input_ids: torch.Tensor, splits_size: torch.Tensor):
        splits_length = (splits_size != self._special_tokens.pad_id).count_nonzero(dim=1)
        num_splits = torch.sum(splits_length).item()
        result = torch.zeros((num_splits, self._token_encoder_context_size), dtype=torch.long).to(self._device)
        result_idx = 0
        for batch_idx in range(len(splits_size)):
            p_sum = 0
            current_split = splits_size[batch_idx]
            for split_idx in range(splits_length[batch_idx].item()):
                current_split_size = current_split[split_idx]
                start_pos = 0
                if self._splits.add_begin_end_tokens_in_splits:
                    result[result_idx][0] = self._special_tokens.bos_id
                    result[result_idx][current_split_size + 1] = self._special_tokens.eos_id
                    start_pos = 1
                result[result_idx][start_pos : start_pos + current_split_size] = input_ids[batch_idx][
                    p_sum : p_sum + current_split_size
                ]
                p_sum += current_split_size
                result_idx += 1
        return result
    
    # TODO: think about how we handle zeroes in linear layer 
    # (in case not all the splits are not the same size what is nearly always true)
    def _pad_to_match_linear_layer(self, x: torch.Tensor, split_sizes: torch.Tensor):
        batch_split = split_sizes.count_nonzero(dim=1)
        max_splits = torch.max(batch_split).item()
        num_splits = batch_split.size()[0]
        result = torch.zeros(
            (num_splits, max_splits, self._token_encoder_context_size, x.shape[2])
        ).to(self._device)
        p_sum = 0
        for i, split in enumerate(batch_split):
            result[i][:split, : x.shape[1]] = x[p_sum : p_sum + split]
            p_sum += split
        return result
    
    def _generate_input_and_labels_for_decoder(self, input_embeddings: torch.Tensor, labels: torch.Tensor, context: torch.Tensor, split_sizes: torch.Tensor):
        start_token_embedding = input_embeddings[0, 0, :]
        input_embeddings = input_embeddings[:, 1:-1, :] # we would like to exclude bos and eos representations
        splits_by_batch = split_sizes.count_nonzero(dim=1).to(self._device)
        empty_context = torch.zeros(input_embeddings.size(-1)).to(self._device)
        num_splits = torch.sum(splits_by_batch)
        decoder_input = torch.zeros((num_splits, input_embeddings.size(-2) + 2, input_embeddings.size(-1))).to(self._device) # +2, as before, needed for start and context tokens
        decoder_input[:, 0] = start_token_embedding
        processed_labels = torch.zeros((num_splits, input_embeddings.size(-2) + 2), dtype=torch.long).to(self._device)
        processed_labels[:, : 2] = self._special_tokens.labels_pad_id # don't want to compute loss for bos and context tokens
        split_idx = 0
        for batch_num in range(context.size(0)):
            split_sum = 0
            for idx in range(splits_by_batch[batch_num]):
                if idx == 0:
                    decoder_input[split_idx][1] = empty_context
                else:
                    decoder_input[split_idx][1] = context[batch_num][idx - 1]
                split_size = split_sizes[batch_num][idx - 1]
                decoder_input[split_idx][2: 2 + split_size] = input_embeddings[batch_num][:split_size]
                processed_labels[split_idx][2: 2 + split_size] = labels[batch_num][split_sum: split_sum + split_size]
                split_sum += split_sizes[batch_num][idx]
                split_idx += 1
        return decoder_input, processed_labels


    # keep in mind that sum(split_sizes[i]) may be less that len(input_ids[i]) for any i
    def forward(
            self, batch: BatchedTextTokens, step: str
    ):
        
        # splits = self._generate_samples_from_splits(input_ids, split_sizes)
        # attention_mask = (splits == self._special_tokens.pad_id)
        # context_hidden_states = self._token_encoder(
        #     input_ids=splits, attention_mask=attention_mask
        # ).hidden_states
        # input_embeddings, context = context_hidden_states[0], context_hidden_states[-1]
        # context = self._pad_to_match_linear_layer(context, split_sizes)
        # context = context.permute(0, 1, 3, 2)
        # context = self._split_accumulator_linear(context).squeeze(dim=3)
        # # square mask has to be applied here automatically
        # context = self._split_encoder(inputs_embeds=context).last_hidden_state 
        # context, processed_labels = self._generate_input_and_labels_for_decoder(input_embeddings, labels, context, split_sizes)
        # print("context", context.size())
        # print("processed_labels", processed_labels.size())
        # lm_casual_output = self._decoder(inputs_embeds=context, labels=processed_labels)
        # return lm_casual_output.logits, lm_casual_output.loss, processed_labels
        return None
