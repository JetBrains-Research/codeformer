from typing import Any

import torch
from omegaconf import DictConfig
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

# notation:
# batch_size - initial batch size as stated in config
# h_dim - hidden dimension
# max_chunks_number - as stated in config
# max_chunk_size - as stated in config
# n_chunks - total number of chunks from all the samples
# max_n_chunks - the maximum number of chunks from all the samples (needed for padding)
# n_layers - number of layers inside transformer



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
        # generate chunks according to text_data and splits sizes
        chunks, _ = generate_batches_from_splits_without_last_chunk(
            input_ids=batch.text_tokens, 
            splits_size=batch.batch_split, 
            context_size=self._token_encoder_context_size,
            pad_id=self._pad_id, 
            bos_id=self._bos_id, 
            eos_id=self._eos_id,
            device=self._device,
        )
        # splits: (n_chunks, max_chunk_size + 2)
        
        # generate attention mask according chunks
        attention_mask = (chunks == self._pad_id)
        # attention_mask: (n_chunks, max_chunk_size + 2)

        # pass chunks to the encoder
        context_hidden_states = self._token_encoder(
            input_ids=chunks, attention_mask=attention_mask
        ).hidden_states
        # attention_mask: (n_layers + 1, n_chunks, max_chunk_size + 2, h_dim)

        # extract input embeddings and chunks representations after encoder
        input_embeddings, context = context_hidden_states[0], context_hidden_states[-1]
        # input_embeddings: (n_chunks, max_chunk_size + 2, h_dim)
        # context: (n_chunks, max_chunk_size + 2, h_dim)

        # make padding to apply linear layer
        cropped_context = pad_to_match_linear_layer(
            x=context,
            split_sizes=batch.batch_split,
            context_size=self._token_encoder_context_size,
            pad_id=self._pad_id,
            device=self._device,
            last_chunk_omitted=True,
        )
        # cropped_context: (batch_size, max_n_chunks, max_chunk_size + 2, h_dim)

        # reshape dimensions to apply linear layer
        cropped_context = cropped_context.permute(0, 1, 3, 2)
        # cropped_context: (batch_size, max_n_chunks, h_dim, max_chunk_size + 2)

        # apply linear layer to get a representation of the whole chunks
        cropped_context = self._chunk_accumulator_linear(cropped_context).squeeze(dim=3)
        # cropped_context: (batch_size, max_n_chunks, h_dim)

        # concatenate with embeddings of the BOS token to aplly decoder
        cropped_context = concat_with_bos_embedding(cropped_context, input_embeddings[0][0])
        # cropped_context: (batch_size, max_n_chunks + 1, h_dim)
        
        # generate "chunk level" attention mask for the decoder
        attention_mask = generate_chunk_level_attention_mask(
            split_sizes=batch.batch_split,
            pad_id=self._pad_id,
            device=self._device,
        )
        # attention_mask: (batch_size, max_n_chunks + 1)

        # apply decoder to chunks
        cropped_context = self._chunk_decoder(inputs_embeds=cropped_context, attention_mask=attention_mask).last_hidden_state
        # cropped_context: (batch_size, max_n_chunks + 1, h_dim)

        # generate context and labels for the second encoder
        context, labels, hf_labels, cross_attention, cross_attention_mask = generate_context_for_last_decoder(
            input_ids=chunks,
            input_embeddings=input_embeddings,
            chunk_representations=cropped_context,
            splits_size=batch.batch_split,
            pad_id=self._pad_id,
            device=self._device,
        )
        # context: (n_chunks, max_chunk_size + 3, h_dim)
        # labels: (n_chunks, max_chunk_size + 3)
        # hf_labels: (n_chunks, max_chunk_size + 3)
        
        # apply second decoder
        result = self._decoder(inputs_embeds=context, labels=hf_labels)
        # context: (n_chunks, max_chunk_size + 3, h_dim)

        return result.logits, labels, result.loss
