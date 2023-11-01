from functools import partial

import torch
from torch import nn, Tensor, LongTensor
from transformers import (DebertaV2Model, DebertaV2ForMaskedLM, DebertaV2Tokenizer,
                          AutoConfig, AutoModel, AutoModelForMaskedLM)

from lm.data_utils import BatchedTextTokens
from lm.deberta_patch import patch_deberta_causal
from lm.eval_utils import metrics
from lm.utils import get_model_module

__all__ = ['CodeformerLM', 'PatchedDebertaAsCausalLM']


class CodeformerLM(nn.Module):
    def __init__(self,
                 hf_model_name: str,
                 do_random_init: bool) -> None:
        super().__init__()
        # We do not need the token embeddings for the chunk level
        modules_dict = self._get_modules(hf_model_name, do_random_init)
        self.encoder_token = modules_dict['encoder_token']
        self.encoder_chunk = modules_dict['encoder_chunk']
        self.decoder = modules_dict['decoder']
        self.chunk_sos_embedding = modules_dict['chunk_sos_embedding']
        self.register_buffer('pad_token_id', modules_dict['pad_token_id'])
        self.pad_token_id: Tensor

    def forward(self, batch: BatchedTextTokens) -> dict[str, Tensor]:
        # token_ids_chunk.shape = batch_size, max_chunks, max_tokens_per_chunk
        token_ids_chunk = batch.token_ids_chunk
        batch_size, max_chunks, max_tokens_per_chunk = token_ids_chunk.shape

        # Chunk embeddings
        token_ids_stacked = token_ids_chunk.reshape(batch_size * max_chunks, max_tokens_per_chunk)
        att_mask_chunk_tokens = batch.att_mask_chunk_tokens.resize_as(token_ids_stacked)
        token_units = self.encoder_token(input_ids=token_ids_stacked,
                                         attention_mask=att_mask_chunk_tokens).last_hidden_state
        hidden_size = token_units.shape[2]
        chunk_embs = token_units.reshape(batch_size, max_chunks, max_tokens_per_chunk, hidden_size)[:, :, 0, :]

        # Add chunk positional encodings
        if self.encoder_chunk.embeddings.position_embeddings is not None:
            # We don't need positionals for DebertaV[2,3] due to the
            # Disentangled Attention mechanism, position_embeddings attribute
            # is None in this case
            chunk_pos_ids = self.encoder_chunk.embeddings.position_ids[:, :max_chunks]
            chunk_pos_embs = self.encoder_chunk.embeddings.position_embeddings(chunk_pos_ids)
            chunk_embs = chunk_embs + chunk_pos_embs

        # Chunk representations
        chunk_units = self.encoder_chunk(inputs_embeds=chunk_embs,
                                         attention_mask=batch.att_mask_chunks).last_hidden_state

        # Decoder

        chunk_sos_emb = self.chunk_sos_embedding.reshape(1, 1, -1).repeat(batch_size, 1, 1)
        chunk_units_sos = torch.cat([chunk_sos_emb, chunk_units[:, :-1]], 1)

        decoder_input_embs = torch.zeros(batch_size,
                                         max_chunks,
                                         max_chunks + max_tokens_per_chunk,
                                         hidden_size,
                                         dtype=chunk_units.dtype,
                                         device=chunk_units.device)

        decoder_token_embs = get_model_module(self.decoder).embeddings(token_ids_chunk)
        # TODO: check computations step by step

        for sample_num in range(batch_size):
            num_chunks = len(batch.split_sizes_list[sample_num])
            for chunk_num in range(num_chunks):
                num_tokens = batch.split_sizes_list[sample_num][chunk_num]
                decoder_input_embs[sample_num, chunk_num, :chunk_num + 1] = chunk_units_sos[sample_num, :chunk_num + 1]
                decoder_input_embs[
                    sample_num,
                    chunk_num,
                    chunk_num + 1: chunk_num + 1 + num_tokens
                ] = decoder_token_embs[sample_num, chunk_num, :num_tokens]
        decoder_input_embs = decoder_input_embs.reshape(
            batch_size * max_chunks, max_chunks + max_tokens_per_chunk, hidden_size
        )
        units_stacked = get_model_module(self.decoder)(inputs_embeds=decoder_input_embs).last_hidden_state
        vocab_size = units_stacked.shape[2]
        units_stacked = units_stacked.reshape(batch_size, max_chunks, max_chunks + max_tokens_per_chunk, vocab_size)

        # Logits

        # do not predict bos and predict eos
        units_shape = [batch_size, max_chunks, max_tokens_per_chunk - 1, hidden_size]
        units_reassembled = torch.zeros(*units_shape,
                             device=units_stacked.device,
                             dtype=units_stacked.dtype)
        for sample_num in range(batch_size):
            num_chunks = len(batch.split_sizes_list[sample_num])
            for chunk_num in range(num_chunks):
                num_tokens = batch.split_sizes_list[sample_num][chunk_num]
                curr_units = units_stacked[sample_num, chunk_num, chunk_num + 1: chunk_num + 1 + num_tokens - 1]
                units_reassembled[sample_num, chunk_num, :num_tokens - 1] = curr_units
        # logits = torch.permute(logits, [0, 3, 1, 2])
        logits = self.decoder.cls(units_reassembled)
        targets = token_ids_chunk[:, :, 1:]
        return metrics(logits, targets, batch.pad_token_id)

    def _get_modules(self, hf_model_name: str, do_random_init: bool) -> dict[str: nn.Module | nn.Parameter | Tensor]:
        deberta_v2_prefix = 'microsoft/deberta-v2'
        deberta_v3_prefix = 'microsoft/deberta-v3'
        if hf_model_name[:len(deberta_v3_prefix)] in {deberta_v2_prefix, deberta_v3_prefix}:
            if do_random_init:
                cfg = AutoConfig.from_pretrained(hf_model_name)
                get_model_class = partial(AutoModel.from_config, cfg)
                get_model_class_with_lm_head = partial(AutoModelForMaskedLM.from_config, cfg)
            else:
                get_model_class = partial(DebertaV2Model.from_pretrained, hf_model_name)
                get_model_class_with_lm_head = partial(AutoModelForMaskedLM.from_pretrained, hf_model_name)
            tokenizer_class = DebertaV2Tokenizer
            patching_method = patch_deberta_causal
        else:
            # The patching method might be not needed if the model
            # supports causal masking out of the box
            patching_method = lambda x: x  # noqa: E731
            raise NotImplementedError
        encoder_token = get_model_class()
        encoder_chunk = patching_method(get_model_class())
        decoder = get_model_class_with_lm_head()
        decoder = patching_method(decoder)
        tokenizer = tokenizer_class.from_pretrained(hf_model_name)

        encoder_chunk_emb = get_model_module(encoder_chunk).embeddings.word_embeddings.weight[tokenizer.bos_token_id]
        chunk_sos_embedding = nn.Parameter(encoder_chunk_emb)

        encoder_chunk.embeddings.word_embeddings = nn.Identity()
        pad_token_id = torch.scalar_tensor(tokenizer.pad_token_id, dtype=torch.long)
        return {
            'encoder_token': encoder_token,
            'encoder_chunk': encoder_chunk,
            'decoder': decoder,
            'chunk_sos_embedding': chunk_sos_embedding,
            'pad_token_id': pad_token_id
        }

    def generate(self, input_ids: LongTensor, max_new_tokens: int = 30):
        batch_size, _ = input_ids.shape
        assert batch_size == 1
        num_predicted_tokens = torch.randint(max_new_tokens // 2, max_new_tokens, [1]).item()
        min_tok_idx = 1000
        max_tok_idx = self.decoder.embeddings.word_embeddings.weight.shape[0]
        return torch.cat([input_ids, torch.randint(min_tok_idx, max_tok_idx, [1, num_predicted_tokens])], dim=1)


class PatchedDebertaAsCausalLM(DebertaV2ForMaskedLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        patch_deberta_causal(self)


# TODO: attempt to make everything in pure torch without loops
# chunk_units_sos_rep = chunk_units_sos.reshape(batch_size, max_chunks, 1, hidden_size).repeat(1, 1, max_chunks, 1)
# ar_mask = torch.tril(torch.ones(max_chunks, max_chunks, dtype=chunk_units.dtype, device=chunk_units.device))
# chunk_units_sos_rep = chunk_units_sos_rep * ar_mask.reshape(max_chunks, max_chunks, 1)
# torch.roll
