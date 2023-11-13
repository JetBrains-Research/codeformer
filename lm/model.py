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
                 do_random_init: bool,
                 num_context_chunks: int = 1,
                 padded_to_max_chunks_per_sample: bool = False) -> None:
        super().__init__()
        # We do not need the token embeddings for the chunk level
        modules_dict = self._get_modules(hf_model_name, do_random_init)
        self.encoder_token = modules_dict['encoder_token']
        self.encoder_chunk = modules_dict['encoder_chunk']
        self.decoder = modules_dict['decoder']
        self.chunk_sos_embedding = modules_dict['chunk_sos_embedding']
        self.padded_to_max_chunks_per_sample = padded_to_max_chunks_per_sample

        self.num_context_chunks = num_context_chunks
        self.register_buffer('pad_token_id', modules_dict['pad_token_id'])
        self.pad_token_id: Tensor

    def forward(self, batch: BatchedTextTokens) -> dict[str, Tensor]:
        # token_ids_chunk.shape = batch_size, max_chunks, max_tokens_per_chunk
        # token_ids_chunk_bos_eos =
        batch_size = batch.batch_size
        max_chunks = batch.max_chunks_per_sample
        max_tokens_per_chunk = batch.max_tokens_per_chunk

        # Chunk embeddings

        if self.padded_to_max_chunks_per_sample:
            # NOTE: performance, move next two lines to the collate function
            token_ids_stacked = batch.token_ids_chunk_bos_eos.reshape(batch_size * max_chunks, max_tokens_per_chunk)
            att_mask_chunk_tokens = batch.att_mask_chunk_tokens.resize_as(token_ids_stacked)
            token_units = self.encoder_token(input_ids=token_ids_stacked,
                                             attention_mask=att_mask_chunk_tokens).last_hidden_state
            hidden_size = token_units.shape[2]
            chunk_embs = token_units.reshape(batch_size, max_chunks, max_tokens_per_chunk, hidden_size)[:, :, 0, :]
        else:
            chunk_embs = self.encoder_token(batch.token_ids_chunk_stacked_bos_eos).last_hidden_state[:, 0, :]
            chunk_embs = self.assemble_chunk_representations_from_stacked(chunk_embs, batch.chunk_sizes_tensor)
        # TODO: check both cases above on bos chunk

        chunk_embs = self._add_positional_encoding(chunk_embs)

        # Chunk representations
        chunk_units = self.encoder_chunk(inputs_embeds=chunk_embs,
                                         attention_mask=batch.att_mask_chunks).last_hidden_state

        decoder_chunk_embs = self.get_chunk_representations_for_decoder(chunk_units, batch.chunk_sizes_tensor)
        # Decoder

        decoder_token_embs = get_model_module(self.decoder).embeddings(batch.decoder_inp_tok_ids)
        # TODO: check computations step by step
        decoder_input_embs = torch.cat([decoder_chunk_embs, decoder_token_embs], dim=1)
        decoder_units = get_model_module(self.decoder)(inputs_embeds=decoder_input_embs).last_hidden_state
        decoder_units = decoder_units[:, self.num_context_chunks:]
        logits = self.decoder.cls(decoder_units)
        return metrics(logits, batch.decoder_targ_tok_ids, batch.pad_token_id)

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

    def _add_positional_encoding(self, chunk_embs: Tensor) -> Tensor:
        if self.encoder_chunk.embeddings.position_embeddings is not None:
            # We don't need positionals for DebertaV[2,3] due to the
            # Disentangled Attention mechanism, position_embeddings attribute
            # is None in this case
            max_chunks = chunk_embs.shape[1]
            chunk_pos_ids = self.encoder_chunk.embeddings.position_ids[:, :max_chunks]
            chunk_pos_embs = self.encoder_chunk.embeddings.position_embeddings(chunk_pos_ids)
            chunk_embs = chunk_embs + chunk_pos_embs
        return chunk_embs

    def get_chunk_representations_for_decoder(self, chunk_units: Tensor, chunk_sizes: LongTensor) -> Tensor:
        # chunk_units.shape = batch_size, max_chunks_per_sample, hidden_size
        #   note that chunk units have the first item in the sequence equal to the padding emb
        # outputs.shape = num_chunks_total, num_context_chunks, hidden_size
        num_chunks = (chunk_sizes > 0).count_nonzero()
        batch_size, max_chunks_per_sample, hidden_size = chunk_units.shape
        output_tensor = torch.zeros(num_chunks, self.num_context_chunks, hidden_size,
                                    device=chunk_units.device, dtype=chunk_units.dtype)
        count = 0
        for sample_num in range(batch_size):
            for chunk_num in range(max_chunks_per_sample):
                if chunk_sizes[sample_num, chunk_num] > 0:
                    indices = [max(0, n) for n in range(chunk_num - self.num_context_chunks, chunk_num)]
                    output_tensor[count] = chunk_units[sample_num, indices]
                    count += 1
        return output_tensor

    def assemble_chunk_representations_from_stacked(self,
                                                    stacked_chunk_units: Tensor,
                                                    chunk_sizes: LongTensor) -> Tensor:
        # stacked_chunk_units.shape = num_chunks_total, hidden_size
        # chunk_sizes.shape = batch_size, max_chunks_per_sample
        batch_size, max_chunks_per_sample = chunk_sizes.shape
        hidden_size = stacked_chunk_units.shape[1]
        # -1 because of sos
        chunk_units = torch.zeros(batch_size, max_chunks_per_sample - 1, hidden_size,
                                  device=stacked_chunk_units.device, dtype=stacked_chunk_units.dtype)
        count = 0
        # NOTE: can be precomputed
        num_chunks = (chunk_sizes > 0).sum(1).tolist()
        for sample_num in range(batch_size):
            cur_num_chunks = num_chunks[sample_num] - 1
            chunk_units[sample_num, :cur_num_chunks] = stacked_chunk_units[count: count + cur_num_chunks]
            count += cur_num_chunks
        chunk_sos_emb = self.chunk_sos_embedding.reshape(1, 1, -1).repeat(batch_size, 1, 1)
        chunk_units_sos = torch.cat([chunk_sos_emb, chunk_units], 1)
        return chunk_units_sos

    def assemble_chunk_representations_from_uniform(self,
                                                    chunk_units: Tensor,
                                                    chunk_sizes: LongTensor) -> Tensor:
        batch_size = chunk_sizes.shape[0]
        chunk_sos_emb = self.chunk_sos_embedding.reshape(1, 1, -1).repeat(batch_size, 1, 1)
        chunk_units_sos = torch.cat([chunk_sos_emb, chunk_units[:, :-1]], 1)
        return chunk_units_sos


class PatchedDebertaAsCausalLM(DebertaV2ForMaskedLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        patch_deberta_causal(self)


# TODO: attempt to make everything in pure torch without loops
# chunk_units_sos_rep = chunk_units_sos.reshape(batch_size, max_chunks, 1, hidden_size).repeat(1, 1, max_chunks, 1)
# ar_mask = torch.tril(torch.ones(max_chunks, max_chunks, dtype=chunk_units.dtype, device=chunk_units.device))
# chunk_units_sos_rep = chunk_units_sos_rep * ar_mask.reshape(max_chunks, max_chunks, 1)
# torch.roll
