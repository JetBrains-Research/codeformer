from functools import partial

import torch
from torch import nn, Tensor, LongTensor
from transformers import (DebertaV2Model, DebertaV2ForMaskedLM, DebertaV2Tokenizer,
                          AutoConfig, AutoModel, AutoModelForMaskedLM)

from lm.data_utils import BatchedTextTokens
from lm.deberta_patch import patch_deberta_causal
from lm.eval_utils import metrics
from lm.utils import (get_model_module, assemble, disassemble, assemble_decoder_inputs,
                      expand_filler, put_token_embeddings_at_specified_positions,
                      prepare_token_ids_for_decoder)

__all__ = ['CodeformerLM', 'PatchedDebertaAsCausalLM']


class CodeformerLM(nn.Module):
    def __init__(self,
                 hf_model_name: str,
                 do_random_init: bool,
                 num_context_chunks: int = 1) -> None:
        super().__init__()
        # We do not need the token embeddings for the chunk level
        modules_dict = self._get_modules(hf_model_name, do_random_init)
        self.encoder_token = modules_dict['encoder_token']
        self.encoder_chunk = modules_dict['encoder_chunk']
        self.decoder = modules_dict['decoder']
        self.chunk_sos_embedding = modules_dict['chunk_sos_embedding']

        self.num_context_chunks = num_context_chunks
        self.register_buffer('pad_token_id', modules_dict['pad_token_id'])
        self.register_buffer('bos_token_id', modules_dict['bos_token_id'])
        self.register_buffer('eos_token_id', modules_dict['eos_token_id'])
        self.pad_token_id: Tensor
        self.bos_token_id: Tensor
        self.eos_token_id: Tensor

    # def forward(self, batch: BatchedTextTokens) -> dict[str, Tensor]:
    #     # token_ids_chunk.shape = batch_size, max_chunks, max_tokens_per_chunk
    #     token_ids_chunk = batch.token_ids_chunk
    #     batch_size, max_chunks, max_tokens_per_chunk = token_ids_chunk.shape
    #
    #     # Chunk embeddings
    #     token_ids_stacked = token_ids_chunk.reshape(batch_size * max_chunks, max_tokens_per_chunk)
    #     att_mask_chunk_tokens = batch.att_mask_chunk_tokens.resize_as(token_ids_stacked)
    #     token_units = self.encoder_token(input_ids=token_ids_stacked,
    #                                      attention_mask=att_mask_chunk_tokens).last_hidden_state
    #     hidden_size = token_units.shape[2]
    #     # chunk_embs.shape = batch_size, max_chunks, hidden_size
    #     chunk_embs = token_units.reshape(batch_size, max_chunks, max_tokens_per_chunk, hidden_size)[:, :, 0, :]
    #
    #     chunk_embs = self._add_positional_encoding(chunk_embs)
    #
    #     # Chunk representations
    #     chunk_units = self.encoder_chunk(inputs_embeds=chunk_embs,
    #                                      attention_mask=batch.att_mask_chunks).last_hidden_state
    #
    #     # Decoder
    #
    #     chunk_sos_emb = self.chunk_sos_embedding.reshape(1, 1, -1).repeat(batch_size, 1, 1)
    #     chunk_units_sos = torch.cat([chunk_sos_emb, chunk_units[:, :-1]], 1)
    #
    #     decoder_input_embs = torch.zeros(batch_size,
    #                                      max_chunks,
    #                                      max_chunks + max_tokens_per_chunk,
    #                                      hidden_size,
    #                                      dtype=chunk_units.dtype,
    #                                      device=chunk_units.device)
    #
    #     decoder_token_embs = get_model_module(self.decoder).embeddings(token_ids_chunk)
    #     # TODO: check computations step by step
    #
    #     for sample_num in range(batch_size):
    #         num_chunks = len(batch.split_sizes_list[sample_num])
    #         for chunk_num in range(num_chunks):
    #             num_tokens = batch.split_sizes_list[sample_num][chunk_num]
    #             decoder_input_embs[sample_num, chunk_num, :chunk_num + 1] = chunk_units_sos[sample_num, :chunk_num + 1]
    #             decoder_input_embs[
    #                 sample_num,
    #                 chunk_num,
    #                 chunk_num + 1: chunk_num + 1 + num_tokens
    #             ] = decoder_token_embs[sample_num, chunk_num, :num_tokens]
    #     decoder_input_embs = decoder_input_embs.reshape(
    #         batch_size * max_chunks, max_chunks + max_tokens_per_chunk, hidden_size
    #     )
    #     units_stacked = get_model_module(self.decoder)(inputs_embeds=decoder_input_embs).last_hidden_state
    #     vocab_size = units_stacked.shape[2]
    #     units_stacked = units_stacked.reshape(batch_size, max_chunks, max_chunks + max_tokens_per_chunk, vocab_size)
    #
    #     # Logits
    #
    #     # do not predict bos and predict eos
    #     units_shape = [batch_size, max_chunks, max_tokens_per_chunk - 1, hidden_size]
    #     units_reassembled = torch.zeros(*units_shape,
    #                                     device=units_stacked.device,
    #                                     dtype=units_stacked.dtype)
    #     for sample_num in range(batch_size):
    #         num_chunks = len(batch.split_sizes_list[sample_num])
    #         for chunk_num in range(num_chunks):
    #             num_tokens = batch.split_sizes_list[sample_num][chunk_num]
    #             curr_units = units_stacked[sample_num, chunk_num, chunk_num + 1: chunk_num + 1 + num_tokens - 1]
    #             units_reassembled[sample_num, chunk_num, :num_tokens - 1] = curr_units
    #     # logits = torch.permute(logits, [0, 3, 1, 2])
    #     logits = self.decoder.cls(units_reassembled)
    #     targets = token_ids_chunk[:, :, 1:]
    #     return metrics(logits, targets, batch.pad_token_id)

    def forward(self,
                batch: BatchedTextTokens) -> dict[str: Tensor]:
               # token_ids: LongTensor,  # batch_size * max_number_of_chunks, max_chunk len
               # chunk_sizes: LongTensor) -> Tensor:
        # Paddings
        chunk_sizes = batch.split_sizes_tensor
        token_ids = batch.token_ids
        batch_size, max_chunks_per_sample = chunk_sizes.shape
        num_chunks_total = chunk_sizes.count_nonzero()

        encoder_input_ids = disassemble(token_ids.detach(),
                                        chunk_sizes.detach(),
                                        self.pad_token_id,
                                        self.bos_token_id,
                                        self.eos_token_id)
        device = token_ids.device

        att_mask = (encoder_input_ids != self.pad_token_id).float().detach()
        print(att_mask.sum(1))
        encoder_hidden_states = self.encoder_token(input_ids=encoder_input_ids.detach(),
                                                   attention_mask=att_mask).last_hidden_state
        encoder_hidden_states = encoder_hidden_states[:, 0, :]
        chunk_encoder_inputs = assemble(encoder_hidden_states, chunk_sizes, 0.0)
        chunk_encoder_inputs = self._add_positional_encoding(chunk_encoder_inputs)

        chunk_att_mask = (chunk_sizes > 0)[:, :-1].float()
        # print(chunk_att_mask.shape, chunk_encoder_inputs.shape)
        chunk_encoder_outputs = self.encoder_chunk(inputs_embeds=chunk_encoder_inputs[:, :-1], attention_mask=chunk_att_mask).last_hidden_state
        chunk_encoder_outputs = self._prepend_with_chunk_bos(chunk_encoder_outputs)

        token_ids_for_decoder, lens, lens_prev = prepare_token_ids_for_decoder(token_ids,
                                                                               chunk_sizes,
                                                                               self.pad_token_id,
                                                                               self.bos_token_id,
                                                                               self.eos_token_id)
        decoder_token_embeddings = get_model_module(self.decoder).embeddings(token_ids_for_decoder.detach().clone())
        decoder_token_embeddings = decoder_token_embeddings * (token_ids_for_decoder != self.pad_token_id).unsqueeze(2).float()
        max_len = torch.max(lens + self.num_context_chunks).detach()

        nums_context_chunks = torch.full(chunk_sizes.shape, self.num_context_chunks, device=device) * (chunk_sizes > 0).long()
        from_starting_points = torch.arange(max_chunks_per_sample, device=device).view(1, max_chunks_per_sample).repeat(batch_size, 1)
        from_starting_points = torch.max(from_starting_points - self.num_context_chunks + 1, torch.scalar_tensor(0)).long()
        to_starting_points = torch.zeros_like(from_starting_points)
        decoder_chunk_inputs = assemble_decoder_inputs(chunk_encoder_outputs,
                                                       nums_context_chunks.detach(),
                                                       from_starting_points.detach(),
                                                       to_starting_points.detach(),
                                                       max_len.detach())

        starting_positions = torch.full([num_chunks_total], self.num_context_chunks, device=device).detach()
        token_embs = put_token_embeddings_at_specified_positions(decoder_token_embeddings, lens, starting_positions, max_len)
        # print(token_embs.shape)
        # print(decoder_chunk_inputs.shape)
        input_reprs = token_embs + decoder_chunk_inputs
        output_reprs = get_model_module(self.decoder)(inputs_embeds=input_reprs[:, :-1]).last_hidden_state
        logits = self.decoder.cls(output_reprs[:, 1:])
        # TODO: make work for chunks > 1
        num_toks_decoder = token_ids_for_decoder.shape[1]
        # -1 because of bos
        to_predict_mask = torch.arange(1, num_toks_decoder + 1 - 1, device=device).view(1, num_toks_decoder - 1).repeat(num_chunks_total, 1)
        # print(to_predict_mask.shape, lens_prev.shape, lens.shape)
        # + 1 for skipped chunks
        to_predict_mask = (to_predict_mask >= (lens_prev[chunk_sizes > 0].view(-1, 1) + 1)) & (to_predict_mask < (lens.view(-1, 1) + 1 - 1))
        # print(to_predict_mask.long())
        targets = token_ids_for_decoder[:, 1:]
        targets[~to_predict_mask] = self.pad_token_id
        return metrics(logits, targets, self.pad_token_id)

    def _add_positional_encoding(self, chunk_embs: Tensor) -> Tensor:
        # Add chunk positional encodings
        if self.encoder_chunk.embeddings.position_embeddings is not None:
            # We don't need positionals for DebertaV[2,3] due to the
            # Disentangled Attention mechanism, position_embeddings attribute
            # is None in this case
            chunk_pos_ids = self.encoder_chunk.embeddings.position_ids[:, :chunk_embs.shape[1]]
            chunk_pos_embs = self.encoder_chunk.embeddings.position_embeddings(chunk_pos_ids)
            chunk_embs = chunk_embs + chunk_pos_embs
        return chunk_embs

    def _prepend_with_chunk_bos(self, chunk_embs: Tensor) -> Tensor:
        chunk_bos_expanded = expand_filler(self.chunk_sos_embedding, chunk_embs.shape).to(chunk_embs.device)
        return torch.cat([chunk_bos_expanded, chunk_embs], dim=1)

    def generate(self, input_ids: LongTensor, max_new_tokens: int = 30):
        batch_size, _ = input_ids.shape
        assert batch_size == 1
        num_predicted_tokens = torch.randint(max_new_tokens // 2, max_new_tokens, [1]).item()
        min_tok_idx = 1000
        max_tok_idx = self.decoder.embeddings.word_embeddings.weight.shape[0]
        return torch.cat([input_ids, torch.randint(min_tok_idx, max_tok_idx, [1, num_predicted_tokens])], dim=1)

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
        bos_token_id = torch.scalar_tensor(tokenizer.bos_token_id, dtype=torch.long)
        eos_token_id = torch.scalar_tensor(tokenizer.eos_token_id, dtype=torch.long)
        return {
            'encoder_token': encoder_token,
            'encoder_chunk': encoder_chunk,
            'decoder': decoder,
            'chunk_sos_embedding': chunk_sos_embedding,
            'pad_token_id': pad_token_id,
            'bos_token_id': bos_token_id,
            'eos_token_id': eos_token_id
        }

class PatchedDebertaAsCausalLM(DebertaV2ForMaskedLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        patch_deberta_causal(self)


# TODO: attempt to make everything in pure torch without loops
# chunk_units_sos_rep = chunk_units_sos.reshape(batch_size, max_chunks, 1, hidden_size).repeat(1, 1, max_chunks, 1)
# ar_mask = torch.tril(torch.ones(max_chunks, max_chunks, dtype=chunk_units.dtype, device=chunk_units.device))
# chunk_units_sos_rep = chunk_units_sos_rep * ar_mask.reshape(max_chunks, max_chunks, 1)
# torch.roll
