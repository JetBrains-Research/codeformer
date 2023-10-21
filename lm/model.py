import torch
from torch import nn, Tensor
from transformers import DebertaV2Model, DebertaV2Tokenizer

from lm.data_util import BatchedTextTokens
from lm.deberta_patch import patch_deberta_causal


class CodeformerLM(nn.Module):
    def __init__(self,
                 hf_model_name: str) -> None:
        super().__init__()
        deberta_v2_prefix = 'microsoft/deberta-v2'
        deberta_v3_prefix = 'microsoft/deberta-v3'
        if hf_model_name[:len(deberta_v3_prefix)] in {deberta_v2_prefix, deberta_v3_prefix}:
            model_class = DebertaV2Model
            tokenizer_class = DebertaV2Tokenizer
            patching_method = patch_deberta_causal
        else:
            patching_method = lambda x: x
            raise NotImplementedError
        self.encoder_token = model_class.from_pretrained(hf_model_name)
        self.encoder_chunk = patching_method(model_class.from_pretrained(hf_model_name))
        self.decoder = patching_method(model_class.from_pretrained(hf_model_name))
        self.tokenizer = tokenizer_class.from_pretrained(hf_model_name)

        encoder_chunk_emb = self.encoder_chunk.embeddings.word_embeddings.weight[self.tokenizer.bos_token_id]
        self.chunk_sos_embedding = nn.Parameter(encoder_chunk_emb)
        # We do not need the token embeddings for the chunk level
        self.encoder_chunk.embeddings.word_embeddings = nn.Identity()

    def forward(self, batch: BatchedTextTokens) -> Tensor:
        x = batch.batch
        batch_size, max_splits, max_tokens_per_split = x.shape
        x = x.reshape(batch_size * max_splits, max_tokens_per_split)
        att_mask = x != self.tokenizer.pad_token_id
        token_units = self.encoder_token(input_ids=x, attention_mask=att_mask).last_hidden_state
        hidden_size = token_units.shape[2]
        chunk_units = token_units.reshape(batch_size, max_splits, max_tokens_per_split, hidden_size)[:, :, 0, :]
        chunk_units = self.encoder_chunk(inputs_embeds=chunk_units, attention_mask=att_mask)
        chunk_sos_emb = self.chunk_sos_embedding.reshape(1, 1, -1) * torch.ones_like(chunk_units[:, :1])
        chunk_units = torch.cat([chunk_sos_emb, chunk_units], 1)
        # self.decoder(chunk_units)
