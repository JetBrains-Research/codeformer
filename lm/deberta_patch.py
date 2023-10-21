from types import MethodType

import torch
from transformers import DebertaV2Model


def patch_deberta_causal(model: DebertaV2Model) -> DebertaV2Model:
    if getattr(model.encoder, '_get_attention_mask_non_casual', None) is not None:
        raise RuntimeError('Trying to make model casual second time!')
    model.encoder._get_attention_mask_non_casual = model.encoder.get_attention_mask

    def get_attention_mask(self, attention_mask):
        attention_mask = self._get_attention_mask_non_casual(attention_mask)
        seq_len = attention_mask.shape[-1]
        causal_mask = torch.ones(seq_len, seq_len, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = attention_mask * torch.tril(causal_mask)
        return attention_mask

    model.encoder.get_attention_mask = MethodType(get_attention_mask, model.encoder)
    return model
