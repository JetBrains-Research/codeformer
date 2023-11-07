import torch
from lm.utils import get_model_from_config, get_tokenizer_from_config
from omegaconf import OmegaConf

from lm_eval.models.gpt2 import HFLM


class EvalHarnessAdapter(HFLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self._pretrained = "debertav2"
        self.tokenizer = tokenizer
        self.VOCAB_SIZE = tokenizer.vocab_size
        self.EOT_TOKEN_ID = tokenizer.eos_token_id
        self.batch_size_per_gpu = 1
        self._max_length = self.tokenizer.model_max_length
        self._device = "cpu"


