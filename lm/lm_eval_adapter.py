import torch
from lm.utils import get_model_from_config, get_tokenizer_from_config
from omegaconf import OmegaConf

from lm_eval import evaluator, tasks