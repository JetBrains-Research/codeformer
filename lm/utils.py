from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import Tensor, LongTensor
import wandb

from lm.data_utils import (WikiText2RawDataset, WikiText2Dataset,
                           WikiText103Dataset, WikiText103RawDataset)

WIKITEXT_DATASET_CLASSES =[WikiText2Dataset, WikiText2RawDataset,
                           WikiText103Dataset, WikiText103RawDataset]


def setup_wandb(args=None):
    mode = 'disabled' if args['dbg'] else None
    run = wandb.init(project="codeformer", mode=mode)
    wandb.define_metric("epoch")
    wandb.define_metric("val_loss", step_metric="epoch", summary='min')
    wandb.define_metric("val_ppl", step_metric="epoch", summary='min')
    wandb.run.name = args['name']
    # Deep copy is really needed here, hydra says
    wandb.config.update(dict(deepcopy(args)))
    return run


def perplexity(logits: Tensor, targets: LongTensor, pad_id: int) -> Tensor:
    # logits and targets must be compatible with torch.nn.functional.cross_entropy
    loss_tens = torch.nn.functional.cross_entropy(logits, targets, reduction='none', ignore_index=pad_id)
    loss = loss_tens.sum() / torch.count_nonzero(loss_tens)
    ppl = loss.exp()
    return ppl
