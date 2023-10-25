import torch
from torch import Tensor, LongTensor
import wandb


def setup_wandb(args=None):
    mode = 'disabled' if args['dbg'] else None
    run = wandb.init(project="codeformer", mode=mode)
    wandb.run.name = args['name']
    wandb.config.update(args)
    return run


def perplexity(logits: Tensor, targets: LongTensor, pad_id: int) -> Tensor:
    # logits and targets must be compatible with torch.nn.functional.cross_entropy
    loss_tens = torch.nn.functional.cross_entropy(logits, targets, reduction='none', ignore_index=pad_id)
    loss = loss_tens.sum() / torch.count_nonzero(loss_tens)
    ppl = loss.exp()
    return ppl
