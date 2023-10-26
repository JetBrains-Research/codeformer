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


def dump_wikitext_dataset(dump_dir: str | Path | None = None) -> None:
    if dump_dir is None:
        dump_dir = Path('./')
    max_text_tokens = 2048
    max_chunk_size = 14
    max_chunks_number = 384
    min_chunks = 1
    min_tokens = 1
    tokenizer = None
    ds_to_file_name = {
        WikiText2Dataset: 'wiki-text-2',
        WikiText2RawDataset: 'wiki-text-2-raw',
        WikiText103Dataset: 'wiki-text-103',
        WikiText103RawDataset: 'wiki-text-103-raw',
    }
    for dataset_class in WIKITEXT_DATASET_CLASSES:
        for split in ['train', 'validation', 'test']:
            ds = dataset_class(split, tokenizer, max_text_tokens,
                               max_chunks_number, max_chunk_size,
                               min_chunks, min_tokens)
            with open(dump_dir / f'{ds_to_file_name[dataset_class]}-{split}.jsonl', 'w') as fp:
                for sample in ds.ds:
                    fp.write(json.dumps({'text': sample}) + '\n')

