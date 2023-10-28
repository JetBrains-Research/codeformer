from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf
from torch import Tensor
import wandb
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutput

from lm.data_utils import (WikiText2RawDataset, WikiText2Dataset,
                           WikiText103Dataset, WikiText103RawDataset,
                           BatchedTextTokens)
from lm.eval_utils import metrics
from lm.model import CodeformerLM

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
            
            # noinspection PyTypeChecker
            ds = dataset_class(split, tokenizer, max_text_tokens,
                               max_chunks_number, max_chunk_size,
                               min_chunks, min_tokens)

            with open(dump_dir / f'{ds_to_file_name[dataset_class]}-{split}.jsonl', 'w') as fp:
                for sample in ds.ds:
                    fp.write(json.dumps({'text': sample}) + '\n')


def get_tokenizer_from_config(config: str | Path | OmegaConf) -> nn.Module:
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    if config.model_name == 'codeformer':
        name = config.base_model_name
    else:
        name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(name)
    if name == 'gpt2':
        print('Warning! Populating pad token!')
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def get_train_batch_preprocessor(config: str | Path | OmegaConf) -> callable:
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    if config.model_name == 'codeformer':
        return lambda batch, device: {'batch': batch.to(device)}
    else:
        return train_batch_preprocessor_hf


def train_batch_preprocessor_hf(batch: BatchedTextTokens, device: torch.DeviceObjType) -> dict[str, Tensor]:
    input_dict = {
        'input_ids': batch.token_ids,
        'attention_mask': batch.att_mask,
    }
    return dict_to_device(input_dict, device)


def get_model_output_postprocessor(config: str | Path | OmegaConf) -> callable:
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    if config.model_name == 'codeformer':
        return lambda batch, outputs: outputs
    else:
        return model_output_postprocessor_hf


def model_output_postprocessor_hf(batch: BatchedTextTokens, outputs: BaseModelOutput) -> dict[str, Tensor]:
    targets = batch.token_ids[:, 1:]
    logits = outputs.logits[:, :-1].reshape(-1, outputs.logits.shape[2])
    return metrics(logits, targets, batch.pad_token_id)


def dict_to_device(input_dict: dict[str, Tensor | int | float],
                   device: torch.DeviceObjType) -> dict[str, Tensor | int | float]:
    on_device_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, Tensor):
            on_device_dict[k] = v.to(device)
        else:
            on_device_dict[k] = v

    return on_device_dict


def get_model_from_config(config: str | Path | OmegaConf) -> nn.Module:
    # config: either a path to a YAML file or an OmegaConf constructed
    # from the YAML
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    if config.model_name == 'codeformer':
        from lm.model import CodeformerLM
        model = CodeformerLM(config.base_model_name, do_random_init=config.random_init)
    else:
        if config.random_init:
            model_cfg = AutoConfig.from_pretrained(config.model_name)
            model = AutoModelForCausalLM.from_config(model_cfg)
        else:
            model = AutoModelForCausalLM.from_pretrained(config.model_name)
    if config.load_path is not None:
        print(f'Loading model from: {config.load_path}')
        model.load_state_dict(torch.load(config.load_path, map_location='cpu'))
    return model


def get_model_module(model: PreTrainedModel) -> PreTrainedModel:
    children = list(model.children())
    is_pretrained_model = [isinstance(child, PreTrainedModel) for child in children]
    if any(is_pretrained_model):
        assert sum(is_pretrained_model) == 1
        return next(child for child, is_pret in zip(children, is_pretrained_model) if is_pret)
    else:
        return model


