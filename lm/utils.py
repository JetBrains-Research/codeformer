from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn, Tensor, LongTensor
from omegaconf import OmegaConf
import wandb
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from lm.data_utils import (WikiText2RawDataset, WikiText2Dataset,
                           WikiText103Dataset, WikiText103RawDataset,
                           BatchedTextTokens)
from lm.eval_utils import metrics

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


def get_tokenizer_from_config(config: str | Path | OmegaConf) -> Tokenizer:
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    if config.model_name in {'codeformer', 'deberta_causal'}:
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
        if config.model_name == 'deberta_causal':
            from lm.model import PatchedDebertaAsCausalLM
            model = PatchedDebertaAsCausalLM.from_pretrained(config.base_model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(config.model_name)
            if config.load_path is not None:
                print(f'Loading model from: {config.load_path}')
                model.load_state_dict(torch.load(config.load_path, map_location='cpu'))
        for module in model.modules():
            model._init_weights(module)
    return model


def get_model_module(model: PreTrainedModel) -> PreTrainedModel:
    children = list(model.children())
    is_pretrained_model = [isinstance(child, PreTrainedModel) for child in children]
    if any(is_pretrained_model):
        assert sum(is_pretrained_model) == 1
        return next(child for child, is_pret in zip(children, is_pretrained_model) if is_pret)
    else:
        return model


def check_tokenizer_pad_id(tokenizer: Tokenizer) -> None:
    assert (tokenizer.bos_token_id is not None) and (tokenizer.bos_token_id != tokenizer.pad_token_id)
    assert (tokenizer.eos_token_id is not None) and (tokenizer.eos_token_id != tokenizer.pad_token_id)
    # assert (tokenizer.sos_token_id != tokenizer.bos_token_id)  # it seems to be ok
    assert tokenizer.pad_token_id is not None


def expand_filler(fill_empty_val: int | float | Tensor, inputs_shape: torch.Size) -> Tensor:
    is_scalar_tensor = isinstance(fill_empty_val, Tensor) and fill_empty_val.numel() == 1
    # Expand filler if needed
    if isinstance(fill_empty_val, int) or is_scalar_tensor:
        filler = torch.full([inputs_shape[0]] + [1] + list(inputs_shape)[2:],
                            fill_empty_val)
        # print(filler.shape)
    else:
        filler = fill_empty_val.view(1, 1, *fill_empty_val.shape).repeat(inputs_shape[0], 1, *([1] * len(fill_empty_val.shape)))
    return filler


def disassemble(inputs: Tensor, sizes: LongTensor, fill_empty_val: Tensor | int | float) -> Tensor:
    # inputs.shape = [batch_size, seq_len, *other_shapes]
    # sizes.shape = [batch_size, max_chunks_per_sample]
    # fill_empty_val.shape = [*other_shapes] where other shapes can be empty
    #   so that the inputs are 2D, OR scalar

    # Example (intermediate values for this example will be shown further with "E:" signature):
    #
    # inputs:
    # [[1, 2, 3],
    #  [4, 5, 6]]
    #
    # sizes:
    # [[2, 1],
    #  [3, 0]]
    #
    # fill_empty_val:
    # 0
    #
    # Expected output:
    # [[1, 2, 0],
    #  [3, 0, 0],
    #  [4, 5, 6]]

    device = inputs.device
    batch_size = inputs.shape[0]
    num_chunks_total = sizes.count_nonzero()  # E: 3
    max_chunk_length = sizes.max()  # E: 3
    num_chunks_per_sample = sizes.count_nonzero(dim=0)  # E: [2, 1]

    num_chunks_range = torch.arange(1, num_chunks_total + 1, device=device).view(num_chunks_total, 1)  # E: [[1], [2], [3]]
    num_chunkscum_sum = num_chunks_per_sample.cumsum(0).view(1, batch_size)  # E: [[2, 3]]
    # E: num_chunks_range > num_chunkscum_sum =
    # [[0, 0]
    #  [0, 0],
    #  [1, 0]]
    sample_nums = torch.sum(num_chunks_range > num_chunkscum_sum, 1)  # E: [0, 0, 1]

    sample_nums = sample_nums.view(num_chunks_total, 1).repeat(1, max_chunk_length)  # E: [[0, 0, 0], [0, 0, 0], [1, 1, 1]]
    sample_nums = sample_nums.view(num_chunks_total * max_chunk_length)  # E: [0, 0, 0, 0, 0, 0, 1, 1, 1]

    non_zero_sizes_mask = sizes != 0  # E: [[1, 1], [0, 0]]
    batch_size_zeros = torch.zeros(batch_size, 1, dtype=sizes.dtype, device=sizes.device)
    # E: torch.cat([batch_size_zeros, sizes[:, :-1]], 1).cumsum(1) =
    # [[0, 2]
    #  [0, 3]
    intra_sample_shifts = torch.cat([batch_size_zeros, sizes[:, :-1]], 1).cumsum(1)[non_zero_sizes_mask]  # E: [0, 2, 0]

    max_chunk_range = torch.arange(1, max_chunk_length + 1).view(1, max_chunk_length)
    chunk_mask = max_chunk_range <= sizes[non_zero_sizes_mask].view(num_chunks_total, 1)
    plain_indices = torch.arange(1, max_chunk_length + 1).view(1, max_chunk_length).repeat(num_chunks_total, 1)
    print(f'plain_indices: {plain_indices}')
    # E: plain_indices =
    # [[1, 2, 3],
    #  [1, 2, 3],
    #  [1, 2, 3]]
    indices = (plain_indices + intra_sample_shifts.view(num_chunks_total, 1)) * chunk_mask  # E: [[1, 2, 0], [3, 0, 0], [1, 2, 3]]
    indices = indices.view(num_chunks_total * max_chunk_length)  # E: [1, 2, 0, 3, 0, 0, 1, 2, 3]

    filler = expand_filler(fill_empty_val, inputs.shape).to(inputs.device).to(inputs.dtype)

    inputs_with_filler_first = torch.cat([filler, inputs], dim=1)  # E: [[0, 1, 2, 3], [0, 4, 5, 6]]
    # E: return [[1, 2, 0], [3, 0, 0], [4, 5, 6]]
    return inputs_with_filler_first[sample_nums, indices].view(num_chunks_total, max_chunk_length, *inputs.shape[2:])


