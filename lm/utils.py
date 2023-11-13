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
    if config.tokenizer_name is not None:
        name = config.tokenizer_name
    else:
        if config.model_name in {'codeformer', 'deberta_causal'}:
            name = config.base_model_name
        else:
            name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(name)
    if name == 'gpt2':
        special_tokens_dict = {
            'bos_token': '<|beginingoftext|>',
            'pad_token': '<|padtoken|>',
            'unk_token': '<|unktoken|>',
        }
        tokenizer.add_special_tokens(special_tokens_dict)
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
        'input_ids': batch.token_ids_bos_eos,
        'attention_mask': batch.att_mask_bos_eos,
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
    targets = batch.token_ids_bos_eos[:, 1:]
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

def is_deberta_v2_v3(model_name: str) -> bool:
    deberta_v2_prefix = 'microsoft/deberta-v2'
    deberta_v3_prefix = 'microsoft/deberta-v3'
    return model_name[:len(deberta_v3_prefix)] in {deberta_v2_prefix, deberta_v3_prefix}


def get_model_from_config(config: str | Path | OmegaConf,
                          tokenizer: Tokenizer) -> nn.Module:
    # config: either a path to a YAML file or an OmegaConf constructed
    # from the YAML
    num_tokens = len(tokenizer.vocab)
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    if config.model_name == 'codeformer':
        from lm.model import CodeformerLM
        original_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        num_tokens_original = len(original_tokenizer.vocab)
        model = CodeformerLM(config.base_model_name, do_random_init=config.random_init)
        if num_tokens_original != num_tokens:
            model.encoder_token.resize_token_embeddings(num_tokens)
            model.decoder.resize_token_embeddings(num_tokens)
            print(f'Resized model embeddings to: {num_tokens}')
    else:
        original_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        num_tokens_original = len(original_tokenizer.vocab)
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
        if num_tokens_original != num_tokens:
            model.resize_token_embeddings(num_tokens)
            print(f'Resized model embeddings to: {num_tokens}')
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


def disassemble(inputs: Tensor,
                sizes: LongTensor,
                fill_empty_val: Tensor | int | float,
                bos_value: None | Tensor | int = None,
                eos_value: None | Tensor | int = None) -> Tensor:
    # inputs.shape = [batch_size, seq_len, *other_shapes]
    # sizes.shape = [batch_size, max_chunks_per_sample]
    # fill_empty_val.shape = [*other_shapes] where other shapes can be empty
    #   so that the inputs are 2D, OR scalar
    # bos_value: if not None, every element (along 0 dimension) will be prepended with it
    # eos_value: if not None, every element (along 0 dimension) will be prepended with it

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
    num_chunks_per_sample = sizes.count_nonzero(dim=1)  # E: [2, 1]

    num_chunks_range = torch.arange(1, num_chunks_total + 1, device=device).view(num_chunks_total, 1)
    # E: [[1], [2], [3]]

    num_chunks_cumsum = num_chunks_per_sample.cumsum(0).view(1, batch_size)  # E: [[2, 3]]
    # E: num_chunks_range > num_chunkscum_sum =
    # [[0, 0]
    #  [0, 0],
    #  [1, 0]]
    sample_nums = torch.sum(num_chunks_range > num_chunks_cumsum, 1)  # E: [0, 0, 1]

    sample_nums = sample_nums.view(num_chunks_total, 1).repeat(1, max_chunk_length)
    # E: [[0, 0, 0], [0, 0, 0], [1, 1, 1]]

    sample_nums = sample_nums.view(num_chunks_total * max_chunk_length)  # E: [0, 0, 0, 0, 0, 0, 1, 1, 1]

    non_zero_sizes_mask = sizes != 0  # E: [[1, 1], [0, 0]]
    batch_size_zeros = torch.zeros(batch_size, 1, dtype=sizes.dtype, device=sizes.device)
    # E: torch.cat([batch_size_zeros, sizes[:, :-1]], 1).cumsum(1) =
    # [[0, 2]
    #  [0, 3]

    intra_sample_shifts = torch.cat([batch_size_zeros, sizes[:, :-1]], 1).cumsum(1)[non_zero_sizes_mask]
    # E: [0, 2, 0]

    max_chunk_range = torch.arange(1, max_chunk_length + 1).view(1, max_chunk_length)
    chunk_mask = max_chunk_range <= sizes[non_zero_sizes_mask].view(num_chunks_total, 1)
    plain_indices = torch.arange(1, max_chunk_length + 1).view(1, max_chunk_length).repeat(num_chunks_total, 1)
    # E: plain_indices =
    # [[1, 2, 3],
    #  [1, 2, 3],
    #  [1, 2, 3]]
    indices = (plain_indices + intra_sample_shifts.view(num_chunks_total, 1)) * chunk_mask
    # E: [[1, 2, 0], [3, 0, 0], [1, 2, 3]]

    indices = indices.view(num_chunks_total * max_chunk_length)  # E: [1, 2, 0, 3, 0, 0, 1, 2, 3]

    filler = expand_filler(fill_empty_val, inputs.shape).to(inputs.device).to(inputs.dtype)

    inputs_with_filler_first = torch.cat([filler, inputs], dim=1)  # E: [[0, 1, 2, 3], [0, 4, 5, 6]]
    # E: return [[1, 2, 0], [3, 0, 0], [4, 5, 6]]
    outputs = inputs_with_filler_first[sample_nums, indices].view(num_chunks_total, max_chunk_length, *inputs.shape[2:])
    if eos_value is not None:
        outputs = torch.cat([outputs, torch.full([num_chunks_total, 1], fill_empty_val)], 1)
        sizes_flat = sizes[non_zero_sizes_mask]
        outputs[torch.arange(num_chunks_total), sizes_flat] = eos_value
    if bos_value is not None:
        outputs = torch.cat([torch.full([num_chunks_total, 1], bos_value), outputs], 1)

    return outputs


def assemble(inputs: Tensor, sizes: Tensor, fill_empty_val: int | float | Tensor) -> Tensor:
    # inputs.shape = [num_chunks, max_chunk_len, *other_shapes]
    # sizes.shape = [batch_size, seq_len]
    # fill_empty_val.shape = [*other_shapes] where other shapes can be empty
    #   so that the inputs are 2D, OR scalar

    # Example (intermediate values for this example will be shown further with "E:" signature):
    #
    # inputs:
    # [[1, 2, 0],
    #  [3, 0, 0],
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
    # [[[1., 2., 0.],
    #   [3., 0., 0.]],
    #  [[4., 5., 6.],
    #   [0., 0., 0.]]]

    max_chunk_len = inputs.shape[1]
    batch_size, max_chunks = sizes.shape
    ext_inputs = torch.cat([torch.zeros(1, max_chunk_len), inputs], 0)
    # E: [[0, 0, 0], [1, 2, 0], [3, 0, 0], [4, 5, 6]]

    chunk_ids = torch.arange(max_chunks).view(1, batch_size).repeat(batch_size, 1) + 1  # E: [[1, 2], [1, 2]]
    non_zero_sizes_cumsum = (sizes > 0).float().sum(1, keepdim=True)[:-1].cumsum(0).long()  # E: [2]
    shifts = torch.cat([torch.zeros(1, 1, device=sizes.device, dtype=torch.long),
                        non_zero_sizes_cumsum], 0)  # E: [[0], [2]]
    mask = (sizes > 0).float()  # E: [[1, 1], [1, 0]]
    chunk_flat_ids = (chunk_ids + shifts) * mask  # E: [[1, 2], [3, 0]]
    chunk_flat_ids = chunk_flat_ids.view(batch_size * max_chunks).long()  # E: [1, 2, 3, 0]
    # E: return = [[[1., 2., 0.],
    #               [3., 0., 0.]],
    #              [[4., 5., 6.],
    #               [0., 0., 0.]]]
    return ext_inputs[chunk_flat_ids.view(batch_size, max_chunks)]


def assemble_decoder_inputs(source: Tensor,
                            sizes: LongTensor,
                            from_starting_points: LongTensor,
                            to_starting_points: LongTensor,
                            max_len: int) -> Tensor:
    device = source.device
    input_hidden_sizes = source.shape[2:]
    mask = sizes > 0
    batch_size, chunks_per_sample = source.shape[:2]
    num_chunks = sizes.count_nonzero()
    max_size = sizes.max()

    from_starting_points_flat = from_starting_points[mask].view(num_chunks, 1)
    d1_from_indices = torch.arange(max_size).view(1, max_size).repeat(num_chunks, 1) + from_starting_points_flat
    sizes_flat = sizes[mask].view(num_chunks, 1)
    mask_by_size = d1_from_indices < sizes_flat + from_starting_points_flat

    d1_from_indices_flat = d1_from_indices[mask_by_size]

    d0_from_indices = torch.arange(batch_size).view(batch_size, 1).repeat(1, chunks_per_sample)[mask]
    d0_from_indices_flat = d0_from_indices.view(num_chunks, 1).repeat(1, max_size)[mask_by_size]

    d0_to_indices = torch.arange(num_chunks).view(num_chunks, 1).repeat(1, max_size)[mask_by_size]
    d1_to_indices = torch.arange(max_size).view(1, max_size).repeat(num_chunks, 1)[mask_by_size]
    d1_to_indices = d1_to_indices + to_starting_points[mask].view(num_chunks, 1).repeat(1, max_size)[mask_by_size]

    output = torch.zeros(num_chunks, max_len, *input_hidden_sizes, dtype=source.dtype, device=device)

    output[d0_to_indices, d1_to_indices] = source[d0_from_indices_flat, d1_from_indices_flat]

    return output


def prepare_token_ids_for_decoder(token_ids: LongTensor,
                                  sizes: LongTensor,
                                  pad_id: int | LongTensor,
                                  bos_id: int | LongTensor,
                                  eos_id: int | LongTensor) -> tuple[Tensor, LongTensor]:
    dtype = token_ids.dtype
    device = token_ids.device
    batch_size, seq_len = token_ids.shape
    max_chunks_per_sample = sizes.shape[1]
    cur_plus_prev_sizes = sizes.detach().clone()
    cur_plus_prev_sizes[:, 1:] = cur_plus_prev_sizes[:, 1:] + cur_plus_prev_sizes[:, :-1]
    mask = sizes > 0
    cur_plus_prev_sizes = cur_plus_prev_sizes * mask.long()
    cur_plus_prev_sizes_flat = cur_plus_prev_sizes[cur_plus_prev_sizes > 0]
    num_chunks_total = mask.count_nonzero()

    from_starting_points = torch.concat([torch.zeros(batch_size, 2, device=device, dtype=torch.long),
                                         sizes[:, :-2].cumsum(1)],
                                        1)[:, :max_chunks_per_sample][mask]
    max_len = cur_plus_prev_sizes.max()
    # outputs = assemble_decoder_inputs(token_ids, cur_plus_prev_sizes,
    #                                   from_starting_points, to_starting_points, max_len)
    outputs = torch.zeros(num_chunks_total, max_len, dtype=dtype, device=device)

    d1_from_ids = torch.arange(seq_len, device=device).view(1, seq_len).repeat(num_chunks_total, 1)
    mask_chunk = d1_from_ids < cur_plus_prev_sizes_flat.view(num_chunks_total, 1)
    d1_from_ids = d1_from_ids[mask_chunk]
    d1_to_ids = d1_from_ids
    d1_from_ids = d1_from_ids + from_starting_points.view(num_chunks_total, 1).repeat(1, max_len)[mask_chunk]
    d0_from_ids = torch.arange(batch_size, device=device).view(batch_size, 1).repeat(1, max_chunks_per_sample)[mask]
    d0_from_ids = d0_from_ids.view(num_chunks_total, 1).repeat(1, max_len)[mask_chunk]

    # d1_to_ids = torch.arange(max_len, device=device).view(1, max_len).repeat(num_chunks_total, 1)
    # d1_to_ids = d1_from_ids[mask_chunk]
    d0_to_ids = torch.arange(num_chunks_total, device=device).view(num_chunks_total, 1).repeat(1, max_len)
    d0_to_ids = d0_to_ids[mask_chunk]
    outputs[d0_to_ids, d1_to_ids] = token_ids[d0_from_ids, d1_from_ids]

    # add eos
    outputs = torch.cat([outputs, torch.full([num_chunks_total, 1], pad_id)], 1)
    outputs[torch.arange(num_chunks_total), cur_plus_prev_sizes_flat] = eos_id

    # add bos
    outputs = torch.cat([torch.full([num_chunks_total, 1], bos_id), outputs], 1)

    # output = disassemble(token_ids, cur_plus_prev_sizes, pad_id, bos_id, eos_id)
    lens: LongTensor = (cur_plus_prev_sizes + 2) * mask
    return outputs, lens[lens > 0]


def put_token_embeddings_at_specified_positions(token_embs_by_chunk_flat: Tensor,
                                                prev_and_curr_chunk_lens_plus_bos_eos: LongTensor,
                                                start_positions: LongTensor,
                                                max_len: int) -> Tensor:
    lens = prev_and_curr_chunk_lens_plus_bos_eos
    rest_shape = token_embs_by_chunk_flat.shape[2:]
    chunk_max_len = lens.max()
    num_chunks = len(lens)
    device = token_embs_by_chunk_flat.device
    dtype = token_embs_by_chunk_flat.dtype
    total_chunks = token_embs_by_chunk_flat.shape[0]
    outputs = torch.zeros(total_chunks, max_len, *rest_shape, device=device, dtype=dtype)
    d0_ids = torch.arange(num_chunks).view(num_chunks, 1).repeat(1, chunk_max_len)
    mask = d0_ids < lens.view(num_chunks, 1)
    d0_ids = d0_ids[mask]
    d1_ids = torch.arange(chunk_max_len).view(1, chunk_max_len).repeat(num_chunks, 1)
    d1_ids = d1_ids[mask]

    to_d0_ids = d0_ids
    shifts = start_positions.view(num_chunks, 1).repeat(1, chunk_max_len)[mask]
    to_d1_ids = d1_ids + shifts
    outputs[to_d0_ids, to_d1_ids] = token_embs_by_chunk_flat[d0_ids, d1_ids]
    return outputs
