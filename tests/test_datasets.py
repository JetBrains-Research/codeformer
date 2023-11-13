from itertools import product

import pytest
import torch
from transformers import AutoTokenizer

from consts import (WIKITEXT_SPLITS, MAX_TEXT_TOKENS,
                    MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE,
                    MIN_TOKENS, MIN_CHUNKS,
                    TEST_TOKENIZER_NAME, BATCH_SIZE,
                    DEFAULT_DATASET_NAME)
from lm.data_utils import (AllDatasetsDataModule,
                           WikiTextDatasetBase)
from lm.utils import WIKITEXT_DATASET_CLASSES


@pytest.mark.parametrize('ds_class,split', product(WIKITEXT_DATASET_CLASSES, WIKITEXT_SPLITS))
def test_wikitext_datasets(ds_class: WikiTextDatasetBase, split: str):
    max_samples_to_test = 1000
    tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_NAME)
    ds = ds_class(split, tokenizer, MAX_TEXT_TOKENS, MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE, MIN_CHUNKS, MIN_TOKENS)
    count = 0
    for sample in ds:
        assert torch.tensor(sample.token_ids, dtype=torch.long).ndim == 1
        assert torch.tensor(sample.split_sizes, dtype=torch.long).ndim == 1
        assert len(sample.token_ids) > 0
        assert len(sample.split_sizes) > 0
        count += 1
        if count > max_samples_to_test:
            break


@pytest.mark.parametrize('dataset_name', AllDatasetsDataModule.name_to_dataset.keys())
def test_wikitext_data_module(dataset_name):
    tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_NAME)
    dm = AllDatasetsDataModule(dataset_name, BATCH_SIZE, tokenizer, MAX_TEXT_TOKENS,
                               MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE, MIN_TOKENS,
                               MIN_CHUNKS, num_workers=0, prefetch_factor=None)
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))


def test_batch_class():
    sample_0 = TextTokens([1, 2, 3, 4, 5], [2, 2, 1])
    sample_1 = TextTokens([6, 7, 8], [1, 2])
    samples = [sample_0, sample_1]
    num_previous_chunks = 1
    pad_token_id = 0
    bos_token_id = -1
    eos_token_id = -2
    batch = BatchedTextTokens(samples, pad_token_id, bos_token_id, eos_token_id, num_previous_chunks)
    print(batch.decoder_inp_tok_ids)
    assert torch.all(batch.decoder_inp_tok_ids == torch.tensor([
        [-1, 1, 2, 0, 0],
        [-1, 1, 2, 3, 4],
        [-1, 3, 4, 5, 0],
        [-1, 6, 0, 0, 0],
        [-1, 6, 7, 8, 0]
    ]))
    assert torch.all(batch.decoder_targ_tok_ids == torch.tensor([
        [1, 2, -2, 0, 0],
        [0, 0, 3, 4, -2],
        [0, 0, 5, -2, 0],
        [6, -2, 0, 0, 0],
        [0, 7, 8, -2, 0]
    ]))
