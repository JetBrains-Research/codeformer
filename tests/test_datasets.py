from itertools import product

import pytest
import torch
from transformers import AutoTokenizer

from consts import (WIKITEXT_SPLITS, MAX_TEXT_TOKENS,
                    MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE,
                    MIN_TOKENS, MIN_CHUNKS,
                    TEST_TOKENIZER_NAME, BATCH_SIZE)
from lm.data_utils import (AllDatasetsDataModule,
                           WikiTextDatasetBase)
from lm.utils import WIKITEXT_DATASET_CLASSES


@pytest.mark.parametrize('ds_class,split', product(WIKITEXT_DATASET_CLASSES, WIKITEXT_SPLITS))
def test_wikitext_datasets(ds_class: WikiTextDatasetBase, split: str):
    tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_NAME)
    ds = ds_class(split, tokenizer, MAX_TEXT_TOKENS, MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE, MIN_CHUNKS, MIN_TOKENS)
    for sample in ds:
        assert torch.tensor(sample.token_ids, dtype=torch.long).ndim == 1
        assert torch.tensor(sample.split_sizes, dtype=torch.long).ndim == 1
        assert len(sample.token_ids) > 0
        assert len(sample.split_sizes) > 0


def test_wikitext_data_module():
    tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_NAME)
    dm = AllDatasetsDataModule(BATCH_SIZE, tokenizer, MAX_TEXT_TOKENS,
                               MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE, MIN_TOKENS,
                               MIN_CHUNKS, num_workers=0, prefetch_factor=None)
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))


def test_the_pile_data_module():
    tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_NAME)
    dm = AllDatasetsDataModule('the_pile', BATCH_SIZE, tokenizer, MAX_TEXT_TOKENS,
                               MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE, MIN_TOKENS,
                               MIN_CHUNKS, num_workers=0, prefetch_factor=None)
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))
    # TODO: add tensor checks
