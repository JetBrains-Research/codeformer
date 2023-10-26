from itertools import product

import pytest
import torch
from transformers import AutoTokenizer

from lm.data_utils import (WikiText2Dataset, WikiText103Dataset,
                           WikiText2RawDataset, WikiText103RawDataset,
                           ThePileDataset, ThePileDataModule,
                           WikiTextDatasetBase)

TEST_TOKENIZER_NAME = 'microsoft/deberta-v3-base'

MAX_TEXT_TOKENS = 2048
MAX_CHUNK_SIZE = 14
MAX_CHUNKS_NUMBER = 384
MIN_CHUNKS = 1
MIN_TOKENS = 1
BATCH_SIZE = 2

WIKITEXT_DATASET_CLASSES =[WikiText2Dataset,
                           WikiText2RawDataset,
                           WikiText103Dataset,
                           WikiText103RawDataset]
WIKITEXT_SPLITS = ['train', 'validation', 'test']


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
    dm = ThePileDataModule(BATCH_SIZE, tokenizer, MAX_TEXT_TOKENS,
                           MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE, MIN_TOKENS,
                           MIN_CHUNKS, num_workers=0, prefetch_factor=None)
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))


def test_the_pile_data_module():
    tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_NAME)
    dm = ThePileDataModule(BATCH_SIZE, tokenizer, MAX_TEXT_TOKENS,
                           MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE, MIN_TOKENS,
                           MIN_CHUNKS, num_workers=0, prefetch_factor=None)
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))
    # TODO: add tensor checks
