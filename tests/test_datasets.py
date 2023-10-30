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
