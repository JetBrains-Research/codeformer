import torch
from transformers import AutoTokenizer
import pytest

from lm.model import CodeformerLM
from lm.data_utils import AllDatasetsDataModule

from consts import (MAX_TEXT_TOKENS, MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE,
                    MIN_TOKENS, MIN_CHUNKS, BATCH_SIZE,
                    DEFAULT_DATASET_NAME, DEFAULT_BASE_MODEL_NAME)


@pytest.mark.parametrize('rnd_init', [True, False])
def test_codeformer(rnd_init):
    device = torch.device('cpu')
    model = CodeformerLM(DEFAULT_BASE_MODEL_NAME, rnd_init).to(device)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL_NAME)
    dm = AllDatasetsDataModule(DEFAULT_DATASET_NAME, BATCH_SIZE, tokenizer, MAX_TEXT_TOKENS,
                               MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE, MIN_TOKENS,
                               MIN_CHUNKS, num_workers=0, prefetch_factor=None)
    dl = dm.test_dataloader()
    batch = next(iter(dl))
    batch = batch.to(device)
    outputs = model(batch)
