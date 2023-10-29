import torch
from transformers import AutoTokenizer

from lm.model import CodeformerLM
from lm.data_utils import AllDatasetsDataModule

from consts import (MAX_TEXT_TOKENS, MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE,
                    MIN_TOKENS, MIN_CHUNKS, TEST_TOKENIZER_NAME, BATCH_SIZE,
                    DEFAULT_DATASET_NAME)


def test_codeformer():
    device = torch.device('cpu')
    model = CodeformerLM(TEST_TOKENIZER_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_NAME)
    dm = AllDatasetsDataModule(DEFAULT_DATASET_NAME, BATCH_SIZE, tokenizer, MAX_TEXT_TOKENS,
                               MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE, MIN_TOKENS,
                               MIN_CHUNKS, num_workers=0, prefetch_factor=None)
    dl = dm.test_dataloader()
    batch = next(iter(dl))
    batch = batch.to(device)
    outputs = model(batch)
