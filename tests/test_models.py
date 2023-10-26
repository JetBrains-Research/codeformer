import torch
from transformers import AutoTokenizer

from lm.model import CodeformerLM
from lm.data_utils import WikiText2RawDataModule

from consts import (WIKITEXT_DATASET_CLASSES, WIKITEXT_SPLITS, MAX_TEXT_TOKENS,
                    MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE, MIN_TOKENS, MIN_CHUNKS,
                    TEST_TOKENIZER_NAME, BATCH_SIZE)


def test_codeformer():
    device = torch.device('cpu')
    split = 'test'
    model = CodeformerLM(TEST_TOKENIZER_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_NAME)
    dm = WikiText2RawDataModule(BATCH_SIZE, tokenizer, MAX_TEXT_TOKENS,
                                MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE, MIN_TOKENS,
                                MIN_CHUNKS, num_workers=0, prefetch_factor=None)
    dl = dm.test_dataloader()
    batch = next(iter(dl))
    batch = batch.to(device)
    outputs = model(batch)
