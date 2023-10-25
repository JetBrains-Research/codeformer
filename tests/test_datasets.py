from transformers import AutoTokenizer

from lm.data_utils import (WikiText2Dataset, WikiText103Dataset,
                           WikiText2RawDataset, WikiText103RawDataset,
                           ThePileDataset, ThePileDataModule)

TEST_TOKENIZER_NAME = 'microsoft/deberta-v3-base'

MAX_TEXT_TOKENS = 2048
MAX_CHUNK_SIZE = 14
MAX_CHUNKS_NUMBER = 384
MIN_CHUNKS = 8
MIN_TOKENS = 128
BATCH_SIZE = 2


def test_wikitext_datasets():
    ...


def test_the_pile_data_module():
    tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_NAME)
    dm = ThePileDataModule(BATCH_SIZE, tokenizer, MAX_TEXT_TOKENS,
                           MAX_CHUNKS_NUMBER, MAX_CHUNK_SIZE, MIN_TOKENS,
                           MIN_CHUNKS, num_workers=0, prefetch_factor=None)
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))
    # TODO: add tensor checks
