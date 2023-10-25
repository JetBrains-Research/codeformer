from transformers import AutoTokenizer

from lm.data_utils import (WikiText2Dataset, WikiText103Dataset,
                           WikiText2RawDataset, WikiText103RawDataset,
                           ThePileDataset, ThePileDataModule)

TEST_TOKENIZER_NAME = 'microsoft/deberta-v3-base'


def test_wikitext_datasets():
    ...


def test_the_pile_data_module():
    max_text_tokens = 2048
    max_chunk_size = 14
    max_chunks_number = 384
    min_chunks = 8
    min_tokens = 128
    batch_size = 2
    tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_NAME)
    dm = ThePileDataModule(batch_size, tokenizer, max_text_tokens,
                           max_chunks_number, max_chunk_size, min_tokens,
                           min_chunks, num_workers=0, prefetch_factor=None)
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))
    a = 'pass'
    # TODO: add tensor checks
