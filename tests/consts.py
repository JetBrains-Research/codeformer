from lm.data_utils import (WikiText2Dataset, WikiText103Dataset,
                           WikiText2RawDataset, WikiText103RawDataset)

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
