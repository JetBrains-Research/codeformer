import json
from string import punctuation, whitespace

from omegaconf import DictConfig
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary


class HFTokenizerVocabulary(Vocabulary):
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        super().__init__(tokenizer=tokenizer)

    def __len__(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, token: str) -> list[int]:
        return self.tokenizer.encode(token)

    def decode(self, encoded) -> str:
        return self.tokenizer.decode(encoded, skip_special_tokens=True)

    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def bos_id(self) -> int:
        return self.tokenizer.bos_token_id

    def eos_id(self) -> int:
        return self.tokenizer.eos_token_id

    def unk_id(self) -> int:
        return self.tokenizer.unk_token_id


def from_holdout(file: str, config: DictConfig) -> HFTokenizerVocabulary:
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_tokenizer)
    print(config.checkpoint_tokenizer)
    if config.train_new_tokenizer:
        training_corpus = []
        with open(file, "r") as f:
            for line in f:
                code = json.loads(line)["code"]
                training_corpus.extend(code.split())
        return HFTokenizerVocabulary(
            tokenizer.train_new_from_iterator(
                training_corpus, config.max_tokenizer_vocab
            )
        )
    return HFTokenizerVocabulary(tokenizer)
