import json
import os.path
from os.path import exists
from typing import Optional, List
from string import punctuation, whitespace

from commode_utils.filesystem import get_lines_offsets, get_line_by_offset
from omegaconf import DictConfig
from torch.utils.data import Dataset

from jetnn.data_processing.vocabularies.plain.plain_code_vocabulary import (
    PlainCodeVocabulary,
)
from jetnn.data_processing.plain_code_method.labeled_plain_code import LabeledCodeTokens
from jetnn.data_processing.tree_code_representation.my_code_tree import MyCodeTree


class PlainCodeDataset(Dataset):
    _log_file = "bad_samples.log"
    _separator = "|"

    def __init__(
        self, data_file: str, config: DictConfig, vocabulary: PlainCodeVocabulary
    ):
        if not exists(data_file):
            raise ValueError(f"Can't find file with data: {data_file}")
        self._data_file = data_file
        self._config = config.data
        self._vocab = vocabulary

        self._line_offsets = get_lines_offsets(data_file)
        self._n_samples = len(self._line_offsets)
        
        self._code_tree = MyCodeTree()

        open(self._log_file, "w").close()

    def __len__(self):
        return self._n_samples

    def __getitem__(self, index) -> Optional[LabeledCodeTokens]:
        try:
            raw_sample = get_line_by_offset(self._data_file, self._line_offsets[index])
            sample = json.loads(raw_sample)
            label = sample["label"].replace(self._separator, " ")
            cleaned_code = self._code_tree.remove_comments(sample["code"])
            code = "".join(
                [
                    (ch if ch not in (punctuation + whitespace) else " ")
                    for ch in cleaned_code
                ]
            )
            code = " ".join(code.split())
            return LabeledCodeTokens(
                self.tokenize(label, self._config.max_label_parts),
                self.tokenize(code, self._config.max_code_parts),
            )
        except ValueError as e:
            with open(self._log_file, "a") as f_out:
                f_out.write(f"Error parsing sample from line #{index}: {e}")
            return None

    def tokenize(self, text: str, max_parts: int) -> List[int]:
        tokenizer = self._vocab.tokenizer
        return tokenizer.encode(
            text,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_parts,
            truncation="longest_first",
        )
