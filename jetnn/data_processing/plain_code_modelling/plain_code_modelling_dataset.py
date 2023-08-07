import json
import os.path
from os.path import exists
from typing import Optional, List
from string import punctuation, whitespace
import torch

from commode_utils.filesystem import get_lines_offsets, get_line_by_offset
from omegaconf import DictConfig
from torch.utils.data import Dataset

from jetnn.data_processing.vocabularies.plain.plain_code_vocabulary import (
    PlainCodeVocabulary,
)
from jetnn.data_processing.plain_code_modelling.labeled_plain_code_modelling import (
    LabeledCodeModellingTokens,
)
from jetnn.data_processing.tree_representation.my_code_tree import MyCodeTree
from jetnn.models.utils import (
    transform_sequence_according_to_split_with_begin_end_tokens,
    get_labels_for_code_modelling,
)


class PlainCodeModellingDataset(Dataset):
    _log_file = "bad_samples.log"

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

    def __getitem__(self, index):
        try:
            raw_sample = get_line_by_offset(self._data_file, self._line_offsets[index])
            sample = json.loads(raw_sample)
            cleaned_code = self._code_tree.remove_comments(sample["code"])
            tokenized_code = self.tokenize(cleaned_code, self._config.max_code_parts)

            return LabeledCodeModellingTokens(
                tokenized_label, transformed_tokenized_code, num_splits
            )
        except ValueError as e:
            with open(self._log_file, "a") as f_out:
                f_out.write(f"Error parsing sample from line #{index}: {e}\n")
            return None

    def tokenize(self, text, max_parts):
        tokenizer = self._vocab.tokenizer
        return tokenizer.encode(
            text,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_parts,
            truncation="longest_first",
        )
