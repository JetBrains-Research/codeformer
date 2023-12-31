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
from jetnn.data_processing.plain_code_ast_method.labeled_plain_code_ast import (
    LabeledCodeAstTokens,
)
from jetnn.data_processing.tree_representation.my_code_tree import MyCodeTree
from jetnn.models.utils import (
    transform_sequence_according_to_split,
    transform_sequence_according_to_split_with_begin_end_tokens,
)


class PlainCodeAstDataset(Dataset):
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

        self._code_tree = MyCodeTree(
            self._config.programming_language, self._config.path_to_tree_sitter
        )

        open(self._log_file, "w").close()

    def __len__(self):
        return self._n_samples

    # TODO: remove string annotations for python and add context cutting
    def __getitem__(self, index) -> Optional[LabeledCodeAstTokens]:
        try:
            raw_sample = get_line_by_offset(self._data_file, self._line_offsets[index])
            sample = json.loads(raw_sample)
            label = sample["label"].replace(self._separator, " ")
            cleaned_code, _, method_location = self._code_tree.remove_comments(sample["code"], sample["method_location"])
            tokenized_label = self.tokenize(label, self._config.max_label_parts)
            tokenized_code = self.tokenize(cleaned_code, self._config.max_code_parts)
            tokenized_code = list(filter(lambda x: x != self._vocab.pad_id(), tokenized_code))[1:-1]
            tokens = [self._vocab.tokenizer.decode(token) for token in tokenized_code]
            tokens_split = self._code_tree.process_code(
                cleaned_code, tokens, self._config.max_subsequence_size
            )
            num_splits = min(self._config.max_subsequences_number, len(tokens_split))
            tokenized_code = (
                transform_sequence_according_to_split_with_begin_end_tokens(
                    torch.tensor(tokenized_code),
                    tokens_split,
                    num_splits,
                    self._config.max_subsequence_size,
                    self._vocab.bos_id(),
                    self._vocab.eos_id(),
                )
            )
            return LabeledCodeAstTokens(tokenized_label, tokenized_code, num_splits)
        except ValueError as e:
            with open(self._log_file, "a") as f_out:
                f_out.write(f"Error parsing sample from line #{index}: {e}\n")
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
