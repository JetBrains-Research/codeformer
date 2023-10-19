import json
from os.path import exists
from typing import Optional, List

from commode_utils.filesystem import get_lines_offsets, get_line_by_offset
from omegaconf import DictConfig
from torch.utils.data import Dataset
import torch
from jetnn.data_processing.vocabularies.vocabulary import (
    Vocabulary,
)
from jetnn.data_processing.base_data_classes import (
    SampleData
)
from jetnn.data_processing.tree_representation.my_code_tree import MyCodeTree


class MethodNamingDataset(Dataset):
    _log_file = "bad_samples.log"
    _separator = "|"

    def __init__(
        self, data_file: str, config: DictConfig, vocabulary: Vocabulary
    ):
        if not exists(data_file):
            raise ValueError(f"Can't find file with data: {data_file}")
        self._data_file = data_file
        self._config = config.data
        self._vocab = vocabulary

        self._line_offsets = get_lines_offsets(data_file)
        self._n_samples = len(self._line_offsets)
        self._empty_chunk = [0 for _ in range(self._config.max_chunks_number)]

        self._code_tree = MyCodeTree(
            self._config.programming_language, self._config.path_to_tree_sitter
        )

        open(self._log_file, "w").close()

    def __len__(self):
        return self._n_samples

    def __getitem__(self, index: int) -> Optional[SampleData]:
        try:
            raw_sample = get_line_by_offset(self._data_file, self._line_offsets[index])
            sample = json.loads(raw_sample)
            label = sample["label"].replace(self._separator, " ")
            tokenized_label = self.tokenize(label, self._config.max_label_tokens)
            cleaned_code, _, method_location = self._code_tree.remove_comments(sample["code"], sample["method_location"])
            tokenized_code = torch.tensor(self.tokenize(sample['code'], self._config.max_code_tokens))
            tokens_split = torch.tensor([])
            if self._config.use_ast_splitter:
                tmp_tokenized_code = list(filter(lambda x: x != self._vocab.pad_id(), tokenized_code))[1:-1]
                tokens = [self._vocab.tokenizer.decode(token) for token in tmp_tokenized_code]
                tokens_split = self._code_tree.process_code(
                    cleaned_code, tokens, self._config.max_chunk_size
                )
                num_splits = min(self._config.max_chunks_number, len(tokens_split))
                tmp_tokens_split = self._empty_chunk.copy()
                tmp_tokens_split[:num_splits] = tokens_split[:num_splits]
                tokens_split = torch.tensor(tmp_tokens_split, dtype=torch.long)
            return SampleData(tokenized_code,
                              label_tokens=tokenized_label,
                              split=tokens_split)
        except ValueError as e:
            with open(self._log_file, "a") as f_out:
                f_out.write(f"Error parsing sample from line #{index}: {e}\n")
            return None

    def tokenize(self, text: str, max_parts: int) -> List[int]:
        return self._vocab.tokenizer.encode(
            text,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_parts,
            truncation="longest_first",
        )
