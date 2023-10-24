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
from jetnn.data_processing.tree_representation.my_text_tree import MyTextTree


class LanguageModelingDataset(Dataset):
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

        self._text_tree = MyTextTree()

        self._empty_chunk = [0 for _ in range(self._config.max_chunks_number)]

        open(self._log_file, "w").close()

    def __len__(self):
        return self._n_samples

    def __getitem__(self, index: int) -> Optional[SampleData]:
        try:
            raw_sample = get_line_by_offset(self._data_file, self._line_offsets[index])
            sample = json.loads(raw_sample)
            tokenized_text = torch.tensor(self.tokenize(sample['text'], self._config.max_text_tokens))
            tokens_split = torch.tensor([])
            if self._config.use_ast_splitter:
                tmp_tokenized_text = list(filter(lambda x: x != self._vocab.pad_id(), tokenized_text))[1:-1]
                tokens = list(map(self._vocab.tokenizer.decode, tmp_tokenized_text))
                tokens_split = self._text_tree.process_text(
                    sample['text'], tokens, self._config.max_chunk_size
                )
                num_splits = min(self._config.max_chunks_number, len(tokens_split))
                tmp_tokens_split = self._empty_chunk.copy()
                tmp_tokens_split[:num_splits] = tokens_split[:num_splits]
                tokens_split = torch.tensor(tmp_tokens_split, dtype=torch.long)
            return SampleData(text_tokens=tokenized_text,
                              label_tokens=None,
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
