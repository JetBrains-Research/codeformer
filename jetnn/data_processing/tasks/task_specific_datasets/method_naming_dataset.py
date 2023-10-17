import json
from os.path import exists
from typing import Optional, List

from commode_utils.filesystem import get_lines_offsets, get_line_by_offset
from omegaconf import DictConfig
from torch.utils.data import Dataset

from jetnn.data_processing.vocabularies.plain.plain_code_vocabulary import (
    PlainCodeVocabulary,
)
from jetnn.data_processing.tasks.language_modeling import (
    TextTokens
)
from jetnn.data_processing.tree_representation.my_code_tree import MyCodeTree


class CodeModelingDataset(Dataset):
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

    def __getitem__(self, index) -> Optional[TextTokens]:
        try:
            raw_sample = get_line_by_offset(self._data_file, self._line_offsets[index])
            sample = json.loads(raw_sample)
            cleaned_code, _, _ = self._code_tree.remove_comments(sample["code"])
            tokenized_code = self.tokenize(cleaned_code, self._config.max_code_parts)
            tokenized_code = list(filter(lambda x: x != self._vocab.pad_id(), tokenized_code))[1:-1]
            tokens = [self._vocab.tokenizer.decode(token) for token in tokenized_code]
            tokens_split = self._code_tree.process_code(
                cleaned_code, tokens, self._config.max_chunk_size
            )
            num_splits = min(self._config.max_chunks_number, len(tokens_split))
            tokens_split = tokens_split[:num_splits]
            return TextTokens(tokenized_code, tokens_split)
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
