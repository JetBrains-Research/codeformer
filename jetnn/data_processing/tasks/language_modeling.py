from dataclasses import dataclass
from typing import List, Optional, cast

import torch


@dataclass
class TextTokens:
    text: torch.tensor # can be code, not just text
    split: torch.tensor


def transpose(list_of_lists: List[List[int]]) -> List[List[int]]:
    return [cast(List[int], it) for it in zip(*list_of_lists)]


class BatchedTextTokens:
    def __init__(self, all_samples: List[Optional[TextTokens]]):
        samples = [s for s in all_samples if s is not None]
        self._len = len(samples)
        self.text_tokens = (
            torch.cat([sample.text for sample in samples])
            if self._len > 0
            else torch.tensor([])
        )
        self.batch_split = (
            torch.cat([sample.split for sample in samples])
            if self._len > 0
            else torch.tensor([])
        )

    def __len__(self) -> int:
        return self._len

    def pin_memory(self) -> "BatchedTextTokens":
        self.text_tokens.pin_memory()
        self.batch_split.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.text_tokens = self.code_tokens.to(device)
        self.batch_split = self.batch_split.to(device)
