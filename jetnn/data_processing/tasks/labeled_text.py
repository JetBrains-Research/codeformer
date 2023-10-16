from dataclasses import dataclass
from typing import List, Optional, cast

import torch


@dataclass
class LabeledTextTokens:
    label: List[int]
    text: torch.tensor # can be code, not just text
    split: torch.tensor


def transpose(list_of_lists: List[List[int]]) -> List[List[int]]:
    return [cast(List[int], it) for it in zip(*list_of_lists)]


class BatchedLabeledTokens:
    def __init__(self, all_samples: List[Optional[LabeledTextTokens]]):
        samples = [s for s in all_samples if s is not None]
        self._len = len(samples)
        self.label_tokens = torch.tensor(
            transpose([sample.label for sample in samples]), dtype=torch.long
        )
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

    def pin_memory(self) -> "BatchedLabeledTokens":
        self.label_tokens.pin_memory()
        self.text_tokens.pin_memory()
        self.batch_split.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.label_tokens = self.label_tokens.to(device)
        self.text_tokens = self.text_tokens.to(device)
        self.batch_split = self.batch_split.to(device)
