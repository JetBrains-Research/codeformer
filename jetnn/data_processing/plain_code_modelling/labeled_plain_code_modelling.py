from dataclasses import dataclass
from typing import List, Optional, cast

import torch


@dataclass
class LabeledCodeModellingTokens:
    label: torch.tensor
    code: torch.tensor


class BatchedLabeledCodeModellingTokens:
    def __init__(self, all_samples):
        samples = [s for s in all_samples if s is not None]
        self._len = len(samples)
        self.label_tokens = torch.stack([sample.label for sample in samples]) if len(samples) > 0 else torch.tensor([])
        self.label_tokens = self.label_tokens.permute(1, 0)
        self.code_tokens = torch.stack([sample.code for sample in samples]) if len(samples) > 0 else torch.tensor([])
        self.batch_split = torch.tensor([sample.num_splits for sample in samples])

    def __len__(self) -> int:
        return self._len

    def pin_memory(self):
        self.label_tokens.pin_memory()
        self.code_tokens.pin_memory()
        self.batch_split.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.label_tokens = self.label_tokens.to(device)
        self.code_tokens = self.code_tokens.to(device)
        self.batch_split = self.batch_split.to(device)
