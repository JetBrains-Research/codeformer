# from torchscale.architecture.config import RetNetConfig
# from torchscale.architecture.retnet import RetNetEncoder


from omegaconf import DictConfig
from torch import nn, Tensor

from jetnn.data_processing.plain_code_method.labeled_plain_code import (
    BatchedLabeledCodeTokens,
)
from jetnn.data_processing.vocabularies.vocabulary import Vocabulary


class MethodNameRetNetEncoder(nn.Module):
    def __init__(self, config: DictConfig, vocab: Vocabulary):
        super().__init__()
        

    def forward(self, batch: BatchedLabeledCodeTokens) -> Tensor:
        pass
