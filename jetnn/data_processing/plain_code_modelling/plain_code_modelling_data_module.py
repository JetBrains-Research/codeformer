from pickle import load, dump
from os.path import exists, join, basename
from typing import List, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from jetnn.data_processing.vocabularies.plain.plain_code_vocabulary import (
    PlainCodeVocabulary,
    from_holdout,
)
from jetnn.data_processing.plain_code_modelling.labeled_plain_code_modelling import (
    BatchedLabeledCodeModellingTokens,
)
from jetnn.data_processing.plain_code_modelling.plain_code_modelling_dataset import (
    PlainCodeModellingDataset,
)


class PlainCodeModellingDataModule(LightningDataModule):
    _train = "train"
    _val = "val"
    _test = "test"

    def __init__(self, config: DictConfig):
        super().__init__()
        self._config = config
        self._data_dir = config.data.root
        self._name = basename(self._data_dir)

        self._vocabulary = self.setup_vocabulary()

    @property
    def vocabulary(self) -> PlainCodeVocabulary:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup data module for initializing vocabulary")
        return self._vocabulary

    def setup_vocabulary(self) -> PlainCodeVocabulary:
        vocabulary_path = self._config.data.path
        if not exists(vocabulary_path):
            print("Can't find vocabulary, collect it from train holdout")
            vocab = from_holdout(join(self._data_dir, "train.jsonl"), self._config.data)
            with open(vocabulary_path, "wb") as f:
                dump(vocab, f)
            return vocab
        else:
            with open(vocabulary_path, "rb") as f:
                return load(f)

    @staticmethod
    def collate_wrapper(
        batch,
    ) -> BatchedLabeledCodeModellingTokens:
        return BatchedLabeledCodeModellingTokens(batch)

    def _create_dataset(self, holdout_file: str):
        if self._vocabulary is None:
            raise RuntimeError(f"Setup vocabulary before creating data loaders")
        return PlainCodeModellingDataset(holdout_file, self._config, self._vocabulary)

    def _shared_dataloader(self, holdout: str) -> DataLoader:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup vocabulary before creating data loaders")

        holdout_file = join(self._data_dir, f"{holdout}.jsonl")
        dataset = self._create_dataset(holdout_file)
        dataloader_config = self._config[holdout]["dataloader"]

        return DataLoader(dataset, collate_fn=self.collate_wrapper, **dataloader_config)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._train)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._val)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._test)

    def predict_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.test_dataloader(*args, **kwargs)

    def transfer_batch_to_device(
        self, batch, device: torch.device, dataloader_idx: int
    ):
        batch.move_to_device(device)
        return batch
