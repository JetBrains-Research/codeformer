from typing import List, Optional, Any, Dict, Tuple

import torch
from omegaconf import DictConfig
from torch import nn
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from jetnn.data_processing.vocabularies.vocabulary import Vocabulary
from jetnn.models.lm.codeformer_lm import (
    CodeformerLM,
)
from jetnn.models.lm.gpt_lm import (
    GPTLM,
)
from jetnn.models.utils.utils import configure_optimizers_alon


class LanguageModelingModel(LightningModule):
    def __init__(self, config: DictConfig, vocab: Vocabulary) -> None:
        super().__init__()
        self._config = config
        self._vocab = vocab
        self._lm = self._get_lm()

    def _get_lm(self) -> nn.Module:
        if self._config.model.LM == "CodeformerLM":
            return CodeformerLM(
                self._config.model.CodeformerLM, self._config.data, self._vocab
            )
        elif self._config.model.LM == "GPTLM":
            return GPTLM(
                self._config.model.GPTLM, self._config.data, self._vocab
            )
        else:
            raise ValueError("Unknown LM type")

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        return configure_optimizers_alon(self._config.optimizer, self.parameters())

    def calc_loss_and_perplexity(self, logits: torch.Tensor, targets: torch.LongTensor, pad_id: int) -> torch.Tensor:
        # logits and targets must be compatible with torch.nn.functional.cross_entropy
        # check pad_id != end of chunk
        loss_tens = torch.nn.functional.cross_entropy(logits, targets, reduction='none', ignore_index=pad_id)
        loss = loss_tens.sum() / torch.count_nonzero(loss_tens)
        ppl = loss.exp()
        return loss, ppl

    def forward(self, batch, step: Optional[str]) -> Any:
        if step is None:
            step = "test"
        return self._lm(batch)

    def _shared_step(self, batch, step: str) -> Dict:
        logits, labels, hf_loss = self(batch, step)
        logits = logits.flatten(0, 1)
        labels = labels.flatten(0, 1)
        loss, ppl = self.calc_loss_and_perplexity(logits=logits, targets=labels, pad_id=self._vocab.pad_id())
        result = {f"{step}/loss": loss, f"{step}/hf_loss": hf_loss, f"{step}/ppl": ppl}
        return result

    def training_step(self, batch, batch_idx: int) -> Dict:  # type: ignore
        result = self._shared_step(batch, "train")
        self.log_dict(result, on_step=True, on_epoch=False)
        self.log("hf_loss", result["train/hf_loss"], prog_bar=True, logger=False)
        self.log("ppl", result["train/ppl"], prog_bar=True, logger=False)
        return result["train/loss"]

    def validation_step(self, batch, batch_idx: int) -> Dict:  # type: ignore
        result = self._shared_step(batch, "val")
        self.log_dict(result, on_step=True, on_epoch=False)
        # self.log("ppl", result["val/ppl"], prog_bar=True, logger=False)
        return result["val/loss"]

    def test_step(self, batch, batch_idx: int) -> Dict:  # type: ignore
        result = self._shared_step(batch, "test")
        return result["test/loss"]

    def _shared_epoch_end(self, step_outputs, step: str) -> None:
        with torch.no_grad():
            losses = [
                so if isinstance(so, torch.Tensor) else so["loss"]
                for so in step_outputs
            ]
            mean_loss = torch.stack(losses).mean()
            log = {
                f"{step}/loss": mean_loss,
            }
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, step_outputs) -> None:
        self._shared_epoch_end(step_outputs, "train")

    def validation_epoch_end(self, step_outputs) -> None:
        self._shared_epoch_end(step_outputs, "val")

    def test_epoch_end(self, step_outputs) -> None:
        self._shared_epoch_end(step_outputs, "test")
