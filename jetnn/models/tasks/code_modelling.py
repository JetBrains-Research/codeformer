from typing import List, Optional, Any, Dict, Tuple
import torch
from commode_utils.losses import SequenceCrossEntropyLoss
from commode_utils.metrics import SequentialF1Score, ClassificationMetrics
from omegaconf import DictConfig
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection, Metric

from jetnn.data_processing.vocabularies.vocabulary import Vocabulary
from jetnn.metrics.chrf import ChrF
from jetnn.models.decoders.transformer_decoder import MethodNameTransformerDecoder
from jetnn.models.decoders.big_bird_decoder import MethodNameBigBirdDecoder
from jetnn.models.decoders.code_modelling_transformer_decoder import CodeModellingTransformerDecoder
from jetnn.models.encoders.transformers.transformer_encoder import (
    MethodNameTransformerEncoder,
)
from jetnn.models.encoders.transformers.my_transformer_encoder import (
    MethodNameMyTransformerEncoder,
)
from jetnn.models.encoders.transformers.big_bird_encoder import (
    MethodNameBigBirdEncoder,
)
from jetnn.models.encoders.transformers.longformer_encoder import (
    MethodNameLongformerEncoder,
)
from jetnn.models.utils import configure_optimizers_alon


class CodeModellingModel(LightningModule):
    def __init__(self, config, vocab):
        super().__init__()
        self._config = config
        self._vocab = vocab
        self._decoder = self._get_decoder()

        metrics: Dict[str, Metric] = {
            f"{holdout}_f1": SequentialF1Score(
                pad_idx=vocab.pad_id(),
                eos_idx=vocab.eos_id(),
                ignore_idx=[vocab.bos_id(), vocab.unk_id()],
            )
            for holdout in ["train", "val", "test"]
        }
        self._metrics = MetricCollection(metrics)
        self._loss = SequenceCrossEntropyLoss(vocab.pad_id(), reduction="seq-mean")

    def _get_decoder(self):
        if self._config.model.decoder == "code_modelling_transformer":
            return CodeModellingTransformerDecoder(
                self._config.model.code_modelling_transformer, self._vocab, self._config.data.max_subsequence_size
            )
        else:
            raise ValueError("Unknown decoder type")

    def configure_optimizers(self):
        return configure_optimizers_alon(self._config.optimizer, self.parameters())

    def forward(self, batch, step):
        return self._decoder(batch, step)

    # double check indexes
    def _shared_step(self, batch, step):
        logits = self(batch, step)
        logits = logits[1:-1]
        batch.label_tokens = batch.label_tokens[1:]
        result = {f"{step}/loss": self._loss(logits, batch.label_tokens)}
        with torch.no_grad():
            prediction = logits.argmax(-1)
            metric: ClassificationMetrics = self._metrics[f"{step}_f1"](
                prediction, batch.label_tokens
            )
            result.update(
                {
                    f"{step}/f1": metric.f1_score,
                    f"{step}/precision": metric.precision,
                    f"{step}/recall": metric.recall,
                }
            )
        return result
    
    def training_step(self, batch, batch_idx):
        result = self._shared_step(batch, "train")
        self.log_dict(result, on_step=True, on_epoch=False)
        self.log("f1", result["train/f1"], prog_bar=True, logger=False)
        return result["train/loss"]

    def validation_step(self, batch, batch_idx):
        result = self._shared_step(batch, "val")
        return result["val/loss"]

    def test_step(self, batch, batch_idx):
        result = self._shared_step(batch, "test")
        return result["test/loss"]
    
    def _shared_epoch_end(self, step_outputs, step):
        with torch.no_grad():
            losses = [
                so if isinstance(so, torch.Tensor) else so["loss"]
                for so in step_outputs
            ]
            mean_loss = torch.stack(losses).mean()
            metric = self._metrics[f"{step}_f1"].compute()
            log = {
                f"{step}/loss": mean_loss,
                f"{step}/f1": metric.f1_score,
                f"{step}/precision": metric.precision,
                f"{step}/recall": metric.recall,
            }
            self._metrics[f"{step}_f1"].reset()
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, step_outputs):
        self._shared_epoch_end(step_outputs, "train")

    def validation_epoch_end(self, step_outputs):
        self._shared_epoch_end(step_outputs, "val")

    def test_epoch_end(self, step_outputs):
        self._shared_epoch_end(step_outputs, "test")
