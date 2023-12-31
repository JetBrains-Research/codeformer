import torch
from sacrebleu import CHRF
from torchmetrics import Metric

from jetnn.data_processing.vocabularies.vocabulary import Vocabulary


class ChrF(Metric):
    def __iter__(self):
        pass

    def __init__(self, vocab: Vocabulary, **kwargs):
        super().__init__(**kwargs)
        self.__vocab = vocab
        self.__chrf = CHRF()

        # Metric states
        self.add_state(
            "chrf", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        """Calculated ChrF metric on predicted tensor w.r.t. target tensor.

        :param predicted: [pred seq len; batch size] -- tensor with predicted tokens
        :param target: [target seq len; batch size] -- tensor with ground truth tokens
        :return:
        """
        batch_size = target.shape[1]
        if predicted.shape[1] != batch_size:
            raise ValueError(
                f"Wrong batch size for prediction (expected: {batch_size}, actual: {predicted.shape[1]})"
            )

        for batch_idx in range(batch_size):
            target_seq = [token.item() for token in target[:, batch_idx]]
            predicted_seq = [token.item() for token in predicted[:, batch_idx]]

            target_str = self.__vocab.decode(target_seq)
            predicted_str = self.__vocab.decode(predicted_seq)

            if target_str == "":
                # Empty target string mean that the original string encoded only with <UNK> token
                continue

            self.chrf += self.__chrf.sentence_score(predicted_str, [target_str]).score
            self.count += 1

    def compute(self) -> torch.Tensor:
        return self.chrf / self.count
