import torch
from torch import Tensor, LongTensor, nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: nn.Module,
             dl: DataLoader,
             device: torch.device,
             split: str,
             preprocessor: callable,
             postprocessor: callable) -> dict[str, float]:

    model.eval()
    results = []
    for batch in dl:
        batch = batch.to(device)
        inputs = preprocessor(batch, device)
        outputs = model(**inputs)
        outputs = postprocessor(batch, outputs)
        results.append(outputs)
    log_probs_sum = 0
    total_num_tokens = 0
    total_loss = 0
    total_samples = 0
    for res in results:
        log_probs_sum = log_probs_sum + res['loss'] * res['num_tokens']
        total_num_tokens = total_num_tokens + res['num_tokens']
        total_loss = total_loss + res['loss'] * res['batch_size']
        total_samples = total_samples + res['batch_size']
    ppl = torch.exp(log_probs_sum / total_num_tokens)
    loss = total_loss / total_samples
    logs = {'ppl': ppl, 'loss': loss}
    model.train()
    return {f'{split}_{key}': val.item() for key, val in logs.items()}


def metrics(logits: Tensor, targets: LongTensor, pad_id: int) -> dict[str, Tensor | int]:
    # logits.shape: [batch_size, ..., V]
    # targets.shape: [batch_size, ...]
    #
    # ... might contain any dimensions, however, ... are the same for logits
    # and targets.
    # TODO: do not predict stop token for PPX
    batch_size = logits.shape[0]
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)
    loss_tens = torch.nn.functional.cross_entropy(logits, targets, reduction='none', ignore_index=pad_id)
    num_tokens = torch.count_nonzero(loss_tens)
    loss = loss_tens.sum() / num_tokens
    ppl = loss.exp()
    return {
        'loss_tens': loss_tens,
        'loss': loss,
        'ppl': ppl,
        'num_tokens': num_tokens,
        'batch_size': batch_size
    }

