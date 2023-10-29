from dataclasses import dataclass

import torch
from transformers import DebertaV2Model

from lm.deberta_patch import patch_deberta_causal
from lm.model import PatchedDebertaAsCausalLM


@dataclass
class ExpectedJacobians:
    regular_jacob_sample_0 = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=torch.long)

    patch_jacob_sample_0 = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
   ], dtype=torch.long)

    regular_jacob_sample_1 = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
   ], dtype=torch.long)

    patch_jacob_sample_1 = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]
   ], dtype=torch.long)


def test_causal_patching_deberta():
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    model = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base').to(device)
    regular_jacob = get_test_case_jacob(model, device)
    model = patch_deberta_causal(model)
    patch_jacob = get_test_case_jacob(model, device)

    expected_jac = ExpectedJacobians()

    r_jac_0 = regular_jacob.abs().sum(-1).bool().long()[0, :, 0, :]
    p_jac_0 = patch_jacob.abs().sum(-1).bool().long()[0, :, 0, :]
    r_jac_1 = regular_jacob.abs().sum(-1).bool().long()[1, :, 1, :]
    p_jac_1 = patch_jacob.abs().sum(-1).bool().long()[1, :, 1, :]
    assert torch.all(r_jac_0 == expected_jac.regular_jacob_sample_0)
    assert torch.all(p_jac_0 == expected_jac.patch_jacob_sample_0)
    assert torch.all(r_jac_1 == expected_jac.regular_jacob_sample_1)
    assert torch.all(p_jac_1 == expected_jac.patch_jacob_sample_1)


def test_causal_patching_deberta_stand_alone_class():
    device = torch.device('cpu')
    model = PatchedDebertaAsCausalLM.from_pretrained('microsoft/deberta-v3-base')
    patch_jacob = get_test_case_jacob(model, device)
    expected_jac = ExpectedJacobians()
    p_jac_0 = patch_jacob.abs().sum(-1).bool().long()[0, :, 0, :]
    p_jac_1 = patch_jacob.abs().sum(-1).bool().long()[1, :, 1, :]
    assert torch.all(p_jac_0 == expected_jac.patch_jacob_sample_0)
    assert torch.all(p_jac_1 == expected_jac.patch_jacob_sample_1)


def get_test_case_jacob(model, device):
    batch_size = 2
    seq_len = 4
    hidden_size = model.config.hidden_size
    embs = torch.randn(batch_size, seq_len, hidden_size, device=device)
    att_mask = torch.ones(batch_size, seq_len, device=device)
    att_mask[0, 2:] = 0.0

    def frw(x):
        output = model.forward(inputs_embeds=x, attention_mask=att_mask)
        if hasattr(output, 'logits'):
            return output.logits.sum(2)
        else:
            return output.last_hidden_state.sum(2)

    jacob = torch.autograd.functional.jacobian(frw, (embs,))[0]
    return jacob
