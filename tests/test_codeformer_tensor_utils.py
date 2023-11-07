import torch
from lm.utils import disassemble, assemble


def test_disassemble():
    inputs = torch.arange(6).view(2, 3) + 1
    sizes = torch.tensor([[2, 1], [3, 0]], dtype=torch.long)
    output = disassemble(inputs, sizes, 0)
    expected_output = torch.tensor([[1, 2, 0],
                                    [3, 0, 0],
                                    [4, 5, 6]])
    assert torch.all(output == expected_output)


def test_assemble():
    inputs_asm = torch.tensor([[1, 2, 0],
                               [3, 0, 0],
                               [4, 5, 6]])
    sizes = torch.tensor([[2, 1],
                          [3, 0]])
    fill_empty_val = 0
    # sizes = torch.tensor([[2, 0], [1, 2]])

    output = assemble(inputs_asm, sizes, fill_empty_val)

    expected_output = torch.tensor([[[1., 2., 0.],
                                     [3., 0., 0.]],
                                    [[4., 5., 6.],
                                     [0., 0., 0.]]])
    assert torch.all(output == expected_output)


