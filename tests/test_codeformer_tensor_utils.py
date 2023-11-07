import torch
from lm.utils import disassemble, assemble, assemble_decoder_inputs, expand_filler


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


def test_assemble_decoder_inputs():
    source = torch.tensor([[1, 2, 0], [3, 4, 5]]).unsqueeze(2)

    # a)
    # print(torch.tensor([, source[:, :-1].shape)
    source_shifted = torch.cat([expand_filler(0, source.shape), source[:, :-1]], 1)

    sizes = torch.tensor([[2, 1, 0],
                          [1, 1, 1]],
                         dtype=torch.long)

    from_starting_points = torch.tensor([[0, 1, 0],
                                         [0, 1, 2]],
                                        dtype=torch.long)

    to_starting_points = torch.tensor([[2, 1, 0],
                                       [0, 0, 0]])

    device = torch.device('cpu')
    max_len = 5

    output = assemble_decoder_inputs(source,
                                     sizes,
                                     from_starting_points,
                                     to_starting_points,
                                     max_len).squeeze(2)

    expected_output = torch.tensor(
        [[0, 0, 1, 2, 0],
         [0, 2, 0, 0, 0],
         [3, 0, 0, 0, 0],
         [4, 0, 0, 0, 0],
         [5, 0, 0, 0, 0]]
    )
    assert torch.all(output == expected_output)
