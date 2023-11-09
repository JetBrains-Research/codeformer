import torch
from lm.utils import (disassemble, assemble, assemble_decoder_inputs,
                      expand_filler, put_token_embeddings_at_specified_positions,
                      prepare_token_ids_for_decoder)


def test_disassemble():
    inputs = torch.arange(6).view(2, 3) + 1
    sizes = torch.tensor([[2, 1], [3, 0]], dtype=torch.long)
    output = disassemble(inputs, sizes, 0)
    expected_output = torch.tensor([[1, 2, 0],
                                    [3, 0, 0],
                                    [4, 5, 6]])
    assert torch.all(output == expected_output)


def test_disassemble_bos():
    bos_id = 9
    inputs = torch.arange(6).view(2, 3) + 1
    sizes = torch.tensor([[2, 1], [3, 0]], dtype=torch.long)
    output = disassemble(inputs, sizes, 0, bos_id)
    expected_output = torch.tensor([[9, 1, 2, 0],
                                    [9, 3, 0, 0],
                                    [9, 4, 5, 6]])
    assert torch.all(output == expected_output)


def test_disassemble_eos():
    eos_id = 9
    inputs = torch.arange(6).view(2, 3) + 1
    sizes = torch.tensor([[2, 1], [3, 0]], dtype=torch.long)
    output = disassemble(inputs, sizes, 0, eos_value=eos_id)
    expected_output = torch.tensor([[1, 2, 9, 0],
                                    [3, 9, 0, 0],
                                    [4, 5, 6, 9]])
    assert torch.all(output == expected_output)


def test_disassemble_bos_eos():
    eos_id = 9
    bos_id = 8
    inputs = torch.arange(6).view(2, 3) + 1
    sizes = torch.tensor([[2, 1], [3, 0]], dtype=torch.long)
    output = disassemble(inputs, sizes, 0, bos_value=bos_id, eos_value=eos_id)
    expected_output = torch.tensor([[8, 1, 2, 9, 0],
                                    [8, 3, 9, 0, 0],
                                    [8, 4, 5, 6, 9]])
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


def test_put_token_embeddings_at_specified_positions():
    token_embs_by_chunk_flat = torch.tensor(
        [[1, 2, 3, 0],
         [1, 4, 5, 3]]
    ).view(2, 4, 1)
    expected_output = torch.tensor([[0, 0, 1, 2, 3, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 4, 5, 3, 0, 0, 0]]).view(2, 10, 1)
    start_positions = torch.tensor([2, 3])
    max_len = 10
    prev_and_curr_chunk_lens_plus_bos_eos = torch.tensor([3, 4])
    output = put_token_embeddings_at_specified_positions(token_embs_by_chunk_flat,
                                                         prev_and_curr_chunk_lens_plus_bos_eos,
                                                         start_positions,
                                                         max_len)
    assert torch.all(output == expected_output)


def test_prepare_token_ids_for_decoder():
    token_ids = torch.tensor([
        [1, 2, 3],
        [4, 5, 0]
    ])
    sizes = torch.tensor([
        [1, 2],
        [1, 0]
    ])
    pad_id = 0
    bos_id = 8
    eos_id = 9

    expected_output = torch.tensor([[8, 1, 9, 0, 0],
                                    [8, 1, 2, 3, 9],
                                    [8, 4, 9, 0, 0]])
    expected_lens = torch.tensor([3, 5, 3])

    output, lens = prepare_token_ids_for_decoder(token_ids,
                                                 sizes,
                                                 pad_id,
                                                 bos_id,
                                                 eos_id)

    assert torch.all(output == expected_output)
    assert torch.all(lens == expected_lens)
