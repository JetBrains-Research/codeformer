import torch


def generate_batches_from_splits(
        input_ids: torch.Tensor,
        splits_size: torch.Tensor,
        context_size: int,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        device: str="cuda",
    ) -> torch.Tensor:
        splits_length = (splits_size != pad_id).count_nonzero(dim=1)
        num_splits = torch.sum(splits_length).item()
        result = torch.zeros((num_splits, context_size), dtype=torch.long).to(device)
        result_idx = 0
        for batch_idx in range(len(splits_size)):
            p_sum = 1 # equal to 1, because tokenizer adds begin token, and we want to omit it
            current_split = splits_size[batch_idx]
            for split_idx in range(splits_length[batch_idx].item()):
                current_split_size = current_split[split_idx]
                result[result_idx][0] = bos_id
                result[result_idx][current_split_size + 1] = eos_id
                result[result_idx][1 : 1 + current_split_size] = input_ids[batch_idx][
                    p_sum : p_sum + current_split_size
                ]
                p_sum += current_split_size
                result_idx += 1
        return result


def generate_batches_from_splits_without_last_chunk(
        input_ids: torch.Tensor,
        splits_size: torch.Tensor,
        context_size: int,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        device: str="cuda",
    ) -> torch.Tensor:
        splits_length = (splits_size != pad_id).count_nonzero(dim=1)
        num_batches = splits_size.size()[0]
        num_splits = torch.sum(splits_length).item() - num_batches
        result = torch.zeros((num_splits, context_size), dtype=torch.long).to(device)
        last_chunks = torch.zeros((num_batches, context_size), dtype=torch.long).to(device)
        result_idx = 0
        for batch_idx in range(num_batches):
            p_sum = 1 # equal to 1, because tokenizer adds begin token, and we want to omit it
            current_split = splits_size[batch_idx]
            for split_idx in range(splits_length[batch_idx].item() - 1):
                current_split_size = current_split[split_idx]
                result[result_idx][0] = bos_id
                result[result_idx][current_split_size + 1] = eos_id
                result[result_idx][1 : 1 + current_split_size] = input_ids[batch_idx][
                    p_sum : p_sum + current_split_size
                ]
                p_sum += current_split_size
                result_idx += 1
            last_split_size = current_split[split_idx]
            last_chunks[batch_idx][:last_split_size] = input_ids[batch_idx][p_sum : p_sum + last_split_size]
        return result, last_chunks


def pad_to_match_linear_layer(
        x: torch.Tensor,
        split_sizes: torch.Tensor,
        context_size: int,
        device: str="cuda",
    ) -> torch.Tensor:
        batch_split = split_sizes.count_nonzero(dim=1)
        max_splits = torch.max(batch_split).item()
        num_splits = batch_split.size()[0]
        result = torch.zeros(
            (num_splits, max_splits, context_size, x.shape[2])
        ).to(device)
        p_sum = 0
        for i, split in enumerate(batch_split):
            result[i][:split, : x.shape[1]] = x[p_sum : p_sum + split]
            p_sum += split
        return result
