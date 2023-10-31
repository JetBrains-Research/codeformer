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
            last_split_size = current_split[splits_length[batch_idx].item() - 1]
            last_chunks[batch_idx][:last_split_size] = input_ids[batch_idx][p_sum : p_sum + last_split_size]
        return result, last_chunks


def pad_to_match_linear_layer(
        x: torch.Tensor,
        split_sizes: torch.Tensor,
        context_size: int,
        device: str="cuda",
        last_chunk_omitted: bool=False
    ) -> torch.Tensor:
        shift = int(last_chunk_omitted)
        batch_split = split_sizes.count_nonzero(dim=1)
        max_split = torch.max(batch_split).item() - shift
        num_splits = batch_split.size()[0]
        result = torch.zeros(
            (num_splits, max_split, context_size, x.shape[2]),
            dtype=torch.float,
        ).to(device)
        p_sum = 0
        for i, split in enumerate(batch_split):
            split_size = split - shift
            result[i][:split_size, : x.shape[1]] = x[p_sum : p_sum + split_size]
            p_sum += split_size
        return result


# here we assume that last chunk is omitted
def generate_chunk_level_attention_mask(
        split_sizes: torch.Tensor,
        device: str="cuda",
) -> torch.Tensor:
        batch_split = split_sizes.count_nonzero(dim=1)
        max_split = torch.max(batch_split).item()
        num_splits = batch_split.size()[0]
        result = torch.zeros(
            (num_splits, max_split),
            dtype=torch.long,
        ).to(device)
        for i, split in enumerate(batch_split):
            result[i][:split] = 1 # would be split + 1 in case of chunk is not omitted
        return result


def concat_with_bos_embedding(
        x: torch.Tensor,
        bos_embedding: torch.Tensor,
    ) -> torch.Tensor:
        num_batches = x.size()[0]
        bos_embeddings = bos_embedding.repeat(num_batches, 1)
        bos_embeddings = torch.unsqueeze(bos_embeddings, 1)
        return torch.cat((bos_embeddings, x), dim=1)


def generate_context_for_last_decoder(
        input_ids: torch.Tensor,
        input_embeddings: torch.Tensor,
        chunk_representations: torch.Tensor,
        splits_size: torch.Tensor,
        pad_id: int,
        device: str="cuda",
    ):
    splits_length = [split_size - 1 for split_size in (splits_size != pad_id).count_nonzero(dim=1)]
    num_batches = chunk_representations.size()[0]
    result = torch.zeros((input_embeddings.size()[0], input_embeddings.size()[1] + 1, input_embeddings.size()[2]), dtype=torch.float).to(device)
    labels = torch.zeros((input_embeddings.size()[0], input_embeddings.size()[1] + 1), dtype=torch.long).to(device)
    p_sum = 0
    
    for batch_num in range(num_batches):
        split_size = splits_length[batch_num]
        
        labels[p_sum: p_sum + split_size, 1:] = input_ids[p_sum: p_sum + split_size]
        batch_input_embeddings = input_embeddings[p_sum: p_sum + split_size]
        batch_chunk_representations = chunk_representations[batch_num][:split_size]

        shifted_batch_chunk_representations = torch.zeros(batch_chunk_representations.size(), dtype=torch.float).to(device)
        shifted_batch_chunk_representations[1: split_size] = batch_chunk_representations[:split_size - 1]
        shifted_batch_chunk_representations = shifted_batch_chunk_representations.unsqueeze(dim=1)

        result[p_sum: p_sum + split_size] = torch.cat((shifted_batch_chunk_representations, batch_input_embeddings), dim=1)
        p_sum += split_size
    
    return result, labels, None, None
