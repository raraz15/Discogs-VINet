import torch

from .tensor_op import (
    pairwise_cosine_similarity,
    pairwise_distance_matrix,
    pairwise_dot_product,
)


def create_class_matrix(
    labels: torch.Tensor, zero_diagonal: bool = False, memory_efficient: bool = False
) -> torch.Tensor:
    """Takes a 1D tensor of integer class labels and creates a binary metrix where each row
    indicates if the columns are from the same clique. It is important to use double
    precision to avoid numerical errors. Believe me.

    Parameters:
    -----------
    labels: torch.Tensor
        1D tensor of shape (n,) where n is the number of samples and labels[i] is the
        integer label of the i-th sample.
    zero_diagonal: bool = False
        If True, set the diagonal of the class matrix to 0.
    memory_efficient: bool = False
        If True, calculate the class matrix for each embedding separately. If False, use
        matrix operations to calculate the class matrix faster.

    Returns:
    --------
    class_matrix: torch.Tensor
        2D tensor of shape (n, n) where class_matrix[i, j] is 1 if labels[i] == labels[j]
        and 0 otherwise. If zero_diagonal is set to True, class_matrix[i, j] = 0
        dtype is torch.int32
    """

    assert labels.dim() == 1, "Labels must be a 1D tensor"

    if memory_efficient:
        class_matrix = torch.zeros(len(labels), len(labels), dtype=torch.int32)
        for i, label in enumerate(labels):
            class_matrix[i] = labels == label
    else:
        class_matrix = (
            pairwise_distance_matrix(
                labels.unsqueeze(1).double(), squared=True, precision="low"
            )
            < 0.5
        ).int()

    if zero_diagonal:
        class_matrix.fill_diagonal_(0)

    return class_matrix


def calculate_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    similarity_search: str = "MIPS",
    noise_works: bool = False,
    chunk_size: int = 256,
    device: str = "cpu",
) -> dict:
    """Perform similarity search for a set of embeddings and calculate the following
    metrics using the ground truth labels.
        mAP => Mean Average Precision
        MRR => Mean Reciprocal Rank of the first correct prediction
        MR => Mean Rank of the first correct prediction
        Top1 => Number of correct predictions in the first closest point
        Top10 => Number of correct predictions in the first 10 closest points

    Adapted from: https://github.com/furkanyesiler/re-move/blob/master/utils/metrics.py

    Parameters:
    -----------
    embeddings: torch.Tensor
        2D tensor of shape (m, n) where m is the number of samples, n is the dimension
        of the embeddings.
    labels: torch.Tensor
        1D tensor of shape (m,) where m is the number of samples and labels[i] is the
        integer label of the i-th sample.
    similarity_search: str = "MIPS"
        The similarity search function to use. "NNS", "MCSS", or "MIPS".
    noise_works: bool = False
        If True, the dataset contains noise works, which are not included in metrics;
        otherwise, they are included.
    chunk_size: int = 256
        The size of the chunks to use during the evaluation.
    device: str = "cpu"
        The device to use for the calculations.

    Returns:
    --------
    metrics: dict
        Dictionary containing the performance metrics.

    """

    assert labels.dim() == 1, "Labels must be a 1D tensor"
    assert (
        embeddings.dim() == 2
    ), f"Embeddings must be a 2D tensor got {embeddings.shape}"
    assert embeddings.size(0) == labels.size(
        0
    ), "Embeddings and labels must have the same size"
    if similarity_search not in ["NNS", "MCSS", "MIPS"]:
        raise ValueError(
            "Similarity must be either euclidean, inner product, or cosine."
        )
    assert chunk_size > 0, "Chunk size must be positive"
    assert chunk_size <= len(
        labels
    ), "Chunk size must be smaller than the number of queries"

    # For unity
    similarity_search = similarity_search.upper()

    # Number of total items in the dataset
    N = len(labels)

    # Number of similar embeddings to consider for each query
    k = N - 1
    # Create the ranking tensor for AP calculation
    ranking = torch.arange(1, k + 1, dtype=torch.float32, device=device).unsqueeze(
        0
    )  # (1, N-1)

    # Each row indicates if the columns are from the same clique,
    # diagonal is set to 0
    class_matrix = create_class_matrix(
        labels, zero_diagonal=True, memory_efficient=True
    ).float()

    # Initialize the tensors for storing the evaluation metrics
    TOP1, TOP10 = torch.tensor([]), torch.tensor([])
    TR, MAP = torch.tensor([]), torch.tensor([])

    # Iterate over the chunks
    for i, (Q, C) in enumerate(
        zip(embeddings.split(chunk_size), class_matrix.split(chunk_size))
    ):

        # Move the tensors to the device
        Q = Q.to(device)
        C = C.to(device)  # Full class matrix would require 53GB of memory

        # Number of relevant items for each query in the chunk
        n_relevant = torch.sum(C, 1)  # (B,)

        # Compute the pairwise similarity matrix
        if similarity_search == "MIPS":
            S = pairwise_dot_product(Q, embeddings)  # (B, N)
        elif similarity_search == "MCSS":
            S = pairwise_cosine_similarity(Q, embeddings)  # (B, N)
        else:
            # Use low precision for faster calculations
            S = -1 * pairwise_distance_matrix(Q, embeddings, precision="low")  # (B, N)

        # Set the similarity of each query with itself to -inf
        torch.diagonal(S, offset=i * chunk_size).fill_(float("-inf"))

        # If Da-TACOS, remove queries with no relevant items (noise works)
        if noise_works:
            non_noise_indices = n_relevant.bool()
            S = S[non_noise_indices]  # (B', N)
            C = C[non_noise_indices]  # (B', N)
            n_relevant = n_relevant[non_noise_indices]  # (B',)

        # Check if there are relevant items for each query
        assert torch.all(
            n_relevant > 0
        ), "There must be at least one relevant item for each query"

        # For each embedding, find the indices of the k most similar embeddings
        _, spred = torch.topk(S, k, dim=1)  # (B', N-1)
        # Get the relevance values of the k most similar embeddings
        relevance = torch.gather(C, 1, spred)  # (B', N-1)

        # Number of relevant items in the top 1 and 10
        top1 = relevance[:, 0].int().cpu()
        top10 = relevance[:, :10].int().sum(1).cpu()

        # Get the rank of the first correct prediction by tie breaking
        temp = (
            torch.arange(k, dtype=torch.float32, device=device).unsqueeze(0) * 1e-6
        )  # (1, N-1)
        _, sel = torch.topk(relevance - temp, 1, dim=1)  # (B', 1)
        top_rank = sel.squeeze(1).float().cpu() + 1  # (B',)

        # Calculate the average precision for each embedding
        prec_at_k = torch.cumsum(relevance, 1).div_(ranking)  # (B', N-1)
        ap = torch.sum(prec_at_k * relevance, 1).div_(n_relevant).cpu()  # (B',)

        # Concatenate the results from all chunks
        TR = torch.cat((TR, top_rank))
        MAP = torch.cat((MAP, ap))
        TOP1 = torch.cat((TOP1, top1))
        TOP10 = torch.cat((TOP10, top10))

    # computing the final evaluation metrics
    TOP1 = TOP1.int().sum().item()
    TOP10 = TOP10.int().sum().item()
    MR1 = TR.mean().item()
    MRR = (1 / TR).mean().item()
    MAP = MAP.mean().item()

    # storing the evaluation metrics
    metrics = {
        "MAP": round(MAP, 3),
        "MRR": round(MRR, 3),
        "MR1": round(MR1, 2),
        "Top1": TOP1,
        "Top10": TOP10 if k > 10 else None,
    }

    # printing the evaluation metrics
    for k, v in metrics.items():
        if k in ["Top1", "Top10"]:
            print(f"{k:>5}: {v}")
        else:
            print(f"{k:>5}: {v:.3f}")

    return metrics
