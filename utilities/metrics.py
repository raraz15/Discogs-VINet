import time

import torch

from torch.utils.data import DataLoader

from model.nets import CQTNet
from utilities.tensor_op import (
    pairwise_cosine_similarity,
    pairwise_distance_matrix,
    pairwise_dot_product,
    create_class_matrix,
)
from utilities.utils import format_time


@torch.no_grad()
def evaluate(
    model: CQTNet,
    loader: DataLoader,
    similarity_search: str,
    chunk_size: int,
    noise_works: bool,
    amp: bool,
    device: str,
) -> dict:
    """Evaluate the model by simulating the retrieval task. Compute the embeddings
    of all versions and calculate the pairwise distances. Calculate the mean average
    precision of the retrieval task. Metric calculations are done on the cpu but you
    can choose the device for the model. Since we normalize the embeddings, MCSS is
    equivalent to NNS. Please refer to the argparse arguments for more information.

    Parameters:
    -----------
    model : CQTNet
        Model to evaluate
    loader : torch.utils.data.DataLoader
        DataLoader containing the test set cliques
    similarity_search: str
        Similarity search function. MIPS, NNS, or MCSS.
    chunk_size : int
        Chunk size to use during metrics calculation.
    noise_works : bool
        Flag to indicate if the dataset contains noise works.
    amp : bool
        Flag to indicate if Automatic Mixed Precision should be used.
    device : str
        Device to use for inference and metric calculation.

    Returns:
    --------
    metrics : dict
        Dictionary containing the evaluation metrics. See utilities.metrics.calculate_metrics
    """

    t0 = time.monotonic()

    model.eval()

    N = len(loader)

    emb_dim = model(
        loader.dataset.__getitem__(0)[0].unsqueeze(0).unsqueeze(1).to(device)
    ).shape[1]

    # Preallocate tensors to avoid https://github.com/pytorch/pytorch/issues/13246
    embeddings = torch.zeros((N, emb_dim), dtype=torch.float32, device=device)
    labels = torch.zeros(N, dtype=torch.int32, device=device)

    print("Extracting embeddings...")
    for idx, (feature, label) in enumerate(loader):
        assert feature.shape[0] == 1, "Batch size must be 1 for inference."
        feature = feature.unsqueeze(1).to(device)  # (1,F,T) -> (1,1,F,T)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            embedding = model(feature)
        embeddings[idx : idx + 1] = embedding
        labels[idx : idx + 1] = label.to(device)
        if (idx + 1) % (len(loader) // 10) == 0 or idx == len(loader) - 1:
            print(f"[{(idx+1):>{len(str(len(loader)))}}/{len(loader)}]")
    print(f"Extraction time: {format_time(time.monotonic() - t0)}")

    # If there are no noise works, remove the cliques with single versions
    # this may happen due to the feature extraction process.
    if not noise_works:
        # Count each label's occurrence
        unique_labels, counts = torch.unique(labels, return_counts=True)
        # Filter labels that occur more than once
        valid_labels = unique_labels[counts > 1]
        # Create a mask for indices where labels appear more than once
        keep_mask = torch.isin(labels, valid_labels)
        if keep_mask.sum() < len(labels):
            print("Removing single version cliques...")
            embeddings = embeddings[keep_mask]
            labels = labels[keep_mask]

    print("Calculating metrics...")
    t0 = time.monotonic()
    metrics = calculate_metrics(
        embeddings,
        labels,
        similarity_search=similarity_search,
        noise_works=noise_works,
        chunk_size=chunk_size,
        device=device,
    )
    print(f"Calculation time: {format_time(time.monotonic() - t0)}")

    return metrics


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
