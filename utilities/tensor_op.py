from typing import Optional, Tuple

import torch


def l2_normalize(x: torch.Tensor, eps: float = 1e-12, precision: str = "high"):
    """L2 normalize the input tensor. In the case that the norm is small,,
    a small value is added to the norm. This is to avoid division by zero or a
    very small value. You can control the precision of this operation by setting
    the `precision` parameter.
    """

    assert x.dim() == 2, "Input tensor must be 2D"
    assert precision in ["high", "mid", "low"], "Invalid precision value"

    # L2 normalization
    norms = torch.norm(x, p=2, dim=1, keepdim=True)

    # Add a small value to zero norm entries
    if precision == "high":
        norms = norms + ((norms == 0).type_as(norms) * eps)
    elif precision == "mid":
        norms = torch.clamp(norms, min=eps)
    elif precision == "low":
        norms = norms + eps

    # L2 Normalize the embeddings
    x = x / norms

    return x


def pairwise_distance_matrix(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    squared: bool = False,
    eps: float = 1e-12,
    precision: str = "high",
) -> torch.Tensor:
    """Pairwise Euclidean distance matrix between the elements of two tensors.

    Parameters:
    -----------
    x: torch.Tensor
        2D tensor of shape (n, d) where n is the number of samples and d is the
        dimensionality of the samples. Must be be a FloatTensor.
    y: torch.Tensor
        2D tensor of shape (m, d) where m is the number of samples and d is the
        dimensionality of the samples. If None, then y = x.
    squared: bool
        If True, return the squared Euclidean distance.
    precision: str
        If squared is False, during sqrt operation, operations are needed to avoid
        numerical errors. You can control the precision of this operation by setting
        the `precision` parameter. The options are "high", "mid", and "low".

    Returns:
    --------
    dist: torch.Tensor
        2D tensor of shape (n, m) where dist[i, j] is the Euclidean distance between
        x[i] and y[j]. If squared is True, then dist[i, j] is the squared Euclidean
        distance between x[i] and y[j].
    """

    assert x.dim() == 2, "Input tensors must be 2D"
    assert y is None or y.dim() == 2, "Input tensors must be 2D"
    assert precision in ["high", "mid", "low"], "Invalid precision value"

    x_norm_sq = x.pow(2).sum(1).view(-1, 1)
    if y is not None:
        y_norm_sq = y.pow(2).sum(1).view(1, -1)
    else:
        y = x
        y_norm_sq = x_norm_sq.view(1, -1)
    dist_sq = x_norm_sq - 2 * torch.mm(x, y.t().contiguous()) + y_norm_sq

    # Deal with floating point errors
    dist_sq.clamp_(min=0.0)  # TODO: why not eps?
    if squared:
        return dist_sq
    else:

        if precision == "high":
            # Introduce an epsilon to zero distances for numerical stability
            mask = (dist_sq == 0).type_as(dist_sq)
            dist_sq = dist_sq + (mask * eps)
            dist = torch.sqrt(dist_sq)
            # Correct the epsilon added
            dist = dist * (1.0 - mask)
            # dist[mask.bool()] = 0 # In-place operation is not allowed for backprop
        elif precision == "mid":
            dist_sq = torch.clamp(dist_sq, min=eps)
            dist = torch.sqrt(dist_sq)
        else:
            dist = torch.sqrt(dist_sq + eps)

        return dist


# NOTE: there is sth going on with cos sim!
def pairwise_cosine_similarity(
    x: torch.Tensor, y: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute cosine similarity between the elements of two tensors

    Parameters:
    -----------
    x: torch.Tensor
        2D tensor of shape (n, d) where n is the number of samples and d is the
        dimensionality of the samples. Must be be a FloatTensor.
    y: torch.Tensor
        2D tensor of shape (m, d) where m is the number of samples and d is the
        dimensionality of the samples. If None, then y = x.

    Returns:
    --------
    similarity: torch.Tensor
        2D tensor of shape (n, m) where similarity[i, j] is the cosine similarity
        between x[i] and y[j].
    """

    assert x.dim() == 2, "Input tensors must be 2D"
    assert y is None or y.dim() == 2, "Input tensors must be 2D"

    x = l2_normalize(x, precision="low")
    if y is not None:
        y = l2_normalize(y, precision="low")
    else:
        y = x

    return torch.mm(x, y.t())


def pairwise_dot_product(
    x: torch.Tensor, y: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute dot product between the elements of two tensors

    Parameters:
    -----------
    x: torch.Tensor
        2D tensor of shape (n, d) where n is the number of samples and d is the
        dimensionality of the samples. Must be be a FloatTensor.
    y: torch.Tensor
        2D tensor of shape (m, d) where m is the number of samples and d is the
        dimensionality of the samples. If None, then y = x.

    Returns:
    --------
    similarity: torch.Tensor
        2D tensor of shape (n, m) where similarity[i, j] is the dot product
        between x[i] and y[j].
    """

    assert x.dim() == 2, "Input tensors must be 2D"
    assert y is None or y.dim() == 2, "Input tensors must be 2D"

    if y is not None:
        return torch.mm(x, y.t())
    else:
        return torch.mm(x, x.t())


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


def create_pos_neg_masks(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given ground truth labels, create masks for the positive and negative samples
    in the batch. The positive mask indicates if the samples are from the same clique,
    and the negative mask indicates if the samples are from different cliques. Diagonals
    in both masks are set to 0 to avoid using the same sample as a positive or negative.

    Parameters:
    -----------
    labels: torch.Tensor
        1D tensor of shape (n,) where n is the number of samples and labels[i] is the
        integer label of the i-th sample.

    Returns:
    --------
    mask_pos: torch.Tensor
        2D tensor of shape (n, n) where mask_pos[i, j] is 1 if ytrue[i] == ytrue[j]
        and 0 otherwise. dtype is torch.float32
    mask_neg: torch.Tensor
        2D tensor of shape (n, n) where mask_neg[i, j] is 1 if ytrue[i] != ytrue[j]
        and 0 otherwise. dtype is torch.float32
    """

    assert labels.dim() == 1, f"labels must be a 1D tensor not {labels.dim()}D"

    # Create masks for the positive and negative samples, an anchor is not considered
    # positive to itself
    mask_pos = create_class_matrix(labels, zero_diagonal=True)
    mask_neg = (1 - mask_pos).fill_diagonal_(0)

    # Convert to float32
    mask_pos = mask_pos.float()
    mask_neg = mask_neg.float()

    return mask_pos, mask_neg
