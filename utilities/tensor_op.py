from typing import Optional

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
