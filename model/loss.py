from typing import Tuple

import torch
import torch.nn.functional as F

import model.triplet_mining as triplet_mining
from utilities.metrics import pairwise_distance_matrix, create_class_matrix


def triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    positive_mining_mode: str = "random",
    negative_mining_mode: str = "hard",
    margin: float = 1,
    squared_distance: bool = False,
) -> torch.Tensor:
    """Compute the triplet loss for given embeddings. We use online sampling to
    select the positive and negative samples. You can choose between random and hard
    mining for both the negatives and positives. Additionally, for the negativess, you
    can choose the semi-hard mining strategy. The margin value is used to compute the
    triplet loss. If squared_distance is True, the pairwise distance matrix is squared.

    Parameters:
    -----------
    embeddings: torch.Tensor
        2D tensor of shape (n, d) where n is the number of samples and d is the
        dimensionality of the samples.
    labels: torch.Tensor
        1D tensor of shape (n,) where labels[i] is the int label of the i-th sample.
    positive_mining_mode: str
        Either "random", "easy, or "hard".
    negative_mining_mode: str
        Either "random", "hard" or "semi-hard.
    margin: float
        The margin value in the triplet loss. In the case of semi-hard mining, this
        value is also used to sample the negatives.
    squared_distance: bool
        If True, the pairwise distance matrix is squared before computing the loss.

    Returns:
    --------
    loss: torch.Tensor
        Tensor of shape (1,) representing the average triplet loss of the batch.
    """

    # Compute the pairwise distance matrix of the anchors
    distance_matrix = pairwise_distance_matrix(embeddings, squared=squared_distance)

    # Create masks for the positive and negative samples
    mask_pos, mask_neg = create_pos_neg_masks(labels)

    # Sample the positives first
    if positive_mining_mode.lower() == "random":
        dist_AP, _ = triplet_mining.random_positive_sampling(distance_matrix, mask_pos)
    elif positive_mining_mode.lower() == "hard":
        dist_AP, _ = triplet_mining.hard_positive_mining(distance_matrix, mask_pos)
    elif positive_mining_mode.lower() == "easy":
        dist_AP, _ = triplet_mining.easy_positive_mining(distance_matrix, mask_pos)
    else:
        raise ValueError("Other positive mining types are not supported.")

    if negative_mining_mode.lower() == "random":
        dist_AN, _ = triplet_mining.random_negative_sampling(distance_matrix, mask_neg)
    elif negative_mining_mode.lower() == "hard":
        dist_AN, _ = triplet_mining.hard_negative_mining(distance_matrix, mask_neg)
    elif negative_mining_mode.lower() == "semi-hard":
        dist_AN, _ = triplet_mining.semi_hard_negative_mining(
            distance_matrix, mask_pos, mask_neg, margin
        )
    else:
        raise ValueError("Other negative mining types are not supported.")

    # Compute the triplet loss
    loss = F.relu(dist_AP - dist_AN + margin)

    # Average the loss over the batch
    loss = loss.mean()

    return loss


############################### Utilities ###############################


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

    # Create the class mask
    class_mask = create_class_matrix(labels)

    # Set the diagonal to 0 to avoid using the same sample as a positive or negative
    mask_diag = 1 - torch.eye(labels.size(0), dtype=torch.int32, device=labels.device)
    mask_pos = mask_diag * class_mask
    mask_neg = mask_diag * (1 - mask_pos)

    # Convert to float32
    mask_pos = mask_pos.float()
    mask_neg = mask_neg.float()

    return mask_pos, mask_neg
