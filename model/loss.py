from typing import Tuple, Union

import torch
import torch.nn.functional as F

import model.triplet_mining as triplet_mining
from utilities.tensor_op import pairwise_distance_matrix, create_pos_neg_masks


def triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    positive_mining_mode: str = "random",
    negative_mining_mode: str = "hard",
    margin: float = 1,
    squared_distance: bool = False,
    non_zero_mean: bool = False,
    stats: bool = False,
) -> Tuple[torch.Tensor, Union[int, None]]:
    """Compute the triplet loss for given embeddings. We use online sampling to
    select the positive and negative samples. You can choose between random and hard
    mining for both the negatives and positives. The margin value is used to compute the
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
        Either "random" or "hard".
    margin: float
        The margin value in the triplet loss. In the case of semi-hard mining, this
        value is also used to sample the negatives.
    squared_distance: bool
        If True, the pairwise distance matrix is squared before computing the loss.
    non_zero_mean: bool
        If True, the loss is averaged only over the non-zero losses.
    stats: bool
        If True, return the number of positive triplets in the batch. Else return None.

    Returns:
    --------
    loss: torch.Tensor
        Tensor of shape (1,) representing the average triplet loss of the batch.
    num_unsatisfied_triplets: int
        Number of triplets that do not satisfy the margin condition in the batch.
        Only returned if stats is True.
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
            distance_matrix, dist_AP, mask_neg, margin
        )
    else:
        raise ValueError("Other negative mining types are not supported.")

    # Compute the triplet loss
    loss = F.relu(dist_AP - dist_AN + margin)

    # See how many triplets per batch are positive
    if stats:
        num_unsatisfied_triplets = int((loss > 0).sum().item())
    else:
        num_unsatisfied_triplets = None

    # Average the loss over the batch (can filter out zero losses if needed)
    if non_zero_mean:
        mask = loss > 0
        if any(mask):
            loss = loss[mask]
    loss = loss.mean()

    return loss, num_unsatisfied_triplets
