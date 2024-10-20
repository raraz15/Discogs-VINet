"""Adapted from: https://github.com/furkanyesiler/re-move/blob/master/utils/loss_utils.py"""

from typing import Tuple

import torch

############################### Random Sampling ###############################


def random_positive_sampling(
    distance_matrix: torch.Tensor, mask_pos: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given a pairwise distance matrix of all the samples and a mask that indicates the
    possible indices for sampling positives, randomly sample a positive sample for each
    anchor. All samples are treated as anchor points.

    Parameters:
    -----------
    distance_matrix: torch.Tensor
        2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
        x[i] and y[j], i.e. the pairwise distance matrix.
    mask_pos: torch.Tensor
        See create_pos_neg_masks() for details.

    Returns:
    --------
    anchor_pos_distances: torch.Tensor
        1D tensor of shape (n,), distances between the anchors and their chosen
        positive samples.
    positive_indices: torch.Tensor
        1D tensor of shape (n,) where positive_indices[i] is the index of the positive
        sample for the i-th anchor point.
    """

    # Get the indices of the positive samples for each anchor point
    positive_indices = torch.multinomial(mask_pos, 1)

    # Get the distances between the anchors and their positive samples
    anchor_pos_distances = torch.gather(distance_matrix, 1, positive_indices)

    return anchor_pos_distances.squeeze(1), positive_indices.squeeze(1)


def random_negative_sampling(
    distance_matrix: torch.Tensor, mask_neg: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given a pairwise distance matrix of all the samples and a mask that indicates the
    possible indices for sampling negatives, randomly sample a negative sample for each
    anchor. All samples are treated as anchor points.

    Parameters:
    -----------
    distance_matrix: torch.Tensor
        2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
        x[i] and y[j], i.e. the pairwise distance matrix.
    mask_neg: torch.Tensor
        See create_pos_neg_masks() for details.

    Returns:
    --------
    anchor_neg_distances: torch.Tensor
        1D tensor of shape (n,), distances between the anchors and their chosen
        negative samples.
    negative_indices: torch.Tensor
        1D tensor of shape (n,) where negative_indices[i] is the index of the negative
        sample for the i-th anchor point.
    """

    # Get the indices of the negative samples for each anchor point
    negative_indices = torch.multinomial(mask_neg, 1)

    # Get the distances between the anchors and their negative samples
    anchor_neg_distances = torch.gather(distance_matrix, 1, negative_indices)

    return anchor_neg_distances.squeeze(1), negative_indices.squeeze(1)


############################### Hard Mining ###############################


def hard_positive_mining(
    distance_matrix: torch.Tensor, mask_pos: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given a pairwise distance matrix of all the anchors and a mask that indicates the
    possible indices for sampling positives, mine the hardest positive sample for each
    anchor. All samples are treated as anchor points.

    Parameters:
    -----------
    distance_matrix: torch.Tensor
        2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
        x[i] and y[j], i.e. the pairwise distance matrix.
    mask_pos: torch.Tensor
        See create_pos_neg_masks() for details.

    Returns:
    --------
    anchor_pos_distances: torch.Tensor
        Distances between the anchors and their chosen positive samples.
    positive_indices: torch.Tensor
        1D tensor of shape (n,) where positive_indices[i] is the index of the hardest
        positive sample for the i-th anchor point.
    """

    # Select the hardest positive for each anchor
    anchor_pos_distances, positive_indices = torch.max(distance_matrix * mask_pos, 1)

    return anchor_pos_distances, positive_indices


def hard_negative_mining(
    distance_matrix: torch.Tensor, mask_neg: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given a pairwise distance matrix of all the anchors and a mask that indicates the
    possible indices for sampling negatives, mine the hardest negative sample for each
    anchor. All samples are treated as anchor points.

    Parameters:
    -----------
    distance_matrix: torch.Tensor
        2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
        x[i] and y[j], i.e. the pairwise distance matrix.
    mask_neg: torch.Tensor
        See create_pos_neg_masks() for details.

    Returns:
    --------
    anchor_neg_distances: torch.Tensor
        Distances between the anchors and their chosen negative samples.
    negative_indices: torch.Tensor
        1D tensor of shape (n,) where negative_indices[i] is the index of the hardest
        negative sample for the i-th anchor point.
    """

    # Modify the distance matrix to only consider the negative samples
    mask_neg = torch.where(
        mask_neg == 0,
        float("inf"),
        0.0,
    )

    # Get the indices of the hardest negative samples for each anchor point
    anchor_neg_distances, negative_indices = torch.min(distance_matrix + mask_neg, 1)

    return anchor_neg_distances, negative_indices


################################# Easy Mining #################################


def easy_positive_mining(
    distance_matrix: torch.Tensor, mask_pos: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given a pairwise distance matrix of all the anchors and a mask that indicates the
    possible indices for sampling positives, mine the easiest positive sample for each
    anchor. All samples are treated as anchor points.

    Parameters:
    -----------
    distance_matrix: torch.Tensor
        2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
        x[i] and y[j], i.e. the pairwise distance matrix.
    mask_pos: torch.Tensor
        See create_pos_neg_masks() for details.

    Returns:
    --------
    anchor_pos_distances: torch.Tensor
        Distances between the anchors and their chosen positive samples.
    positive_indices: torch.Tensor
        1D tensor of shape (n,) where positive_indices[i] is the index of the hardest
        positive sample for the i-th anchor point.
    """

    mask_pos = torch.where(mask_pos == 0, torch.tensor(float("inf")), torch.tensor(0.0))

    # Select the easiest positive for each anchor
    anchor_pos_distances, positive_indices = torch.min(distance_matrix + mask_pos, 1)

    return anchor_pos_distances, positive_indices
