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
        1D tensor of shape (n,) containing the distances between the anchors and their
        chosen positive samples.
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
        1D tensor of shape (n,) containing the distances between the anchors and their
        chosen negative samples.
    negative_indices: torch.Tensor
        1D tensor of shape (n,) where negative_indices[i] is the index of the hardest
        negative sample for the i-th anchor point.
    """

    # make sure same data type as distance_matrix
    inf = torch.tensor(
        float("inf"), device=distance_matrix.device, dtype=distance_matrix.dtype
    )
    zero = torch.tensor(0.0, device=distance_matrix.device, dtype=distance_matrix.dtype)

    # Modify the distance matrix to only consider the negative samples
    mask_neg = torch.where(mask_neg == 0, inf, zero)

    # Get the indices of the hardest negative samples for each anchor point
    anchor_neg_distances, negative_indices = torch.min(distance_matrix + mask_neg, 1)

    return anchor_neg_distances, negative_indices


############################### Semi-hard Mining ###############################


# TODO: check if this is correct
def semi_hard_negative_mining(
    distance_matrix: torch.Tensor,
    dist_AP: torch.Tensor,
    mask_neg: torch.Tensor,
    margin: float,
    mode: str = "hard",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given a pairwise distance matrix of all the anchors and a mask that indicates the
    possible indices for sampling negatives, mine the semi-hard negative sample for each
    anchor. All samples are treated as anchor points. If there are no possible semi-hard
    negatives, sample randomly.

    Parameters:
    -----------
    distance_matrix: torch.Tensor
        2D tensor of shape (n, n) where distance_matrix[i, j] is the distance between
        x[i] and y[j], i.e. the pairwise distance matrix.
    dist_AP: torch.Tensor
        1D tensor of shape (n,) where dist_AP[i] is the distance between the i-th anchor
        and its positive sample.
    mask_neg: torch.Tensor
        See create_pos_neg_masks() for details.
    margin: float
        Margin for the triplet loss.
    mode: str
        Either "hard" or "random". If "hard", the hardest negative sample from the region
        is selected for each anchor. If "random", a random negative sample is selected
        from the region.

    Returns:
    --------
    anchor_neg_distances: torch.Tensor
        1D tensor of shape (n,) containing the distances between the anchors and their
        chosen negative samples.
    negative_indices: torch.Tensor
        1D tensor of shape (n,) where negative_indices[i] is the index of the semi-hard
        negative sample for the i-th anchor point.
    """

    raise NotImplementedError("Having difficulty in implementing it.")

    assert mode in {"hard", "random"}, "mode must be either 'hard' or 'random'"
    assert margin > 0, "margin must be greater than 0"
    assert (
        distance_matrix.shape[0] == dist_AP.shape[0]
    ), "distance_matrix and dist_AP must have the same length"
    assert dist_AP.ndim == 1, "dist_AP must be a 1D tensor"

    # Get the region for semi-hard negatives
    mask_semi_hard_neg = (
        (dist_AP.unsqueeze(1) < distance_matrix)
        & (distance_matrix < (dist_AP.unsqueeze(1) + margin))
        & mask_neg.bool()
    ).float()

    # Initialize the tensors to store the distances and indices of the semi-hard negatives
    device = distance_matrix.device
    n = distance_matrix.shape[0]
    anchor_neg_distances = torch.zeros(n, dtype=torch.float32, device=device)
    negative_indices = torch.zeros(n, dtype=torch.int32, device=device)

    # Search for a semi-hard negative for each anchor, positive pair
    for i in range(n):
        dist = distance_matrix[i].unsqueeze(0)
        mask = mask_semi_hard_neg[i].unsqueeze(0)
        # check if the hollow-sphere is empty
        if mask.any():  # there is at least one semi-hard negative
            if mode == "hard":  # choose the hardest example in the hollow-sphere
                negative_indices[i], anchor_neg_distances[i] = hard_negative_mining(
                    dist, mask
                )
            else:  # choose a random example in the hollow-sphere
                negative_indices[i], anchor_neg_distances[i] = random_negative_sampling(
                    dist, mask
                )
        else:  # there are no semi-hard negatives
            if mode == "hard":  # resort to hard negatives
                negative_indices[i], anchor_neg_distances[i] = hard_negative_mining(
                    dist, mask_neg[i].unsqueeze(0)
                )
            else:  # resort to random negatives
                negative_indices[i], anchor_neg_distances[i] = random_negative_sampling(
                    dist, mask_neg[i].unsqueeze(0)
                )
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
        1D tensor of shape (n,) containing the distances between the anchors and their
        chosen positive samples.
    positive_indices: torch.Tensor
        1D tensor of shape (n,) where positive_indices[i] is the index of the hardest
        positive sample for the i-th anchor point.
    """

    # make sure same data type as distance_matrix
    inf = torch.tensor(
        float("inf"), device=distance_matrix.device, dtype=distance_matrix.dtype
    )
    zero = torch.tensor(0.0, device=distance_matrix.device, dtype=distance_matrix.dtype)

    # Modify the distance matrix to only consider the positive samples
    mask_pos = torch.where(mask_pos == 0, inf, zero)

    # Select the easiest positive for each anchor
    anchor_pos_distances, positive_indices = torch.min(distance_matrix + mask_pos, 1)

    return anchor_pos_distances, positive_indices
