import torch
import numpy as np

def center_of_mass_batch(positions, masses=None):
    """Compute the center of mass for a batch of positions."""
    if masses is None:
        return positions.mean(dim=1, keepdim=True)  # (B, 1, 3)
    return (positions * masses[:, :, None]).sum(dim=1, keepdim=True) / masses.sum(dim=1, keepdim=True)

def kabsch_alignment_batch(A, B):
    """
    Batched Kabsch algorithm to find the optimal rotation that aligns B onto A.
    A: (N, 3) reference coordinates
    B: (B, N, 3) batch of target coordinates
    Returns: (B, 3, 3) optimal rotation matrices
    """
    # Compute covariance matrix H per batch
    H = torch.einsum("ni,bnj->bij", A, B)  # (B, 3, 3)

    # Singular Value Decomposition (SVD)
    U, _, Vt = torch.linalg.svd(H)  # U, Vt: (B, 3, 3)

    # Ensure right-handed coordinate system
    det_sign = torch.det(U @ Vt)  # (B,)
    Vt[det_sign < 0, -1, :] *= -1  # Flip last row of Vt when determinant is negative

    # Compute optimal rotation matrices
    R = U @ Vt  # (B, 3, 3)

    return R

def superimpose_B_onto_A(A, B, idx, masses=None):
    """
    Superimposes each system in B onto the reference system A.
    A: (N, 3) reference coordinates
    B: (B, N, 3) batch of target coordinates
    masses: Optional (B, N) mass array for weighted alignment
    Returns: (B, N, 3) aligned B coordinates
    """
    # Compute centers of mass
    A_com = A.mean(dim=0, keepdim=True)  # (1, 3)
    B_com = center_of_mass_batch(B, masses)  # (B, 1, 3)

    # Center the structures
    A_centered = A - A_com  # (N, 3)
    B_centered = B - B_com  # (B, N, 3)

    # Compute optimal rotation
    R = kabsch_alignment_batch(A_centered[idx], B_centered[:, idx, :])  # (B, 3, 3)

    # Apply rotation and translation
    B_aligned = torch.einsum("bij,bnj->bni", R, B_centered) + A_com  # (B, N, 3)

    return B_aligned



from optimal_transport import wasserstein
def compute_distribution_distances(
    pred, true
):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.

    """
    NAMES = [
        "1-Wasserstein",
        "2-Wasserstein",
    ]
    is_jagged = isinstance(true, list)
    pred_is_jagged = isinstance(pred, list)
    dists = []
    to_return = []
    names = []
    filtered_names = [name for name in NAMES if not is_jagged or not name.endswith("MMD")]
    ts = len(pred) if pred_is_jagged else pred.shape[1]
    for t in np.arange(ts):
        if pred_is_jagged:
            a = pred[t]
        else:
            a = pred[:, t, :]
        if is_jagged:
            b = true[t]
        else:
            b = true[:, t, :]


        w1 = wasserstein(a, b, power=1)
        w2 = wasserstein(a, b, power=2)

        dists.append((w1, w2))
        # For multipoint datasets add timepoint specific distances
        if ts > 1:
            names.extend([f"t{t+1}/{name}" for name in filtered_names])
            to_return.extend(dists[-1])

    to_return.extend(np.array(dists).mean(axis=0))
    names.extend(filtered_names)
    return names, to_return