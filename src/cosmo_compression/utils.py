import numpy as np
import torch

def compute_power_spectrum(
    field: torch.Tensor | np.ndarray,
    box_size: float = 25.0,
    raw_normalization: bool = False,
    aggregate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 2D power spectrum using Pylians, truncated at the Nyquist frequency.

    Args:
        field: Real-space field, typically of shape [H, W] or [1, H, W] or batched [B, ..., H, W].
               If a batch is provided, returns the mean P(k) across the batch unless aggregate=False.
        box_size: Physical side-length of the box in Mpc/h.
        raw_normalization: If True, uses shifted normalization for raw data.
        aggregate: If True, average P(k) over the batch. If False, return [B, n_kbins] array.

    Returns:
        k: 1-D array of k bin centres.
        pk: Mean P(k) if aggregate=True, else stacked P(k) array of shape [B, n_kbins].
    """
    import Pk_library as PKL

    if isinstance(field, torch.Tensor):
        field = field.detach().cpu().numpy()

    # Flatten extra dimensions so we have [B, H, W]
    if field.ndim == 2:
        field = field[np.newaxis, ...]
    elif field.ndim > 3:
        # e.g., B, 1, H, W -> B, H, W
        field = field.reshape(-1, field.shape[-2], field.shape[-1])

    B, H, W = field.shape
    assert H == W, "Only square fields supported"

    pks = []
    k_out = None

    for b in range(B):
        single_field = field[b]

        if raw_normalization:
            shifted = single_field - single_field.min() + 1e-6
            delta = shifted / np.mean(shifted) - 1.0
        else:
            mean_val = np.mean(single_field)
            if mean_val == 0:
                delta = single_field
            else:
                delta = single_field / mean_val - 1.0

        pk_instance = PKL.Pk_plane(delta, box_size, "None", 1, verbose=False)
        k, p = pk_instance.k, pk_instance.Pk

        # Apply Nyquist cutoff
        nyquist = np.pi * H / box_size
        mask = k <= nyquist
        k, p = k[mask], p[mask]

        if k_out is None:
            k_out = k
        pks.append(p)

    if aggregate:
        return k_out, np.mean(pks, axis=0)
    else:
        return k_out, np.array(pks)
