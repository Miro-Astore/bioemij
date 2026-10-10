from typing import Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def compute_optimal_scale_and_offset(
    target_image: Float[Array, "n_pixels n_pixels"],
    ref_image: Float[Array, "n_pixels n_pixels"],
    where: Optional[Bool[Array, "n_pixels n_pixels"]] = None,
) -> Tuple[Float, Float]:
    """Least-squares scale and offset over the pixels in `where` (all if None)."""
    cc = jnp.mean(target_image**2, where=where)
    co = jnp.mean(ref_image * target_image, where=where)
    c = jnp.mean(target_image, where=where)
    o = jnp.mean(ref_image, where=where)

    scale = (co - c * o) / (cc - c**2)
    offset = o - scale * c

    return scale, offset
