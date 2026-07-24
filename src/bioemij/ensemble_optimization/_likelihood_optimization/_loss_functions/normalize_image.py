import jax.numpy as jnp

from ....utils._whitening_filter import _get_corner_mask_nopad


def _normalize_image(image, image_config, original_shape):
    corner_mask = _get_corner_mask_nopad(image_config, original_shape)
    where = jnp.where(corner_mask == 1.0, True, False)
    mean = jnp.mean(image, axis=(0, 1), keepdims=True, where=where)
    std = jnp.std(image, axis=(0, 1), keepdims=True, ddof=1, where=where)
    scale = jnp.where(std > 0, 1.0 / std, 0.0)
    return (image - mean) * scale


def _normalize_image_stack(image_stack, image_config, original_shape):
    corner_mask = _get_corner_mask_nopad(image_config, original_shape)
    where = jnp.where(corner_mask == 1.0, True, False)[None, ...]
    means = jnp.mean(image_stack, axis=(1, 2), keepdims=True, where=where)
    stds = jnp.std(image_stack, axis=(1, 2), keepdims=True, ddof=1, where=where)
    scale = jnp.where(stds > 0, 1.0 / stds, 0.0)
    return (image_stack - means) * scale
