from typing import Optional

import cryojax.ndimage as cxim
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cryospax import RelionParticleDataset
from cryojax.jax_util import filter_bmap
from cryojax.simulator import BasicImageConfig
from jaxtyping import Array, Float
from tqdm import tqdm


def _get_corner_mask_nopad(
    padded_image_config: BasicImageConfig, original_shape: tuple[int, int]
):
    coordinate_grid = padded_image_config.get_coordinate_grid(physical=False)
    corner_mask = cxim.CircularCosineMask(
        coordinate_grid,
        radius=original_shape[0] // 2,
        rolloff_width=1.0,
    ).get()
    pad_mask = cxim.Rectangular2DCosineMask(
        coordinate_grid,
        x_width=original_shape[0],
        y_width=original_shape[1],
        rolloff_width=1.0,
    ).get()
    return (1 - corner_mask) * pad_mask


def apply_filter(image, filter_array):
    """Apply a Fourier-domain filter to a real-space image."""
    if filter_array is None:
        return image
    return jnp.fft.ifftn(jnp.fft.fftn(image) * filter_array).real


def compute_whitening_filter(
    relion_dataset: RelionParticleDataset,
    n_images_in_parallel: int,
    target_box_size: Optional[int],
    padding_factor: float,
):
    image_config = relion_dataset.parameter_file[0]["image_config"]
    if target_box_size is not None:
        image_config = _get_downsampled_image_config(image_config, target_box_size)

    original_shape = image_config.shape
    whitening_mask = _get_corner_mask_nopad(image_config, original_shape)

    n_images = len(relion_dataset)
    n_mask_pixels = jnp.sum(whitening_mask)

    # compute for one image to check the shape
    mean_est = 0.0
    noise_psd_est = jnp.zeros(image_config.shape)
    # batch_pad_to_shape = jax.vmap(cxim.pad_to_shape, in_axes=(0, None))

    for i in tqdm(
        range(0, n_images, n_images_in_parallel),
        desc="Computing whitening filter",
        unit="batch",
    ):
        batch_indices = np.arange(i, min(i + n_images_in_parallel, n_images))
        stack = relion_dataset[batch_indices]
        images = jnp.asarray(stack["images"])
        if target_box_size is not None:
            images, _ = _downsample_images(
                images,
                image_config,
                target_box_size=target_box_size,
                n_images_in_parallel=n_images_in_parallel,
            )
        batch_sum, batch_psd = _compute_batch_stats(images, whitening_mask)
        mean_est += batch_sum
        noise_psd_est += batch_psd

    mean_est /= n_images * n_mask_pixels
    noise_psd_est /= n_images * n_mask_pixels
    noise_psd_est = noise_psd_est.at[0, 0].add(-(mean_est**2))
    whitening_filter = jnp.where(
        jnp.isclose(noise_psd_est, 0.0),
        0.0,
        jax.lax.rsqrt(noise_psd_est),
    )
    padded_shape = (
        int(image_config.shape[0] * padding_factor),
        int(image_config.shape[1] * padding_factor),
    )
    whitening_filter = jnp.fft.fftn(
        cxim.pad_to_shape(jnp.fft.ifftn(whitening_filter), padded_shape)
    ).real
    whitening_filter = whitening_filter.at[0, 0].set(0.0)

    return whitening_filter


@jax.jit
def _compute_batch_stats(
    images: Float[Array, "n y x"],
    whitening_mask: Float[Array, "y x"],
) -> tuple[Array, Array]:
    images = images * whitening_mask[None, ...]
    batch_sum = images.sum()
    batch_psd = jnp.sum(jnp.abs(jnp.fft.fftn(images, axes=(-2, -1))) ** 2, axis=0)
    return batch_sum, batch_psd


@eqx.filter_vmap(in_axes=(0, None))
def _downsample_image(
    image: Float[Array, "y_dim x_dim"],
    ds_shape: tuple[int, int],
):
    image_ds = cxim.fourier_crop_to_shape(
        image, shape=ds_shape, preserve_mean=True, outputs_real_space=True
    )
    return image_ds


def _get_downsampled_image_config(
    image_config: BasicImageConfig, target_box_size: int
) -> BasicImageConfig:
    downsample_factor = image_config.shape[0] / target_box_size
    ds_pixel_size = image_config.pixel_size * downsample_factor
    ds_image_config = BasicImageConfig(
        shape=(target_box_size, target_box_size),
        pixel_size=ds_pixel_size,
        voltage_in_kilovolts=image_config.voltage_in_kilovolts,
        precompute_mode="all",
    )
    return ds_image_config


def _downsample_images(
    images: Float[Array, "n_images y_dim x_dim"],
    image_config: BasicImageConfig,
    target_box_size: int,
    *,
    n_images_in_parallel: int,
):
    new_shape = (target_box_size, target_box_size)
    ds_image_config = _get_downsampled_image_config(image_config, target_box_size)

    return filter_bmap(
        lambda x: _downsample_image(x, new_shape),
        xs=images,
        batch_size=n_images_in_parallel,
    ), ds_image_config
