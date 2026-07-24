from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from ...._custom_types import ParticleStackInfo, PerParticleT
from ....simulator._dilated_mask import DilatedMask
from ....utils import apply_filter as _apply_filter
from ..._pose_search import HierarchicalSO3GridSearch
from .common_functions import compute_optimal_scale_and_offset
from .make_model_utils import make_image_model_from_gmm
from .normalize_image import _normalize_image


def likelihood_isotropic_gaussian(
    structure: Float[Array, "n_atoms 3"],
    relion_stack: ParticleStackInfo,
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    dilated_mask: Optional[DilatedMask],
    pose_grid_search: Optional["HierarchicalSO3GridSearch"],
    constant_args: float,
    per_particle_args: Float[Array, ""],
    whitening_filter: Optional[Float[Array, "image_size image_size"]],
    whiten_experimental_images: bool,
    whiten_calculated_images: bool,
    normalize_experimental_images: bool,
    lowpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
    highpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
) -> Float:
    """
    Compute the likelihood of a structure given a Relion stack using an isotropic Gaussian
    likelihood function.

    **Arguments:**
    - `structure`: A `structure` that is, a point cloud representing an atomic model.
    - `relion_stack`: A cryojax `ParticleStack` object.
    - `amplitudes`: The amplitudes for the GMM atomic volume representation.
    - `variances`: The variances for the GMM atomic volume representation.
    - `dilated_mask`: An optional dilated mask to apply to the computed image.
    - `constant_args`: For this particular function the constant argument
        is the sign of the observed image. For typical Relion stacks this is -1.0.
        For data generated with cryoJAX this is 1.0.
    - `per_particle_args`: The noise variance for the likelihood function.

    **Returns:**
    - The log likelihood of the structure given the Relion stack.

    """
    if relion_stack["parameters"] is None:
        raise ValueError("relion_stack must have non None 'parameters' field.")

    noise_variance = per_particle_args

    observed_image = _apply_filter(
        jnp.asarray(relion_stack["images"]),
        whitening_filter if whiten_experimental_images else None,
    )
    if normalize_experimental_images:
        observed_image = _normalize_image(
            observed_image,
            relion_stack["parameters"]["image_config"],
            relion_stack["parameters"]["image_config"].shape,
        )
    observed_image = _apply_filter(observed_image, lowpass_filter)
    observed_image = _apply_filter(observed_image, highpass_filter)

    image_model = make_image_model_from_gmm(
        structure,
        observed_image,
        relion_stack,
        amplitudes,
        variances,
        pose_grid_search,
        whitening_filter,
        whiten_calculated_images,
        lowpass_filter,
        highpass_filter,
    )
    computed_image = _apply_filter(
        image_model.simulate(),
        whitening_filter if whiten_calculated_images else None,
    )
    computed_image = _apply_filter(computed_image, lowpass_filter)
    computed_image = _apply_filter(computed_image, highpass_filter)

    if dilated_mask is not None:
        mask2d = dilated_mask.project(relion_stack["parameters"]["pose"])
    else:
        mask2d = jnp.ones_like(computed_image)

    computed_image = computed_image * mask2d
    observed_image = constant_args * observed_image * mask2d

    scale, offset = compute_optimal_scale_and_offset(computed_image, observed_image)

    return -jnp.sum((scale * computed_image - observed_image + offset) ** 2) / (
        2 * noise_variance
    )


def likelihood_isotropic_gaussian_marginalized(
    structure: Float[Array, "n_atoms 3"],
    relion_stack: ParticleStackInfo,
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    dilated_mask: Optional[DilatedMask],
    pose_grid_search: Optional["HierarchicalSO3GridSearch"],
    constant_args: float,
    per_particle_args: PerParticleT,
    whitening_filter: Optional[Float[Array, "image_size image_size"]],
    whiten_experimental_images: bool,
    whiten_calculated_images: bool,
    normalize_experimental_images: bool,
    lowpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
    highpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
) -> Float:
    """
    Compute the marginalized likelihood of a structure given a Relion stack using an
    isotropic Gaussian likelihood function where the variance has been marginalized.
    This is useful when the variance is not known or is not fixed.

    **Arguments:**
    - `structure`: A `structure` that is, a point cloud representing an atomic model.
    - `relion_stack`: A cryojax `ParticleStack` object.
    - `amplitudes`: The amplitudes for the GMM atomic volume representation.
    - `variances`: The variances for the GMM atomic volume representation.
    - `dilated_mask`: An optional dilated mask to apply to the computed image.
    - `constant_args`: For this particular function the constant argument
        is the sign of the observed image. For typical Relion stacks this is -1.0.
        For data generated with cryoJAX this is 1.0.
    - `per_particle_args`: not used in this function.
    """
    if relion_stack["parameters"] is None:
        raise ValueError("relion_stack must have non None 'parameters' field.")

    observed_image = _apply_filter(
        jnp.asarray(relion_stack["images"]),
        whitening_filter if whiten_experimental_images else None,
    )
    if normalize_experimental_images:
        observed_image = _normalize_image(
            observed_image,
            relion_stack["parameters"]["image_config"],
            relion_stack["parameters"]["image_config"].shape,
        )
    observed_image = _apply_filter(observed_image, lowpass_filter)
    observed_image = _apply_filter(observed_image, highpass_filter)

    image_model = make_image_model_from_gmm(
        structure,
        observed_image,
        relion_stack,
        amplitudes,
        variances,
        pose_grid_search,
        whitening_filter,
        whiten_calculated_images,
        lowpass_filter,
        highpass_filter,
    )
    computed_image = _apply_filter(
        image_model.simulate(),
        whitening_filter if whiten_calculated_images else None,
    )
    computed_image = _apply_filter(computed_image, lowpass_filter)
    computed_image = _apply_filter(computed_image, highpass_filter)

    if dilated_mask is not None:
        mask2d = dilated_mask.project(relion_stack["parameters"]["pose"])
    else:
        mask2d = jnp.ones_like(computed_image)

    computed_image = computed_image * mask2d
    observed_image = constant_args * observed_image * mask2d

    scale, offset = compute_optimal_scale_and_offset(computed_image, observed_image)
    n_pixels = computed_image.size
    loss = -n_pixels * jnp.log(
        jnp.linalg.norm(scale * computed_image - observed_image + offset)
    )
    return loss
