from functools import partial
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.jax_util import filter_bmap
from jaxtyping import Array, Float

from ...._custom_types import ConstantT, LossFn, ParticleStackInfo, PerParticleT
from ....simulator._dilated_mask import DilatedMask
from ..._pose_search import HierarchicalSO3GridSearch


# `in_axes` for mapping a `ParticleStackInfo` over its leading particle axis.
# `cryospax` returns one `image_config` shared by the whole stack (its `pixel_size`
# and `voltage_in_kilovolts` are scalars), while `pose`, `transfer_theory` and
# `images` carry a per-particle leading axis. So `image_config` must not be mapped.
_PARTICLE_STACK_IN_AXES = {
    "parameters": {
        "image_config": None,
        "pose": eqx.if_array(0),
        "transfer_theory": eqx.if_array(0),
    },
    "images": eqx.if_array(0),
}


def _split_shared_image_config(
    relion_stack: ParticleStackInfo,
) -> tuple[dict, Any]:
    """Split a particle stack into its per-particle part and its shared
    `image_config`, so that only the former is batched over.
    """
    if relion_stack["parameters"] is None:
        raise ValueError("relion_stack must have non None 'parameters' field.")
    parameters = dict(relion_stack["parameters"])
    image_config = parameters.pop("image_config")
    return {"parameters": parameters, "images": relion_stack["images"]}, image_config


def _join_shared_image_config(per_particle_stack: dict, image_config: Any) -> dict:
    """Inverse of `_split_shared_image_config`."""
    return {
        "parameters": {**per_particle_stack["parameters"], "image_config": image_config},
        "images": per_particle_stack["images"],
    }


@partial(
    eqx.filter_vmap,
    in_axes=(
        0,
        None,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),
    out_axes=0,
)
@partial(
    eqx.filter_vmap,
    in_axes=(
        None,
        _PARTICLE_STACK_IN_AXES,
        None,
        None,
        None,
        None,
        None,
        None,
        eqx.if_array(0),
        None,
        None,
        None,
        None,
        None,
        None,
    ),
    out_axes=0,
)
def _compute_likelihood_matrix(
    ensemble_structures: Float[Array, " n_atoms 3"],
    relion_stack: ParticleStackInfo,
    amplitudes: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
    image_to_structure_log_likelihood_fn: LossFn,
    dilated_mask: Optional[DilatedMask],
    pose_grid_search: Optional[HierarchicalSO3GridSearch],
    constant_args: ConstantT,
    per_particle_args: PerParticleT,
    whitening_filter: Optional[Float[Array, "image_size image_size"]],
    whiten_experimental_images: bool,
    whiten_calculated_images: bool,
    normalize_experimental_images: bool,
    lowpass_filter: Optional[Float[Array, "image_size image_size"]],
    highpass_filter: Optional[Float[Array, "image_size image_size"]],
) -> Float[Array, "n_images n_structures"]:
    """
    Compute the likelihood matrix for a set of structures and a Relion stack.
    The likelihood is computed for each structure and each image in the stack.

    **Arguments:**
    - `ensemble_structures`: The structures of the ensemble. This is a 3D array
        with shape (n_structures, n_atoms, 3).
    - `relion_stack`: A cryojax  Dict` object.
    - `amplitudes`: The amplitudes for the GMM atomic volume representation.
    - `variances`: The variances for the GMM atomic volume representation.
    - `image_to_structure_log_likelihood_fn`: The function to compute the likelihood
        between the computed image and the observed image.
    - `per_particle_args`: The arguments to pass to the likelihood function.
    **Returns:**
    - The likelihood matrix of the ensemble. This is a 2D array
        such that the n, m element is p(y_n | x_m), where y_n is the n-th image
        and x_m is the m-th structure (atomic model).

    Note:
    If add arguments then need to update in_axes of the vmap decorators.
    """

    return image_to_structure_log_likelihood_fn(
        ensemble_structures,
        relion_stack,
        amplitudes,
        variances,
        dilated_mask,
        pose_grid_search,
        constant_args,
        per_particle_args,
        whitening_filter,
        whiten_experimental_images,
        whiten_calculated_images,
        normalize_experimental_images,
        lowpass_filter,
        highpass_filter,
    )


def _compute_likelihood_matrix_batched(
    ensemble_structures: Float[Array, "n_structures n_atoms 3"],
    relion_stack: ParticleStackInfo,
    amplitudes: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
    image_to_structure_log_likelihood_fn: LossFn,
    dilated_mask: Optional[DilatedMask],
    pose_grid_search: Optional[HierarchicalSO3GridSearch],
    constant_args: ConstantT,
    per_particle_args: PerParticleT,
    batch_size_structures: int,
    batch_size_images: int,
    whitening_filter: Optional[Float[Array, "image_size image_size"]],
    whiten_experimental_images: bool,
    whiten_calculated_images: bool,
    normalize_experimental_images: bool,
    lowpass_filter: Optional[Float[Array, "image_size image_size"]],
    highpass_filter: Optional[Float[Array, "image_size image_size"]],
) -> Float[Array, "n_images n_structures"]:
    map_over_images = eqx.filter_vmap(
        image_to_structure_log_likelihood_fn,
        in_axes=(
            None,
            _PARTICLE_STACK_IN_AXES,
            None,
            None,
            None,
            None,
            None,
            eqx.if_array(0),
            None,
            None,
            None,
            None,
            None,
            None,
        ),  # these sync up with the function signature of map_over_images
        out_axes=0,
    )

    # `filter_bmap` slices every array leaf of `xs` along axis 0, so the shared
    # `image_config` is held out of the batched pytree and re-attached inside.
    per_particle_stack, shared_image_config = _split_shared_image_config(relion_stack)

    map_over_structures = eqx.filter_vmap(
        lambda w, rs, ppa, amp, var: filter_bmap(
            lambda x: map_over_images(  # see in_axes above
                w,
                _join_shared_image_config(x[0], shared_image_config),
                amp,
                var,
                dilated_mask,
                pose_grid_search,
                constant_args,
                x[1],
                whitening_filter,
                whiten_experimental_images,
                whiten_calculated_images,
                normalize_experimental_images,
                lowpass_filter,
                highpass_filter,
            ),
            (rs, ppa),
            batch_size=batch_size_images,
        ),
        in_axes=(0, None, None, 0, 0),
        out_axes=0,
    )

    return filter_bmap(
        lambda x: map_over_structures(
            x[0],
            per_particle_stack,
            per_particle_args,
            x[1],
            x[2],
        ),
        (ensemble_structures, amplitudes, variances),
        batch_size=batch_size_structures,
    )


@eqx.filter_jit
def compute_likelihood_matrix(
    ensemble_structures: Float[Array, "n_structures n_atoms 3"],
    relion_stack: ParticleStackInfo,
    amplitudes: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
    image_to_structure_log_likelihood_fn: LossFn,
    dilated_mask: Optional[DilatedMask] = None,
    pose_grid_search: Optional[HierarchicalSO3GridSearch] = None,
    *,
    constant_args: ConstantT,
    per_particle_args: PerParticleT,
    batch_size_structures: Optional[int] = None,
    batch_size_images: Optional[int] = None,
    whitening_filter: Optional[Float[Array, "image_size image_size"]] = None,
    whiten_experimental_images: bool = False,
    whiten_calculated_images: bool = False,
    normalize_experimental_images: bool = False,
    lowpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
    highpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
) -> Float[Array, "n_images n_structures"]:
    """
    Compute the likelihood matrix for a set of structures and a Relion stack.
    The likelihood is computed for each structure and each image in the stack.

    **Arguments:**
    - `ensemble_structures`: The structures of the ensemble. This is a 3D array
        with shape (n_structures, n_atoms, 3).
    - `relion_stack`: A cryojax `ParticleStack` object.
    - `amplitudes`: The amplitudes for the GMM atomic volume representation.
    - `variances`: The variances for the GMM atomic volume representation.
    - `image_to_structure_log_likelihood_fn`: The function to compute the likelihood
        between the computed image and the observed image.
    - `per_particle_args`: The arguments to pass to the likelihood function.
    **Returns:**
    - The likelihood matrix of the ensemble. This is a 2D array
        such that the n, m element is p(y_n | x_m), where y_n is the n-th image
        and x_m is the m-th structure (atomic model).
    """
    if batch_size_structures is None and batch_size_images is None:
        return _compute_likelihood_matrix(
            ensemble_structures,
            relion_stack,
            amplitudes,
            variances,
            image_to_structure_log_likelihood_fn,
            dilated_mask,
            pose_grid_search,
            constant_args,
            per_particle_args,
            whitening_filter,
            whiten_experimental_images,
            whiten_calculated_images,
            normalize_experimental_images,
            lowpass_filter,
            highpass_filter,
        ).T
    else:
        batch_size_structures = (
            ensemble_structures.shape[0]
            if batch_size_structures is None
            else batch_size_structures
        )
        batch_size_images = (
            relion_stack["images"].shape[0]
            if batch_size_images is None
            else batch_size_images
        )
        return _compute_likelihood_matrix_batched(
            ensemble_structures,
            relion_stack,
            amplitudes,
            variances,
            image_to_structure_log_likelihood_fn,
            dilated_mask,
            pose_grid_search,
            constant_args,
            per_particle_args,
            batch_size_structures,
            batch_size_images,
            whitening_filter,
            whiten_experimental_images,
            whiten_calculated_images,
            normalize_experimental_images,
            lowpass_filter,
            highpass_filter,
        ).T


@eqx.filter_jit
def compute_neg_log_likelihood_from_weights(
    weights: Float[Array, " n_structures"],
    likelihood_matrix: Float[Array, "n_images n_structures"],
) -> Float:
    """
    Compute the negative log likelihood from the weights and a pre-computed likelihood
    matrix. The likelihood is averaged to avoid numerical issues and dependence on the
    number of images.

    This function is used for optimizing the weights of the ensemble with fixed structures.

    Args:
        weights: The weights of the ensemble.
        likelihood_matrix: The likelihood matrix of the ensemble. This is a 2D array
        such that the n, m element is p(y_n | x_m), where y_n is the n-th image
        and x_m is the m-th structure (atomic model).
    Returns:
        The negative log likelihood of the ensemble.
    """
    log_lklhood = jax.scipy.special.logsumexp(
        a=likelihood_matrix, b=weights[None, :], axis=1
    )
    return -jnp.mean(log_lklhood)


@eqx.filter_jit
def compute_neg_log_likelihood(
    structures: Float[Array, "n_structures n_atoms 3"],
    weights: Float[Array, " n_structures"],
    relion_stack: ParticleStackInfo,
    amplitudes: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
    image_to_structure_log_likelihood_fn: LossFn,
    dilated_mask: Optional[DilatedMask] = None,
    pose_grid_search: Optional[HierarchicalSO3GridSearch] = None,
    *,
    constant_args: ConstantT,
    per_particle_args: PerParticleT,
    batch_size_structures: Optional[int] = None,
    batch_size_images: Optional[int] = None,
    whitening_filter: Optional[Float[Array, "image_size image_size"]],
    whiten_experimental_images: bool,
    whiten_calculated_images: bool,
    normalize_experimental_images: bool,
    lowpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
    highpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
) -> Float:
    """
    Compute the negative log likelihood from the structures and weights. The likelihood is
    averaged to avoid numerical issues and dependence on the number of images.

    This function is used for optimizing the structures of the ensemble with fixed weights.

    Args:
        structures: The structures of the ensemble. This is a 3D array
            with shape (n_structures, n_atoms, 3).
        weights: The weights of the ensemble.
        relion_stack: A cryojax  Dict` object.
        amplitudes: The amplitudes for the GMM atomic volume representation.
        variances: The variances for the GMM atomic volume representation.
        image_to_structure_log_likelihood_fn: The function to compute the likelihood
            between the computed image and the observed image.
        per_particle_args: The arguments to pass to the likelihood function.
    Returns:
        The negative log likelihood of the ensemble.
    """

    lklhood_matrix = compute_likelihood_matrix(
        structures,
        relion_stack,
        amplitudes,
        variances,
        image_to_structure_log_likelihood_fn,
        dilated_mask,
        pose_grid_search,
        constant_args=constant_args,
        per_particle_args=per_particle_args,
        batch_size_structures=batch_size_structures,
        batch_size_images=batch_size_images,
        whitening_filter=whitening_filter,
        whiten_experimental_images=whiten_experimental_images,
        whiten_calculated_images=whiten_calculated_images,
        normalize_experimental_images=normalize_experimental_images,
        lowpass_filter=lowpass_filter,
        highpass_filter=highpass_filter,
    )
    return compute_neg_log_likelihood_from_weights(weights, lklhood_matrix)
