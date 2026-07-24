from typing import Any, Optional
from typing_extensions import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ...._custom_types import ConstantT, LossFn
from ....simulator import DilatedMask
from ..._pose_search import HierarchicalSO3GridSearch
from .ensemble_losses import (
    compute_likelihood_matrix,
    compute_neg_log_likelihood,
    compute_neg_log_likelihood_from_weights,
)
from .euclidean_losses import (
    likelihood_isotropic_gaussian,
    likelihood_isotropic_gaussian_marginalized,
)
from .sliced_wasserstein import likelihood_sliced_wasserstein


class AbstractLikelihoodFn(eqx.Module, strict=True):
    variances: eqx.AbstractVar[Float[Array, "n_structures n_atoms n_gaussians_per_atom"]]
    amplitudes: eqx.AbstractVar[Float[Array, "n_structures n_atoms n_gaussians_per_atom"]]
    image_to_structure_log_likelihood_fn: eqx.AbstractVar[LossFn]
    loss_fn_constant_args: eqx.AbstractVar[ConstantT]
    dilated_mask: eqx.AbstractClassVar[Optional[DilatedMask]]
    pose_grid_search: eqx.AbstractVar[Optional[HierarchicalSO3GridSearch]]

    def __call__(
        self,
        structures: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
        weights: Float[Array, " n_structures"],
        relion_batch: Any,
        batch_size_structures: Optional[int] = None,
        batch_size_images: Optional[int] = None,
        lowpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
        highpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
    ) -> Float:
        raise NotImplementedError


class LikelihoodFn(AbstractLikelihoodFn, strict=True):
    variances: Float[Array, "n_structures n_atoms n_gaussians_per_atom"]
    amplitudes: Float[Array, "n_structures n_atoms n_gaussians_per_atom"]
    image_to_structure_log_likelihood_fn: LossFn
    loss_fn_constant_args: ConstantT
    dilated_mask: Optional[DilatedMask] = None
    pose_grid_search: Optional[HierarchicalSO3GridSearch] = None
    whitening_filter: Optional[Float[Array, "image_size image_size"]] = None
    whiten_experimental_images: bool = False
    whiten_calculated_images: bool = False
    normalize_experimental_images: bool = False

    def __init__(
        self,
        amplitudes: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
        variances: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
        image_to_structure_log_likelihood_fn: Literal[
            "iso_gaussian", "iso_gaussian_var_marg", "sliced_wasserstein"
        ]
        | LossFn,
        loss_fn_constant_args: Optional[ConstantT] = None,
        dilated_mask: Optional[DilatedMask] = None,
        pose_grid_search: Optional[HierarchicalSO3GridSearch] = None,
        whitening_filter: Optional[Float[Array, "image_size image_size"]] = None,
        whiten_experimental_images: bool = False,
        whiten_calculated_images: bool = False,
        normalize_experimental_images: bool = False,
    ):
        assert jnp.all(variances > 0.0), "Variances must be positive."
        assert jnp.all(amplitudes > 0.0), "Amplitudes must be positive."
        if whitening_filter is not None:
            assert whiten_experimental_images or whiten_calculated_images, (
                "Provided `whitening_filter`, but `whiten_experimental_images`"
                " and `whiten_calculated_images` are both False."
            )

        self.variances = variances
        self.amplitudes = amplitudes
        if image_to_structure_log_likelihood_fn == "iso_gaussian":
            self.image_to_structure_log_likelihood_fn = likelihood_isotropic_gaussian
            self.loss_fn_constant_args = (
                1.0 if loss_fn_constant_args is None else loss_fn_constant_args
            )
        elif image_to_structure_log_likelihood_fn == "iso_gaussian_var_marg":
            self.image_to_structure_log_likelihood_fn = (
                likelihood_isotropic_gaussian_marginalized
            )
            self.loss_fn_constant_args = (
                1.0 if loss_fn_constant_args is None else loss_fn_constant_args
            )
        elif image_to_structure_log_likelihood_fn == "sliced_wasserstein":
            self.image_to_structure_log_likelihood_fn = likelihood_sliced_wasserstein
            self.loss_fn_constant_args = (
                (18, 2) if loss_fn_constant_args is None else loss_fn_constant_args
            )
        else:
            assert callable(image_to_structure_log_likelihood_fn), (
                "If `image_to_structure_log_likelihood_fn` is not 'iso_gaussian' or "
                + "'iso_gaussian_var_marg', it must be a callable function."
            )
            self.image_to_structure_log_likelihood_fn = image_to_structure_log_likelihood_fn
            self.loss_fn_constant_args = loss_fn_constant_args

        self.dilated_mask = dilated_mask
        self.pose_grid_search = pose_grid_search
        self.whitening_filter = whitening_filter
        self.whiten_experimental_images = whiten_experimental_images
        self.whiten_calculated_images = whiten_calculated_images
        self.normalize_experimental_images = normalize_experimental_images

    def __call__(
        self,
        structures: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
        weights: Float[Array, " n_structures"],
        relion_batch: Any,
        batch_size_structures: Optional[int] = None,
        batch_size_images: Optional[int] = None,
        lowpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
        highpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
    ):
        return compute_neg_log_likelihood(
            structures,
            weights,
            relion_batch["particle_stack"],
            self.amplitudes,
            self.variances,
            self.image_to_structure_log_likelihood_fn,
            self.dilated_mask,
            self.pose_grid_search,
            constant_args=self.loss_fn_constant_args,
            per_particle_args=relion_batch["per_particle_args"],
            batch_size_structures=batch_size_structures,
            batch_size_images=batch_size_images,
            whitening_filter=self.whitening_filter,
            whiten_experimental_images=self.whiten_experimental_images,
            whiten_calculated_images=self.whiten_calculated_images,
            normalize_experimental_images=self.normalize_experimental_images,
            lowpass_filter=lowpass_filter,
            highpass_filter=highpass_filter,
        ), weights


class LikelihoodOptimalWeightsFn(AbstractLikelihoodFn, strict=True):
    variances: Float[Array, "n_structures n_atoms n_gaussians_per_atom"]
    amplitudes: Float[Array, "n_structures n_atoms n_gaussians_per_atom"]
    image_to_structure_log_likelihood_fn: LossFn
    loss_fn_constant_args: ConstantT
    dilated_mask: Optional[DilatedMask] = None
    pose_grid_search: Optional[HierarchicalSO3GridSearch] = None
    initialize_uniform_weights: bool = True
    n_steps_opt_weights: int = 500
    whitening_filter: Optional[Float[Array, "image_size image_size"]] = None
    whiten_experimental_images: bool = False
    whiten_calculated_images: bool = False
    normalize_experimental_images: bool = False

    def __init__(
        self,
        amplitudes: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
        variances: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
        image_to_structure_log_likelihood_fn: Literal[
            "iso_gaussian", "iso_gaussian_var_marg"
        ]
        | LossFn,
        loss_fn_constant_args: Optional[ConstantT] = None,
        dilated_mask: Optional[DilatedMask] = None,
        pose_grid_search: Optional[HierarchicalSO3GridSearch] = None,
        initialize_uniform_weights: bool = True,
        n_steps_opt_weights: int = 500,
        whitening_filter: Optional[Float[Array, "image_size image_size"]] = None,
        whiten_experimental_images: bool = False,
        whiten_calculated_images: bool = False,
        normalize_experimental_images: bool = False,
    ):
        assert jnp.all(variances > 0.0), "Variances must be positive."
        assert jnp.all(amplitudes > 0.0), "Amplitudes must be positive."
        if whitening_filter is not None:
            assert whiten_experimental_images or whiten_calculated_images, (
                "Provided `whitening_filter`, but `whiten_experimental_images`"
                " and `whiten_calculated_images` are both False."
            )

        self.variances = variances
        self.amplitudes = amplitudes
        if image_to_structure_log_likelihood_fn == "iso_gaussian":
            self.image_to_structure_log_likelihood_fn = likelihood_isotropic_gaussian
            self.loss_fn_constant_args = (
                1.0 if loss_fn_constant_args is None else loss_fn_constant_args
            )
        elif image_to_structure_log_likelihood_fn == "iso_gaussian_var_marg":
            self.image_to_structure_log_likelihood_fn = (
                likelihood_isotropic_gaussian_marginalized
            )
            self.loss_fn_constant_args = (
                1.0 if loss_fn_constant_args is None else loss_fn_constant_args
            )
        elif image_to_structure_log_likelihood_fn == "sliced_wasserstein":
            self.image_to_structure_log_likelihood_fn = likelihood_sliced_wasserstein
            self.loss_fn_constant_args = (
                (18, 2) if loss_fn_constant_args is None else loss_fn_constant_args
            )
        else:
            assert callable(image_to_structure_log_likelihood_fn), (
                "If `image_to_structure_log_likelihood_fn` is not 'iso_gaussian' or "
                + "'iso_gaussian_var_marg', it must be a callable function."
            )
            self.image_to_structure_log_likelihood_fn = image_to_structure_log_likelihood_fn
            self.loss_fn_constant_args = loss_fn_constant_args

        self.dilated_mask = dilated_mask
        self.pose_grid_search = pose_grid_search
        self.initialize_uniform_weights = initialize_uniform_weights
        self.n_steps_opt_weights = n_steps_opt_weights
        self.whitening_filter = whitening_filter
        self.whiten_experimental_images = whiten_experimental_images
        self.whiten_calculated_images = whiten_calculated_images
        self.normalize_experimental_images = normalize_experimental_images

    def __call__(
        self,
        structures: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
        weights: Float[Array, " n_structures"],
        relion_batch: Any,
        batch_size_structures: Optional[int] = None,
        batch_size_images: Optional[int] = None,
        lowpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
        highpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
    ):
        likelihood_matrix = compute_likelihood_matrix(
            structures,
            relion_batch["particle_stack"],
            self.amplitudes,
            self.variances,
            self.image_to_structure_log_likelihood_fn,
            self.dilated_mask,
            self.pose_grid_search,
            constant_args=self.loss_fn_constant_args,
            per_particle_args=relion_batch["per_particle_args"],
            batch_size_structures=batch_size_structures,
            batch_size_images=batch_size_images,
            whitening_filter=self.whitening_filter,
            whiten_experimental_images=self.whiten_experimental_images,
            whiten_calculated_images=self.whiten_calculated_images,
            normalize_experimental_images=self.normalize_experimental_images,
            lowpass_filter=lowpass_filter,
            highpass_filter=highpass_filter,
        )
        if self.initialize_uniform_weights:
            init_weights = jnp.ones(len(weights)) / len(weights)
        else:
            init_weights = weights
        weights = _optimize_weights(
            init_weights, likelihood_matrix, n_steps=self.n_steps_opt_weights
        )
        weights = jnp.clip(weights, 1e-10, 1.0)
        weights = weights / jnp.sum(weights)
        return compute_neg_log_likelihood_from_weights(
            weights, likelihood_matrix
        ), weights


class LikelihoodMCMCWeightsFn(AbstractLikelihoodFn, strict=True):
    variances: Float[Array, "n_structures n_atoms n_gaussians_per_atom"]
    amplitudes: Float[Array, "n_structures n_atoms n_gaussians_per_atom"]
    image_to_structure_log_likelihood_fn: LossFn
    loss_fn_constant_args: ConstantT
    dilated_mask: Optional[DilatedMask] = None
    pose_grid_search: Optional[HierarchicalSO3GridSearch] = None
    initialize_uniform_weights: bool = True
    n_steps_opt_weights: int = 500
    dirichlet_concentration: Float = 1000.0
    whitening_filter: Optional[Float[Array, "image_size image_size"]] = None
    whiten_experimental_images: bool = False
    whiten_calculated_images: bool = False
    normalize_experimental_images: bool = False

    def __init__(
        self,
        amplitudes: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
        variances: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
        image_to_structure_log_likelihood_fn: Literal[
            "iso_gaussian", "iso_gaussian_var_marg"
        ]
        | LossFn,
        loss_fn_constant_args: Optional[ConstantT] = None,
        dilated_mask: Optional[DilatedMask] = None,
        pose_grid_search: Optional[HierarchicalSO3GridSearch] = None,
        initialize_uniform_weights: bool = True,
        n_steps_opt_weights: int = 500,
        dirichlet_concentration: Float = 1000.0,
        whitening_filter: Optional[Float[Array, "image_size image_size"]] = None,
        whiten_experimental_images: bool = False,
        whiten_calculated_images: bool = False,
        normalize_experimental_images: bool = False,
    ):
        assert jnp.all(variances > 0.0), "Variances must be positive."
        assert jnp.all(amplitudes > 0.0), "Amplitudes must be positive."
        if whitening_filter is not None:
            assert whiten_experimental_images or whiten_calculated_images, (
                "Provided `whitening_filter`, but `whiten_experimental_images`"
                " and `whiten_calculated_images` are both False."
            )

        self.variances = variances
        self.amplitudes = amplitudes
        if image_to_structure_log_likelihood_fn == "iso_gaussian":
            self.image_to_structure_log_likelihood_fn = likelihood_isotropic_gaussian
            self.loss_fn_constant_args = (
                1.0 if loss_fn_constant_args is None else loss_fn_constant_args
            )
        elif image_to_structure_log_likelihood_fn == "iso_gaussian_var_marg":
            self.image_to_structure_log_likelihood_fn = (
                likelihood_isotropic_gaussian_marginalized
            )
            self.loss_fn_constant_args = (
                1.0 if loss_fn_constant_args is None else loss_fn_constant_args
            )
        elif image_to_structure_log_likelihood_fn == "sliced_wasserstein":
            self.image_to_structure_log_likelihood_fn = likelihood_sliced_wasserstein
            self.loss_fn_constant_args = (
                (18, 2) if loss_fn_constant_args is None else loss_fn_constant_args
            )
        else:
            assert callable(image_to_structure_log_likelihood_fn), (
                "If `image_to_structure_log_likelihood_fn` is not 'iso_gaussian' or "
                + "'iso_gaussian_var_marg', it must be a callable function."
            )
            self.image_to_structure_log_likelihood_fn = image_to_structure_log_likelihood_fn
            self.loss_fn_constant_args = loss_fn_constant_args

        self.dilated_mask = dilated_mask
        self.pose_grid_search = pose_grid_search
        self.initialize_uniform_weights = initialize_uniform_weights
        self.n_steps_opt_weights = n_steps_opt_weights
        self.dirichlet_concentration = dirichlet_concentration
        self.whitening_filter = whitening_filter
        self.whiten_experimental_images = whiten_experimental_images
        self.whiten_calculated_images = whiten_calculated_images
        self.normalize_experimental_images = normalize_experimental_images

    def __call__(
        self,
        key: PRNGKeyArray,
        structures: Float[Array, "n_structures n_atoms n_gaussians_per_atom"],
        weights: Float[Array, " n_structures"],
        likelihood_weight: Float,
        mcmc_temperature: Float,
        relion_batch: Any,
        batch_size_structures: Optional[int] = None,
        batch_size_images: Optional[int] = None,
        lowpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
        highpass_filter: Optional[Float[Array, "image_size image_size"]] = None,
    ):
        likelihood_matrix = compute_likelihood_matrix(
            structures,
            relion_batch["particle_stack"],
            self.amplitudes,
            self.variances,
            self.image_to_structure_log_likelihood_fn,
            self.dilated_mask,
            self.pose_grid_search,
            constant_args=self.loss_fn_constant_args,
            per_particle_args=relion_batch["per_particle_args"],
            batch_size_structures=batch_size_structures,
            batch_size_images=batch_size_images,
            whitening_filter=self.whitening_filter,
            whiten_experimental_images=self.whiten_experimental_images,
            whiten_calculated_images=self.whiten_calculated_images,
            normalize_experimental_images=self.normalize_experimental_images,
            lowpass_filter=lowpass_filter,
            highpass_filter=highpass_filter,
        )
        if self.initialize_uniform_weights:
            init_weights = jnp.ones(len(weights)) / len(weights)
        else:
            init_weights = weights
        weights = _mcmc_weights(
            init_weights,
            likelihood_matrix,
            likelihood_weight,
            n_steps=self.n_steps_opt_weights,
            key=key,
            concentration=self.dirichlet_concentration,
            mcmc_temperature=mcmc_temperature,
        )
        weights = jnp.clip(weights, 1e-10, 1.0)
        weights = weights / jnp.sum(weights)
        return compute_neg_log_likelihood_from_weights(
            weights, likelihood_matrix
        ), weights


@eqx.filter_jit
def _optimize_weights(
    init_weights: Float[Array, " n_structures"],
    likelihood_matrix: Float[Array, "n_images n_structures"],
    n_steps: Int,
) -> Float[Array, " n_structures"]:
    pg = ProjectedGradient(
        fun=compute_neg_log_likelihood_from_weights,
        projection=projection_simplex,
        maxiter=n_steps,
    )
    return pg.run(init_weights, likelihood_matrix=likelihood_matrix).params


@eqx.filter_jit
def _mcmc_weights(
    weights: Float[Array, " n_structures"],
    likelihood_matrix: Float[Array, "n_images n_structures"],
    likelihood_weight: Float,
    n_steps: Int,
    key: PRNGKeyArray,
    concentration: Float,
    mcmc_temperature: Float,
    k_boltz: Float = 0.008314462618,
    eps: Float = 1e-10,
) -> Float[Array, " n_structures"]:
    def _dirichlet_logpdf(x, alpha, eps):
        x = jnp.clip(x, eps, 1.0)
        alpha = jnp.clip(alpha, eps, None)
        return (
            jax.scipy.special.gammaln(jnp.sum(alpha))
            - jnp.sum(jax.scipy.special.gammaln(alpha))
            + jnp.sum((alpha - 1.0) * jnp.log(x))
        )

    current_nll = likelihood_weight * compute_neg_log_likelihood_from_weights(
        weights, likelihood_matrix
    )

    def _mcmc_step(_, carry):
        key, weights, current_nll = carry

        key, proposal_key, accept_key = jax.random.split(key, 3)

        current_alpha = concentration * weights + eps
        proposal = jax.random.dirichlet(proposal_key, alpha=current_alpha)

        proposal_nll = likelihood_weight * compute_neg_log_likelihood_from_weights(
            proposal, likelihood_matrix
        )

        forward_log_q = _dirichlet_logpdf(proposal, current_alpha, eps)

        reverse_alpha = concentration * proposal + eps
        reverse_log_q = _dirichlet_logpdf(weights, reverse_alpha, eps)

        log_accept_ratio = (
            (-proposal_nll + current_nll) / (k_boltz * mcmc_temperature)
            + reverse_log_q
            - forward_log_q
        )

        rand = jax.random.uniform(accept_key)
        accept = jnp.log(rand) < log_accept_ratio

        weights = jnp.where(accept, proposal, weights)
        current_nll = jnp.where(accept, proposal_nll, current_nll)

        return key, weights, current_nll

    _, weights, _ = jax.lax.fori_loop(
        lower=0,
        upper=n_steps,
        body_fun=_mcmc_step,
        init_val=(key, weights, current_nll),
    )

    return weights
