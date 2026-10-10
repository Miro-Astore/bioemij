import jax
import jax.numpy as jnp
import numpy as np
import yaml
from cryospax import RelionParticleDataset, RelionParticleParameterFile

from bioemij.ensemble_optimization import (
    compute_likelihood_matrix,
    likelihood_isotropic_gaussian,
    likelihood_isotropic_gaussian_marginalized,
    likelihood_sliced_wasserstein,
    make_image_model_from_gmm,
)
from bioemij.internal import DatasetSimulatorConfig
from bioemij.simulator import DilatedMask


def test_compute_likelihood_matrix():
    with open("tests/data/particle_stack/config_data_generation.yaml", "r") as f:
        config_json = yaml.safe_load(f)

    key = jax.random.key(config_json["rng_seed"])
    config = dict(DatasetSimulatorConfig(**config_json).model_dump())
    relion_stack = RelionParticleDataset(
        RelionParticleParameterFile(
            path_to_starfile=config["path_to_starfile"],
            mode="r",
        ),
        path_to_relion_project=config["path_to_relion_project"],
        mode="r",
    )

    n_structures = 2
    n_atoms = jax.random.randint(key, (1,), 10, 20)[0]
    n_gaussians_per_atom = 5
    ensemble_structures = jax.random.normal(key, (n_structures, n_atoms, 3))
    gaussian_amplitudes = jax.random.normal(
        key, (n_structures, n_atoms, n_gaussians_per_atom)
    )
    gaussian_variances = (
        jax.random.normal(key, (n_structures, n_atoms, n_gaussians_per_atom)) ** 2
    )

    # test of likelihood_isotropic_gaussian
    image_to_structure_log_likelihood_fn = likelihood_isotropic_gaussian
    per_particle_args_noise_variance = 1.0

    n_particles = 5

    likelihood_matrix = compute_likelihood_matrix(
        ensemble_structures,
        relion_stack[:n_particles],
        gaussian_amplitudes,
        gaussian_variances,
        image_to_structure_log_likelihood_fn,
        pose_grid_search=None,
        per_particle_args=per_particle_args_noise_variance,
        constant_args=1.0,
    )
    assert likelihood_matrix.shape == (n_particles, n_structures)

    # test of _likelihood_sliced_wasserstein
    image_to_structure_log_likelihood_fn = likelihood_sliced_wasserstein
    per_particle_args_n_projections = 7

    n_particles = 5
    likelihood_matrix = compute_likelihood_matrix(
        ensemble_structures,
        relion_stack[:n_particles],
        gaussian_amplitudes,
        gaussian_variances,
        image_to_structure_log_likelihood_fn,
        constant_args=(per_particle_args_n_projections, 2),
        per_particle_args=(),
    )
    assert likelihood_matrix.shape == (n_particles, n_structures)


def test_compute_likelihood_matrix_batched_matches_unbatched():
    """The `batch_size_*` path must agree with the fully vmapped path."""
    with open("tests/data/particle_stack/config_data_generation.yaml", "r") as f:
        config_json = yaml.safe_load(f)

    key = jax.random.key(config_json["rng_seed"])
    config = dict(DatasetSimulatorConfig(**config_json).model_dump())
    relion_stack = RelionParticleDataset(
        RelionParticleParameterFile(
            path_to_starfile=config["path_to_starfile"],
            mode="r",
        ),
        path_to_relion_project=config["path_to_relion_project"],
        mode="r",
    )

    n_structures = 4
    n_atoms = 12
    n_gaussians_per_atom = 5
    n_particles = 4

    ensemble_structures = jax.random.normal(key, (n_structures, n_atoms, 3))
    gaussian_amplitudes = jax.random.normal(
        key, (n_structures, n_atoms, n_gaussians_per_atom)
    )
    gaussian_variances = (
        jax.random.normal(key, (n_structures, n_atoms, n_gaussians_per_atom)) ** 2
    )

    kwargs = dict(
        constant_args=1.0,
        per_particle_args=1.0,
        pose_grid_search=None,
    )
    unbatched = compute_likelihood_matrix(
        ensemble_structures,
        relion_stack[:n_particles],
        gaussian_amplitudes,
        gaussian_variances,
        likelihood_isotropic_gaussian,
        **kwargs,
    )
    batched = compute_likelihood_matrix(
        ensemble_structures,
        relion_stack[:n_particles],
        gaussian_amplitudes,
        gaussian_variances,
        likelihood_isotropic_gaussian,
        batch_size_structures=2,
        batch_size_images=2,
        **kwargs,
    )
    assert batched.shape == (n_particles, n_structures)
    np.testing.assert_allclose(batched, unbatched, rtol=1e-10, atol=1e-10)


def test_masked_marginalized_likelihood_counts_only_pixels_in_the_mask():
    """With a mask, the scale/offset fit and `n_pixels` cover the projected mask
    only -- not the zeroed pixels around it, which would inflate every
    log-likelihood difference by (image size / mask size)."""
    with open("tests/data/particle_stack/config_data_generation.yaml", "r") as f:
        config_json = yaml.safe_load(f)

    key = jax.random.key(config_json["rng_seed"])
    config = dict(DatasetSimulatorConfig(**config_json).model_dump())
    relion_stack = RelionParticleDataset(
        RelionParticleParameterFile(
            path_to_starfile=config["path_to_starfile"],
            mode="r",
        ),
        path_to_relion_project=config["path_to_relion_project"],
        mode="r",
    )
    particle = relion_stack[0]
    image_config = particle["parameters"]["image_config"]
    box_size = image_config.shape[0]

    n_atoms = 12
    structure = jax.random.normal(key, (n_atoms, 3))
    amplitudes = jnp.abs(jax.random.normal(key, (n_atoms, 5))) + 0.1
    variances = jnp.abs(jax.random.normal(key, (n_atoms, 5))) + 0.1

    # An off-centre block, so its projection covers only part of the image.
    mask_grid = np.zeros((box_size,) * 3)
    mask_grid[12:20, 12:20, 4:12] = 1.0
    dilated_mask = DilatedMask(jnp.asarray(mask_grid), image_config)

    masked = likelihood_isotropic_gaussian_marginalized(
        structure,
        particle,
        amplitudes,
        variances,
        dilated_mask,
        None,
        1.0,
        None,
        None,
        False,
        False,
        False,
    )

    in_mask = np.asarray(dilated_mask.project(particle["parameters"]["pose"]) > 0.5)
    assert 0 < in_mask.sum() < in_mask.size
    observed = np.asarray(particle["images"])[in_mask]
    computed = np.asarray(
        make_image_model_from_gmm(
            structure,
            particle["images"],
            particle,
            amplitudes,
            variances,
            None,
            None,
            False,
        ).simulate()
    )[in_mask]
    scale, offset = np.polyfit(computed, observed, 1)
    expected = -in_mask.sum() * np.log(
        np.linalg.norm(scale * computed - observed + offset)
    )
    np.testing.assert_allclose(masked, expected, rtol=1e-8)
