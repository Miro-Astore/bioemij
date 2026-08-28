import jax
import numpy as np
import yaml
from cryospax import RelionParticleDataset, RelionParticleParameterFile

from bioemij.ensemble_optimization import (
    compute_likelihood_matrix,
    likelihood_isotropic_gaussian,
    likelihood_sliced_wasserstein,
)
from bioemij.internal import DatasetSimulatorConfig


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
