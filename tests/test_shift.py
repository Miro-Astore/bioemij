import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.ndimage import make_coordinate_grid, PhaseShiftFFT

from bioemij.ensemble_optimization._pose_search import (
    compute_correlation_at_optimal_offset,
)


def make_synthetic_2D_gaussian_image(shape, offset, gaussian_std, coordinate_grid):
    image = jnp.exp(
        -(
            (coordinate_grid[..., 0] ** 2 + coordinate_grid[..., 1] ** 2)
            / (2 * gaussian_std**2)
        )
    )  # Gaussian blob

    # `PhaseShiftFFT` infers the frequency grid from the image shape, and takes
    # `offset` in pixel units when `pixel_size` is left at its default of 1.0.
    shift_op = PhaseShiftFFT(offset=jnp.asarray(offset, dtype=float))
    image_shifted = jnp.fft.irfftn(shift_op(jnp.fft.rfftn(image)), s=shape)
    return image, image_shifted


def test_fit_global_shift_by_phase_correlation_accuracy():
    n_trials = 10
    for _ in range(n_trials):
        key = jax.random.key(42)
        size = (
            jax.random.randint(key, (), 64, 129) // 2
        ) * 2  # size between 64 and 128, even
        key = jax.random.key(0)
        shift = jax.random.randint(
            key, (2,), -size // 4, size // 4
        )  # two random integers within +/- size/4
        gaussian_std = jax.random.uniform(jax.random.key(1), (), minval=0.1, maxval=0.2)
        shape = (int(size), int(size))
        coordinate_grid = make_coordinate_grid(shape, grid_spacing=1.0)

        img1, img2 = make_synthetic_2D_gaussian_image(
            shape, shift, gaussian_std, coordinate_grid
        )
        loss, optimal_shift = eqx.filter_jit(compute_correlation_at_optimal_offset)(
            img2, img1, coordinate_grid, corr_mask=None
        )
        np.testing.assert_allclose(optimal_shift, shift, atol=1e-6)
        assert np.isclose(loss, 1.0, atol=1e-2)


def test_fit_global_shift_by_phase_correlation_jittable():
    key = jax.random.key(42)
    size = (
        jax.random.randint(key, (), 64, 129) // 2
    ) * 2  # size between 64 and 128, even
    key = jax.random.key(0)
    shift = jax.random.randint(
        key, (2,), -size // 4, size // 4
    )  # two random integers within +/- size/4
    gaussian_std = jax.random.uniform(jax.random.key(1), (), minval=0.1, maxval=0.2)
    shape = (int(size), int(size))
    coordinate_grid = make_coordinate_grid(shape, grid_spacing=1.0)
    img1, img2 = make_synthetic_2D_gaussian_image(
        shape, shift, gaussian_std, coordinate_grid
    )
    loss, optimal_shift = eqx.filter_jit(compute_correlation_at_optimal_offset)(
        img2, img1, coordinate_grid, corr_mask=None
    )
    np.testing.assert_allclose(optimal_shift, shift, atol=1e-6)
    assert np.isclose(loss, 1.0, atol=1e-2)


def test_grad_through_phase_correlation_max_wrt_spread():
    """Ensure we can take gradients of the max of the correlation map wrt spread.

    Note that taking gradients wrt the correlation is not necessarily meaningful in
    itself, because it does not decay when they match.

    """
    size = 64
    # choose a non-zero integer shift
    shift = (6, -3)

    # spread is the parameter we want gradients for
    spread = jnp.array(0.2)

    # Build images using jax ops so the computation is traced
    shape = (int(size), int(size))
    coordinate_grid = make_coordinate_grid(shape, grid_spacing=1.0)
    img_ref, _ = make_synthetic_2D_gaussian_image(shape, shift, spread, coordinate_grid)

    def loss_fn(s):
        # regenerate images from spread s so gradient flows from s -> images -> corr_map
        _, img_shifted = make_synthetic_2D_gaussian_image(
            shape, shift, s, coordinate_grid
        )
        correlation, _ = compute_correlation_at_optimal_offset(
            img_shifted, img_ref, coordinate_grid, corr_mask=None
        )
        # take the peak value (use max); negative because often losses are minimised
        return -correlation

    # compute gradient
    grad_fn = jax.grad(loss_fn)
    g = grad_fn(spread)

    # Assert gradient is a finite scalar
    assert jnp.shape(g) == ()
    assert jnp.isfinite(g), f"Gradient is not finite: {g}"

    # Optionally ensure it's not NaN
    assert not jnp.isnan(g)


@pytest.mark.parametrize(
    "offset_in_angstroms", [(0.0, 0.0), (8.0, -6.0), (-12.0, 20.0), (4.0, 4.0)]
)
def test_optimal_offset_matches_pose_offset_in_angstroms(offset_in_angstroms):
    """The recovered offset must match a `cryojax` pose offset in angstroms.

    This pins down the sign, axis order and units of the offset returned by
    `compute_correlation_at_optimal_offset`, which `HierarchicalSO3GridSearch`
    feeds straight into `QuaternionPose.offset_{x,y}_in_angstroms`.
    """
    pixel_size = 2.0
    n_atoms = 30
    positions = jax.random.normal(jax.random.key(0), (n_atoms, 3)) * 12.0
    volume = cxs.GaussianMixtureVolume(
        positions,
        amplitudes=jnp.ones((n_atoms, 1)) * 20.0,
        variances=jnp.ones((n_atoms, 1)) * 8.0,
    )
    image_config = cxs.BasicImageConfig(
        shape=(64, 64), pixel_size=pixel_size, voltage_in_kilovolts=300.0
    )
    transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.AstigmaticCTF(defocus_in_angstroms=10000.0)
    )
    wxyz = jnp.array([0.5, 0.5, 0.5, 0.5])

    def simulate(offset_x, offset_y):
        pose = cxs.QuaternionPose(
            wxyz=wxyz,
            offset_x_in_angstroms=offset_x,
            offset_y_in_angstroms=offset_y,
        )
        return cxs.make_image_model(
            volume, image_config, pose, transfer_theory, normalizes_signal=True
        ).simulate()

    image_ref = simulate(0.0, 0.0)
    image_shifted = simulate(*offset_in_angstroms)

    correlation, optimal_offset = compute_correlation_at_optimal_offset(
        image_shifted,
        image_ref,
        image_config.get_coordinate_grid(physical=True),
        corr_mask=None,
    )
    # the offset is read off a pixel grid, so it is exact only to one pixel
    np.testing.assert_allclose(optimal_offset, offset_in_angstroms, atol=pixel_size)
    assert np.isclose(correlation, 1.0, atol=1e-2)
