import cryojax.ndimage as cxim
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cryojax.coordinates import make_coordinate_grid, make_frequency_grid

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

    frequency_grid = make_frequency_grid(shape, grid_spacing=1.0, outputs_rfftfreqs=True)
    shift_op = cxim.PhaseShiftFFT(offset=offset, frequency_grid=frequency_grid)
    image_shifted = cxim.irfftn(shift_op(cxim.rfftn(image)))
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
