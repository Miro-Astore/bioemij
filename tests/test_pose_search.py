import cryojax.simulator as cxs
import jax
import jax.numpy as jnp
import numpy as np

from bioemij.ensemble_optimization import HierarchicalSO3GridSearch


def test_max_shift_keeps_the_search_off_a_neighbouring_particle():
    """A brighter copy of the particle 40 A away wins an unrestricted search; with
    `max_shift_in_angstroms` the offset must stay on the particle at the centre."""
    n_atoms = 30
    positions = jax.random.normal(jax.random.key(0), (n_atoms, 3)) * 8.0
    volume = cxs.GaussianMixtureVolume(
        positions,
        amplitudes=jnp.ones((n_atoms, 1)) * 20.0,
        variances=jnp.ones((n_atoms, 1)) * 8.0,
    )
    image_config = cxs.BasicImageConfig(
        shape=(64, 64), pixel_size=2.0, voltage_in_kilovolts=300.0
    )
    transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.AstigmaticCTF(defocus_in_angstroms=10000.0)
    )
    # A grid orientation, so only the offset is being tested.
    wxyz = HierarchicalSO3GridSearch(1, 0, 1).base_quats[100]

    def simulate(offset_x):
        pose = cxs.QuaternionPose(
            wxyz=wxyz, offset_x_in_angstroms=offset_x, offset_y_in_angstroms=0.0
        )
        return cxs.make_image_model(
            volume, image_config, pose, transfer_theory, normalizes_signal=False
        ).simulate()

    image = simulate(0.0) + 2.0 * simulate(40.0)

    def search_offset(max_shift):
        search = HierarchicalSO3GridSearch(
            1, 0, 1, grid_search_batch_size=64, max_shift_in_angstroms=max_shift
        )
        pose = search(volume, image, image_config, transfer_theory, None, False)
        return np.asarray(pose.offset_in_angstroms)

    np.testing.assert_allclose(search_offset(None), [40.0, 0.0], atol=2.0)
    np.testing.assert_allclose(search_offset(15.0), [0.0, 0.0], atol=2.0)
