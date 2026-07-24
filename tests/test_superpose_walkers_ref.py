import jax.numpy as jnp
import mdtraj
import numpy as np
from jaxtyping import Array, Float

from bioemij.utils import procrustes


def superpose_structures_ref(
    self,
    structures: Float[Array, "n_structures n_atoms 3"],
    reference_structure: mdtraj.Trajectory,
) -> Float[Array, "n_structures n_atoms 3"]:
    """Copied here becuase it was inside a class"""
    superposed_structures = []
    for structure in structures:
        structure_mdtraj = mdtraj.Trajectory(
            xyz=structure / 10.0,
            topology=reference_structure.topology,
        )
        structure_mdtraj.superpose(reference_structure)
        structure_pos = structure_mdtraj.xyz * 10.0
        superposed_structures.append(structure_pos[0])
    return jnp.array(superposed_structures)


def superpose_structures_ref_new(
    self,
    structures: Float[Array, "n_structures n_atoms 3"],
    reference_structure: mdtraj.Trajectory,
) -> Float[Array, "n_structures n_atoms 3"]:
    superposed, _ = procrustes(
        jnp.array(structures), jnp.array(reference_structure.xyz[0] * 10.0)
    )
    return superposed


def test_superpose_structures_ref():
    """
    This test compares the original superpose_structures_ref function that uses MDTraj."""
    reference_structure = mdtraj.load("tests/data/ala_model_1.pdb")
    n_structures = 5
    # create noisy structures in Angstroms by converting ref nm->Å
    # and adding noise + small rotations
    ref_A = reference_structure.xyz[0] * 10.0  # nm -> Å
    rng = np.random.default_rng(123)
    structures = []
    for angle in rng.normal(scale=0.5, size=n_structures):
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0, 0, 1],
            ]
        )
        print(R)
        pert = ref_A @ R.T
        pert += rng.normal(scale=0.00, size=pert.shape)
        structures.append(pert)
    structures = np.stack(structures).astype(float)
    superposed = superpose_structures_ref(None, jnp.array(structures), reference_structure)
    superposed_new = superpose_structures_ref_new(
        None, jnp.array(structures), reference_structure
    )
    np.testing.assert_allclose(superposed, superposed_new, atol=1e-4)
