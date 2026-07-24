from typing import Any, Optional, Tuple

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float, Int, Scalar


class _OptimizerState(eqx.Module, strict=True):
    step: Int[Array, ""]
    old_loss: Scalar
    curr_loss: Scalar
    opt_state: Tuple[Any, Any]


class _AtomicModel(eqx.Module):
    positions: Float[Array, "n_atoms 3"]
    amplitudes: Float[Array, "n_atoms n_gaussians"]
    variances: Float[Array, "n_atoms n_gaussians"]


class _Volume(eqx.Module):
    voxel_grid: Float[Array, "z y x"]
    voxel_size: float


class Solution(eqx.Module):
    rotation_matrix: Float[Array, "3 3"]
    offset: Float[Array, " 3"]
    correlation: Float
    n_steps: Int


class ModelToVolumeAligner(eqx.Module):
    volume: _Volume
    optimizers: Tuple[Any, Any]
    rtol: float
    atol: float
    max_steps: int

    def __init__(
        self,
        real_voxel_grid: Float[Array, "z y x"],
        voxel_size: float,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        max_steps: int = 100,
        optimizers: Optional[Tuple[Any, Any]] = None,
    ):
        assert max_steps > 0, "max_steps must be a positive integer."
        assert voxel_size > 0.0, "voxel_size must be positive."

        if optimizers is None:
            optimizers = (
                optax.adabelief(learning_rate=1e-1, nesterov=True),
                optax.adabelief(learning_rate=1e-1, nesterov=True),
            )
        self.optimizers = optimizers
        self.rtol = rtol
        self.atol = atol
        self.max_steps = int(max_steps)
        self.volume = _Volume(voxel_grid=real_voxel_grid, voxel_size=voxel_size)

    @eqx.filter_jit
    def _init(
        self,
        quat_init: Float[Array, "4"],
        offset_init: Float[Array, "3"],
    ) -> _OptimizerState:
        opt_state = (
            self.optimizers[0].init(quat_init),
            self.optimizers[1].init(offset_init),
        )
        maxval = jnp.array(jnp.finfo(jnp.float32).max, jnp.float32)
        return _OptimizerState(
            step=jnp.array(0), old_loss=maxval, curr_loss=maxval, opt_state=opt_state
        )

    @eqx.filter_jit
    def _step(
        self,
        val: Tuple[Float[Array, "4"], Float[Array, "3"], _OptimizerState],
        args: Tuple[_AtomicModel, _Volume],
    ) -> Tuple[Float[Array, "4"], Float[Array, "3"], _OptimizerState]:
        quat, offset, state = val
        q_opt_state, d_opt_state = state.opt_state
        optim_q, optim_d = self.optimizers

        loss, grad_q = loss_and_grad_quat_fn(quat, offset, args)
        updates_q, q_opt_state = optim_q.update(grad_q, q_opt_state)
        quat = optax.apply_updates(quat, updates_q)
        quat /= jnp.linalg.norm(quat, keepdims=True)

        loss, grad_d = loss_and_grad_offset_fn(quat, offset, args)
        updates_d, d_opt_state = optim_d.update(grad_d, d_opt_state)
        offset = jnp.asarray(optax.apply_updates(offset, updates_d))

        step = state.step + 1
        new_state = _OptimizerState(
            step=step,
            old_loss=state.curr_loss,
            curr_loss=loss,
            opt_state=(q_opt_state, d_opt_state),
        )

        # jax.debug.print("Step={step}, Corr={loss}", step=step, loss=1 - loss)
        return quat, offset, new_state

    @eqx.filter_jit
    def _terminate_condition(self, state: _OptimizerState) -> Bool[Array, ""]:
        cond1 = jnp.abs(state.old_loss - state.curr_loss) > (
            self.atol + self.rtol * jnp.abs(state.curr_loss)
        )
        cond2 = state.step < self.max_steps
        return jax.lax.cond(state.step < 1, lambda x: True, lambda x: cond1 & cond2, None)

    @eqx.filter_jit
    def align(
        self,
        atomic_positions_in_angstroms: Float[Array, "n_atoms 3"],
        amplitudes: Float[Array, "n_atoms n_gaussians"],
        variances: Float[Array, "n_atoms n_gaussians"],
        quat_init: Float[Array, "4"] = jnp.array([1.0, 0.0, 0.0, 0.0]),
        offset_init: Float[Array, "3"] = jnp.array([0.0, 0.0, 0.0]),
    ) -> Tuple[Float[Array, "n_atoms 3"], Solution]:
        atomic_model = _AtomicModel(
            positions=atomic_positions_in_angstroms,
            amplitudes=amplitudes,
            variances=variances,
        )

        args = (atomic_model, self.volume)
        state = self._init(quat_init, offset_init)

        init_val = (quat_init, offset_init, state)
        quat_opt, offset_opt, final_state = jax.lax.while_loop(
            lambda val: self._terminate_condition(val[2]),
            body_fun=lambda val: self._step(val, args),
            init_val=init_val,
        )

        offset_opt = jnp.asarray(offset_opt)

        rot_matrix = cxs.QuaternionPose(wxyz=quat_opt).rotation.as_matrix()

        solution = Solution(
            rotation_matrix=rot_matrix,
            offset=offset_opt,
            correlation=1 - final_state.curr_loss,
            n_steps=final_state.step,
        )

        aligned_positions = atomic_model.positions @ rot_matrix + offset_opt

        return aligned_positions, solution


def _atom_potential_to_volume(
    atom_positions: Float[Array, "n_atoms_atoms 3"],
    gaussian_amp: Float[Array, "n_atoms_atoms n_gaussians"],
    gaussian_var: Float[Array, "n_atoms_atoms n_gaussians"],
    shape: Tuple[int, int, int],
    voxel_size: float,
) -> Float[Array, "z x y"]:
    atom_potential = cxs.GaussianMixtureVolume(atom_positions, gaussian_amp, gaussian_var)

    volume = atom_potential.to_real_voxel_grid(shape=shape, voxel_size=voxel_size)
    return volume


@eqx.filter_jit
def loss_fn(
    quat: Float[Array, " 4"],
    offset: Float[Array, "3"],
    args: Tuple[_AtomicModel, _Volume],
) -> Float[Array, ""]:
    atomic_model, volume = args

    rot_matrix = cxs.QuaternionPose(
        wxyz=quat,
    ).rotation.as_matrix()

    v_rot = _atom_potential_to_volume(
        atomic_model.positions @ rot_matrix + offset,
        atomic_model.amplitudes,
        atomic_model.variances,
        shape=volume.voxel_grid.shape,
        voxel_size=volume.voxel_size,
    )
    v_rot /= v_rot.sum(keepdims=True)

    return 1 - jnp.sum(v_rot * volume.voxel_grid) / jnp.linalg.norm(
        v_rot
    ) / jnp.linalg.norm(volume.voxel_grid)


@eqx.filter_jit
def loss_and_grad_quat_fn(
    quat: Float[Array, " 4"],
    offset: Float[Array, "3"],
    args: Tuple[_AtomicModel, _Volume],
) -> Tuple[Float[Array, ""], Float[Array, " 4"]]:
    # print("compiling!")
    return jax.value_and_grad(loss_fn, argnums=0)(quat, offset, args)


@eqx.filter_jit
def loss_and_grad_offset_fn(
    quat: Float[Array, " 4"],
    offset: Float[Array, "3"],
    args: Tuple[_AtomicModel, _Volume],
) -> Tuple[Float[Array, ""], Float[Array, " 3"]]:
    # print("compiling!")
    return jax.value_and_grad(loss_fn, argnums=1)(quat, offset, args)


@eqx.filter_jit
def procrustes(
    structures: Float[Array, "n_structures n_atoms 3"],
    reference_coords: Float[Array, "n_atoms 3"],
) -> Tuple[Float[Array, "n_structures n_atoms 3"], Float[Array, "n_structures 3 3"]]:
    """Batch Kabsch alignment of structures to a reference coordinate array.

    Parameters
    ----------
    self : ignored (keeps signature similar to method form)
    structures : array (n_structures, n_atoms, 3) in same units as reference_coords
    reference_coords : array (n_atoms, 3)

    Returns
    -------
    superposed_structures : array (n_structures, n_atoms, 3)
        structures after optimal rotation + translation that aligns them to reference.
    rotation_matrices : array (n_structures, 3, 3)
        rotation matrices applied to each structure to align to reference.
    """

    structures = jnp.asarray(structures)
    ref = jnp.asarray(reference_coords)

    ref_centroid = jnp.mean(ref, axis=0)  # (3,)
    structures_centroid = jnp.mean(structures, axis=1, keepdims=True)  # (n_w,1,3)

    structures_centered = structures - structures_centroid  # (n_w, n_atoms, 3)
    ref_centered = ref - ref_centroid  # (n_atoms, 3)

    cov = jnp.einsum("wka,kb->wab", structures_centered, ref_centered)  # (n_w, 3, 3)

    U, S, Vt = jnp.linalg.svd(cov)
    V = jnp.swapaxes(Vt, -1, -2)
    UT = jnp.swapaxes(U, -1, -2)

    # Handle reflections: ensure determinant of rotation is +1
    det = jnp.linalg.det(jnp.matmul(V, UT))
    I = jnp.eye(3)
    diag = jnp.tile(I, (cov.shape[0], 1, 1))
    diag = diag.at[:, 2, 2].set(jnp.where(det < 0.0, -1.0, 1.0))

    R = jnp.matmul(V, jnp.matmul(diag, UT))  # (n_w,3,3)

    rotated = jnp.einsum(
        "wnb,wab->wna", structures_centered, R
    )  # einsum signature defines rotation convention

    # translate to reference centroid
    superposed = rotated + ref_centroid[None, None, :]
    return superposed, R
