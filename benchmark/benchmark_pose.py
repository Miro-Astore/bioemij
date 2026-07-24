import logging
import os
from glob import glob

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from cryojax.dataset import RelionParticleParameterFile, RelionParticleStackDataset

from bioemij.ensemble_optimization import (
    HierarchicalSO3GridSearch,
    make_image_model_from_gmm,
)


logger = logging.getLogger(__name__)
logger.setLevel(
    logging.INFO
)  # run pytest with -o log_cli=true -o log_cli_level=INFO to see these logs


def parse_xyz(filename):
    """
    Parse a simple .xyz file into a NumPy array of shape (n_atoms, 3).
    """
    logger.info(f"Parsing XYZ file: {filename}")
    with open(filename, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    coord_lines = lines[2 : 2 + n_atoms]

    coords = []
    for line in coord_lines:
        parts = line.split()
        x, y, z = map(float, parts[1:4])  # skip atom label
        coords.append([x, y, z])

    return jnp.array(coords)


def geodesic_distance_deg(q1, q2):
    """Compute the geodesic distance between two quaternions."""
    dot_product = jnp.abs(jnp.dot(q1, q2))
    dot_product = jnp.clip(dot_product, -1.0, 1.0)  # Numerical stability
    angle = 2 * jnp.arccos(dot_product)
    return angle * (180.0 / jnp.pi)  # Convert to degrees


def plot_simulated_versus_observed(
    simulated_image, observed_image, snr, rmsd_nm, pose_type, odir
):
    difference = observed_image - simulated_image

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # --- Simulated ---
    im0 = axes[0].imshow(simulated_image, cmap="gray")
    axes[0].set_title(f"Simulated\nSNR={snr}, RMSD={rmsd_nm} nm")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    # --- Observed ---
    im1 = axes[1].imshow(observed_image, cmap="gray")
    axes[1].set_title("Observed")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    # --- Difference ---
    im2 = axes[2].imshow(difference, cmap="gray")
    axes[2].set_title("Observed − Simulated")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    # --- Save ---
    # mkdir images directory if it doesn't exist
    if not os.path.exists(os.path.join(odir, "images")):
        os.makedirs(os.path.join(odir, "images"))
    out_path = os.path.join(
        odir, f"images/image_triplet_{pose_type}_snr{snr}_rmsd{rmsd_nm}nm.png"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def benchmark_pose_finder(data_dir, odir):
    dot_bracket_str = (
        "....((((((...((((((.....(((.((((.(((..(((((((((....))))))))).."
        "((.......))....)))......)))))))....))))))..)).))))((...((((..."
        "(((((((((...)))))))))..))))...)).."
    )
    gauss_amp = 1.0  # TODO: check
    gauss_var = 9.0  # TODO: check

    results = []
    for images_dir in glob(
        os.path.join(data_dir, "images/SNR*/image_*_closed_SNR*/")
    ):  # hanging backslash is important
        logger.info(f"Processing images in directory: {images_dir}")
        for start_structure in glob(
            os.path.join(data_dir, "structures/rmsd_*nm/rmsd_*.xyz")
        ):  # [1:2]: # subset
            snr = float(images_dir.split("SNR")[-1].split("/")[0])
            rmsd_nm = float(
                start_structure.split("rmsd_")[-1].split("nm")[0].split(".xyz")[0]
            )
            logger.info(f"Using SNR: {snr}")
            logger.info(f"Using start structure with RMSD: {rmsd_nm} nm")
            ground_truth_shift = jnp.array([40, -50])  # in angstroms
            ground_truth_quat = jnp.array([0.68041384, 0.27216554, 0.4082483, 0.54433113])
            dirname = os.path.dirname(images_dir)
            logger.info(f"Parsing ground truth from directory name: {dirname}")
            pose_type = None
            if "_rot_" in dirname:
                ground_truth_shift = jnp.zeros(2)
                pose_type = "rotationOnly"
            elif "_trans_" in dirname:
                identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
                ground_truth_quat = identity_quat
                pose_type = "translationOnly"
            else:
                assert "_rotandtrans_" in dirname
                pose_type = "rotationAndTranslation"

            logger.info(f"Using start structure: {start_structure}")
            logger.info(f"Ground truth shift {ground_truth_shift}")
            logger.info(f"Ground truth rotation {ground_truth_quat}")

            logger.info("Creating dataset and dataloader")
            dataset = RelionParticleStackDataset(
                RelionParticleParameterFile(f"{images_dir}/starfile.star"),
                f"{images_dir}/",
            )

            logger.info("Setting up topology and forcefield")

            logger.info("Setting up likelihood function")
            n_residues = len(dot_bracket_str)
            gaussian_amplitudes = jnp.array([[gauss_amp]] * 2 * n_residues)
            gaussian_variances = jnp.array([[gauss_var]] * 2 * n_residues)

            logger.info("Parsing start structure")
            positions = parse_xyz(start_structure)
            assert len(positions) == n_residues * 2, (
                f"Number of beads in structure ({len(positions)}) does"
                " not match the number of residues from dot-bracket ({n_residues})"
            )

            logger.info("Initializing structures")

            logger.info("Creating image model from GMM with pose estimation")
            # Define pose search
            pose_search = HierarchicalSO3GridSearch(
                base_grid_res=1,
                n_rounds=5,
                n_candidates=40,
            )
            image_model = make_image_model_from_gmm(
                structure=positions,
                relion_stack=dataset[0],
                amplitudes=gaussian_amplitudes,
                variances=gaussian_variances,
                pose_grid_search=pose_search,
            )
            simulated_image = image_model.simulate()
            observed_image = dataset[0]["images"]
            fname = os.path.join(
                odir, f"data/data_{pose_type}_snr{snr}_rmsd{rmsd_nm}nm.npz"
            )
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            jnp.savez(
                fname, simulated_image=simulated_image, observed_image=observed_image
            )
            logger.info(f"Saved final images in {fname}")
            plot_simulated_versus_observed(
                simulated_image, observed_image, snr, rmsd_nm, pose_type, odir
            )

            logger.info(f"wxyz={image_model.pose.wxyz}")
            angle_diff = geodesic_distance_deg(image_model.pose.wxyz, ground_truth_quat)
            logger.info(
                f"Angle difference from ground truth quaternion: {angle_diff} degrees"
            )

            logger.info(f"shift={image_model.pose.offset_in_angstroms}")
            distance_shift = jnp.linalg.norm(
                image_model.pose.offset_in_angstroms - ground_truth_shift
            )
            logger.info(f"Distance from ground truth shift: {distance_shift} angstroms")

            results.append(
                {
                    "snr": snr,
                    "rmsd": rmsd_nm,
                    "pose_type": pose_type,
                    "estimated_quat": image_model.pose.wxyz,
                    "ground_truth_quat": ground_truth_quat,
                    "angle_diff_deg": angle_diff,
                    "estimated_shift": image_model.pose.offset_in_angstroms,
                    "ground_truth_shift": ground_truth_shift,
                    "shift_distance_angstroms": distance_shift,
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(odir, "pose_estimation_results.csv"), index=False)


def plot(odir):
    # ---- Aggregate mean errors ----

    df = pd.read_csv(os.path.join(odir, "pose_estimation_results.csv"))
    quat_mean = (
        df.groupby(["snr", "rmsd", "pose_type"])["angle_diff_deg"].mean().reset_index()
    )
    shift_mean = (
        df.groupby(["snr", "rmsd", "pose_type"])["shift_distance_angstroms"]
        .mean()
        .reset_index()
    )

    # ---- Loop over pose types ----
    for pose_type in sorted(df["pose_type"].unique()):
        quat_sub = quat_mean[quat_mean["pose_type"] == pose_type]
        shift_sub = shift_mean[shift_mean["pose_type"] == pose_type]

        # ---- Plot 1: Quaternion error ----
        plt.figure()
        for snr in sorted(quat_sub["snr"].unique()):
            sub = quat_sub[quat_sub["snr"] == snr]
            plt.plot(sub["rmsd"], sub["angle_diff_deg"], marker="o", label=f"SNR={snr}")

        plt.xlabel("RMSD")
        plt.ylabel("Mean Angle Error (deg)")
        plt.title(f"Quaternion Accuracy vs RMSD ({pose_type})")
        plt.legend()

        outpath = os.path.join(odir, f"pose_estimation_results_rotation_{pose_type}.png")
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()

        # ---- Plot 2: Shift error ----
        plt.figure()
        for snr in sorted(shift_sub["snr"].unique()):
            sub = shift_sub[shift_sub["snr"] == snr]
            plt.plot(
                sub["rmsd"],
                sub["shift_distance_angstroms"],
                marker="o",
                label=f"SNR={snr}",
            )

        plt.xlabel("RMSD")
        plt.ylabel("Mean Shift Error (Å)")
        plt.title(f"Shift Accuracy vs RMSD ({pose_type})")
        plt.legend()

        outpath = os.path.join(odir, f"pose_estimation_results_shift_{pose_type}.png")
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    print(HERE)
    odir = os.path.join(HERE, "../benchmark")
    data_dir = os.path.join(HERE, "../tests/data/pose_test_data_simple")
    benchmark_pose_finder(data_dir, odir)
    plot(odir)
