#!/usr/bin/env python
"""
P-S Color Metamer Synthesis Script

This script performs metamer synthesis using the Color adapted Portilla-Simoncelli model with
cross-channel statistics for color images. It processes images from a specified folder
and saves the resulting metamers and synthesis information in a structured output format.

Usage:
    python ps_color_synth.py --input_dir /path/to/images --output_dir /path/to/output [options]

"""

import argparse
import json
import os
import time
from collections import OrderedDict
from datetime import datetime
from typing import Literal

import matplotlib.pyplot as plt
import plenoptic as po
import torch
import torchvision.transforms as transforms
from PIL import Image
from plenoptic.synthesize.metamer import MetamerCTF
from plenoptic.tools import img_transforms, optim
from torch import Tensor


class PortillaSimoncelliCrossChannel(po.simul.PortillaSimoncelli):
    """Model for measuring texture statistics with cross-channel color statistics.

    This model extends the Portilla-Simoncelli model to capture relationships
    between different color channels.

    Parameters
    ----------
    im_shape: tuple
        The size of the images being processed by the model, should be divisible by 2^n_scales
    scale_ch_covar: float
        Scaling factor for cross-channel covariance statistics
    scale_cor_mag: float
        Scaling factor for magnitude correlation statistics
    scale_cor_real: float
        Scaling factor for real component correlation statistics
    """

    def __init__(
        self,
        im_shape,
        scale_ch_covar: float,
        scale_cor_mag: float,
        scale_cor_real: float,
    ):
        super().__init__(im_shape, n_scales=4, n_orientations=4, spatial_corr_width=9)
        self.scale_ch_covar = scale_ch_covar
        self.scale_cor_mag = scale_cor_mag
        self.scale_cor_real = scale_cor_real

    def forward(self, image: torch.Tensor, scales: list | None = None) -> torch.Tensor:
        """Generate Texture Statistics representation with cross-channel statistics."""
        # Call parent class forward method for base statistics
        base_representations = super().forward(image, scales=scales)

        # Compute cross-channel statistics for the image
        cross_channel_stats_img = self._compute_cross_channel_covariance4D(image)

        # Get pyramid coefficients
        pyr_dict, pyr_coeffs, highpass, _ = self._compute_pyr_coeffs(image)
        mag_pyr_coeffs, real_pyr_coeffs = self._compute_intermediate_representations(
            pyr_coeffs
        )

        # Process magnitude coefficients
        cross_channel_correlation_magnitude = []
        for mag_pyr_coeff in mag_pyr_coeffs:
            cross_channel_correlation_magnitude.append(
                self._compute_cross_channel_correlation5D(mag_pyr_coeff)
            )

        # Process real coefficients
        cross_channel_correlation_real = []
        for real_pyr_coeff in real_pyr_coeffs:
            cross_channel_correlation_real.append(
                self._compute_cross_channel_correlation5D(real_pyr_coeff)
            )

        # Prepare tensors
        cross_channel_correlation_magnitude_t = torch.cat(
            cross_channel_correlation_magnitude, dim=1
        )
        cross_channel_correlation_magnitude_t = cross_channel_correlation_magnitude_t[
            None, ...
        ]

        cross_channel_correlation_real_t = torch.cat(
            cross_channel_correlation_real, dim=1
        )
        cross_channel_correlation_real_t = cross_channel_correlation_real_t[None, ...]

        # Combine all representations with scaling factors
        representation_tensor = torch.cat(
            (
                base_representations,
                self.scale_ch_covar * cross_channel_stats_img,
                self.scale_cor_mag * cross_channel_correlation_magnitude_t,
                self.scale_cor_real * cross_channel_correlation_real_t,
            ),
            dim=-1,
        )

        return representation_tensor

    def _compute_cross_channel_covariance4D(self, image: torch.Tensor) -> torch.Tensor:
        """Compute cross-channel covariance/correlation for input image."""
        # Compute mean across spatial dimensions for each channel
        mean_across_channels = image.mean(dim=(2, 3), keepdim=True)

        # Center the data
        centered_data = image - mean_across_channels

        # Compute covariance matrix
        covariance_matrix = (
            centered_data[:, :, None, :, :] * centered_data[:, None, :, :, :]
        ).mean(dim=(3, 4))

        # Compute standard deviations
        std_devs = torch.sqrt(torch.diagonal(covariance_matrix, dim1=1, dim2=2))

        # Compute correlation matrix
        correlation_matrix = covariance_matrix / (std_devs[:, None] * std_devs[None, :])

        return correlation_matrix

    def _compute_cross_channel_correlation5D(self, input_tensor):
        """Compute correlation across channels for pyramid coefficient tensor."""
        batch, channel, scales, height, width = input_tensor.shape

        # Initialize correlation tensor
        correlation_tensor = torch.zeros((batch, scales, channel, channel))

        # Process each batch and scale
        for b in range(batch):
            for s in range(scales):
                # Reshape tensor for correlation computation
                slice_tensor = input_tensor[b, :, s, :, :].reshape(channel, -1)

                # Compute covariance
                covariance_matrix = torch.cov(slice_tensor)

                # Compute standard deviations
                std_devs = torch.sqrt(torch.diag(covariance_matrix))

                # Compute correlation
                correlation_matrix = covariance_matrix / (
                    std_devs[:, None] * std_devs[None, :]
                )

                # Store result
                correlation_tensor[b, s] = correlation_matrix

        return covariance_matrix  # Return covariance matrix

    def convert_to_dict(self, representation_tensor: torch.Tensor) -> OrderedDict:
        """Convert tensor of statistics to dictionary."""
        n_cross_channel_cov = 27
        rep = super().convert_to_dict(representation_tensor[..., :-n_cross_channel_cov])

        # Extract different types of cross-channel statistics
        n_cross_channel_img = 3
        n_cross_channel_mag = 12
        n_cross_channel_real = 12

        cross_channel_cov_real = representation_tensor[..., -n_cross_channel_real:]
        cross_channel_cov_mag = representation_tensor[
            ..., -n_cross_channel_mag - n_cross_channel_real : -n_cross_channel_real
        ]
        cross_channel_cov_img = representation_tensor[
            ...,
            -n_cross_channel_img
            - n_cross_channel_mag
            - n_cross_channel_real : -n_cross_channel_mag - n_cross_channel_real,
        ]

        # Add to dictionary with color prefix for cross-channel statistics
        rep["color_cross_channel_covariance_img"] = cross_channel_cov_img
        rep["color_cross_channel_covariance_mag"] = cross_channel_cov_mag
        rep["color_cross_channel_covariance_real"] = cross_channel_cov_real

        return rep


class ChannelMetamerCTF(MetamerCTF):
    """Extension of MetamerCTF that tracks per-channel losses during synthesis.

    This class allows tracking of loss values per channel during metamer synthesis,
    which is particularly useful for analyzing convergence behavior in color images.
    """

    def __init__(
        self,
        image: Tensor,
        model: torch.nn.Module,
        loss_function=optim.l2_channelwise,
        range_penalty_lambda: float = 0.1,
        allowed_range: tuple[float, float] = (0, 1),
        initial_image: Tensor | None = None,
        coarse_to_fine: Literal["together", "separate"] = "together",
    ):
        super().__init__(
            image=image,
            model=model,
            loss_function=loss_function,
            range_penalty_lambda=range_penalty_lambda,
            allowed_range=allowed_range,
            initial_image=initial_image,
            coarse_to_fine=coarse_to_fine,
        )
        self._channel_losses = [[] for _ in range(self.image.shape[1])]

    def objective_function(
        self, metamer_representation=None, target_representation=None
    ):
        """Compute per-channel losses plus overall loss."""
        if metamer_representation is None:
            metamer_representation = self.model(self.metamer)
        if target_representation is None:
            target_representation = self.target_representation

        # Compute squared differences per feature
        squared_diff = (metamer_representation - target_representation).pow(2)

        # For each channel, compute mean over its features
        n_channels = len(self._channel_losses)
        features_per_channel = squared_diff.shape[-1] // n_channels

        for i in range(n_channels):
            # Get features for this channel
            start_idx = i * features_per_channel
            end_idx = (i + 1) * features_per_channel
            channel_loss = squared_diff[..., start_idx:end_idx].mean()
            self._channel_losses[i].append(channel_loss.item())

        # Overall loss is mean of all squared differences
        loss = squared_diff.mean()

        # Add range penalty
        range_penalty = optim.penalize_range(self.metamer, self.allowed_range)

        return loss + self.range_penalty_lambda * range_penalty

    @property
    def channel_losses(self):
        """Get the per-channel losses over iterations."""
        return [torch.tensor(losses) for losses in self._channel_losses]


def save_metamer_progress(
    metamer, img_name, output_dir, min_val, max_val, rgb2opc, opc2rgb
):
    """Save metamer synthesis progress and results."""
    # Create directory for this image
    img_output_dir = os.path.join(output_dir, f"metamer_{img_name}")
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(os.path.join(img_output_dir, "progress"), exist_ok=True)

    # Save original image and final metamer in both OPC and RGB spaces
    with torch.no_grad():
        # Get original image and final metamer
        original_img_opc = metamer.image
        final_metamer_opc = metamer.metamer

        # Move everything to CPU for visualization and saving
        original_img_opc_cpu = original_img_opc.cpu()
        final_metamer_opc_cpu = final_metamer_opc.cpu()
        min_val_cpu = min_val.cpu()
        max_val_cpu = max_val.cpu()
        opc2rgb_cpu = opc2rgb.cpu()

        # Convert OPC to RGB for visualization
        original_img_rgb = img_transforms.color_transform_image(
            img_transforms.inverse_rescale(
                original_img_opc_cpu, min_val_cpu, max_val_cpu
            ),
            opc2rgb_cpu,
        )

        final_metamer_rgb = img_transforms.color_transform_image(
            img_transforms.inverse_rescale(
                final_metamer_opc_cpu, min_val_cpu, max_val_cpu
            ),
            opc2rgb_cpu,
        )

        # Rest of the function remains the same...

        # Save as PNG images
        for space, img_tensor in [
            ("opc_original", original_img_opc),
            ("opc_metamer", final_metamer_opc),
            ("rgb_original", original_img_rgb),
            ("rgb_metamer", final_metamer_rgb),
        ]:
            img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()

            # Normalize for display if needed
            if space.startswith("opc"):
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

            plt.figure(figsize=(8, 8))
            plt.imshow(img_np, aspect="equal")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                os.path.join(img_output_dir, f"{space}.png"), bbox_inches="tight"
            )
            plt.close()

    # Save channel losses
    channel_names = (
        ["O₁", "O₂", "O₃"]
        if len(metamer.channel_losses) == 3
        else [f"Ch{i}" for i in range(len(metamer.channel_losses))]
    )
    plt.figure(figsize=(10, 6))
    for i, (losses, name) in enumerate(zip(metamer.channel_losses, channel_names)):
        plt.plot(losses.cpu().numpy(), label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Per-Channel Loss Convergence")
    plt.tight_layout()
    plt.savefig(os.path.join(img_output_dir, "channel_losses.png"))
    plt.close()

    # Save overall loss
    plt.figure(figsize=(10, 6))
    plt.plot(metamer.losses.cpu().numpy())
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.title("Overall Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(img_output_dir, "overall_loss.png"))
    plt.close()

    # Save metamer state for later use
    torch.save(
        {
            "metamer_state": metamer.state_dict()
            if hasattr(metamer, "state_dict")
            else None,
            "metamer_object": metamer,
            "min_val": min_val,
            "max_val": max_val,
            "losses": metamer.losses,
            "channel_losses": [losses.cpu() for losses in metamer.channel_losses],
            "num_iterations": len(metamer.losses),
        },
        os.path.join(img_output_dir, "metamer_state.pt"),
    )

    # Save progress frames (if store_progress was True)
    if hasattr(metamer, "saved_metamer") and len(metamer.saved_metamer) > 0:
        progress_dir = os.path.join(img_output_dir, "progress")

        # Create separate directories for OPC and RGB progress frames
        opc_progress_dir = os.path.join(progress_dir, "opc")
        rgb_progress_dir = os.path.join(progress_dir, "rgb")
        os.makedirs(opc_progress_dir, exist_ok=True)
        os.makedirs(rgb_progress_dir, exist_ok=True)

        # Determine how many frames to save (max 100 evenly spaced frames)
        total_frames = len(metamer.saved_metamer)
        step = max(1, total_frames // 100)

        for i in range(0, total_frames, step):
            with torch.no_grad():
                # Get current metamer state and move to CPU
                metamer_opc = metamer.saved_metamer[i].cpu()

                # Convert to RGB
                metamer_rgb = img_transforms.color_transform_image(
                    img_transforms.inverse_rescale(
                        metamer_opc, min_val_cpu, max_val_cpu
                    ),
                    opc2rgb_cpu,
                )

                # Save OPC version
                img_np_opc = metamer_opc[0].permute(1, 2, 0).numpy()
                # Normalize for display
                img_np_opc = (img_np_opc - img_np_opc.min()) / (
                    img_np_opc.max() - img_np_opc.min()
                )
                plt.figure(figsize=(8, 8))
                plt.imshow(img_np_opc, aspect="equal")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(opc_progress_dir, f"frame_{i:06d}.png"),
                    bbox_inches="tight",
                )
                plt.close()

                # Save RGB version
                img_np_rgb = metamer_rgb[0].permute(1, 2, 0).numpy()
                plt.figure(figsize=(8, 8))
                plt.imshow(img_np_rgb, aspect="equal")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(rgb_progress_dir, f"frame_{i:06d}.png"),
                    bbox_inches="tight",
                )
                plt.close()

    # Save synthesis metadata
    metadata = {
        "image_name": img_name,
        "synthesis_time": time.time(),
        "num_iterations": len(metamer.losses),
        "final_loss": float(metamer.losses[-1].cpu().numpy()),
        "image_shape": list(metamer.image.shape),
        "channel_final_losses": [
            float(losses[-1].cpu().numpy()) for losses in metamer.channel_losses
        ],
    }

    with open(os.path.join(img_output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def get_slurm_info():
    """Collect SLURM job information from environment variables."""
    slurm_vars = {
        "job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
        "partition": os.environ.get("SLURM_JOB_PARTITION", "N/A"),
        "num_nodes": os.environ.get("SLURM_JOB_NUM_NODES", "N/A"),
        "num_cpus": os.environ.get("SLURM_JOB_CPUS_PER_NODE", "N/A"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST", "N/A"),
        "gpu_list": os.environ.get("CUDA_VISIBLE_DEVICES", "N/A"),
        "job_name": os.environ.get("SLURM_JOB_NAME", "N/A"),
        "account": os.environ.get("SLURM_ACCOUNT", "N/A"),
        "submit_time": os.environ.get("SLURM_SUBMIT_TIME", "N/A"),
        "start_time": os.environ.get("SLURM_START_TIME", "N/A"),
        "time_limit": os.environ.get("SLURM_TIME_LIMIT", "N/A"),
    }

    # Add additional GPU info if available
    if torch.cuda.is_available():
        cuda_devices = range(torch.cuda.device_count())
        slurm_vars["gpu_info"] = []
        for device in cuda_devices:
            props = torch.cuda.get_device_properties(device)
            slurm_vars["gpu_info"].append(
                {
                    "name": props.name,
                    "total_memory_GB": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )

    return slurm_vars


def main():
    """Main function to run the metamer synthesis."""
    parser = argparse.ArgumentParser(
        description="Run color metamer synthesis on images"
    )

    # Required arguments
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save results"
    )

    # Optional arguments
    parser.add_argument(
        "--max_iter", type=int, default=5000, help="Maximum iterations for synthesis"
    )
    parser.add_argument(
        "--store_progress",
        type=int,
        default=50,
        help="Store progress every N iterations (set to 0 to disable)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Size to resize images (default: 256x256)",
    )
    parser.add_argument(
        "--coarse_to_fine",
        type=str,
        default="together",
        choices=["together", "separate"],
        help="Coarse-to-fine synthesis mode",
    )
    parser.add_argument(
        "--stop_criterion",
        type=float,
        default=1e-6,
        help="Stopping criterion for loss convergence",
    )
    parser.add_argument(
        "--range_penalty", type=float, default=0.1, help="Range penalty lambda"
    )
    parser.add_argument(
        "--scale_ch_covar",
        type=float,
        default=10.0,
        help="Scaling factor for channel covariance",
    )
    parser.add_argument(
        "--scale_ch_mag",
        type=float,
        default=1.0,
        help="Scaling factor for magnitude correlations",
    )
    parser.add_argument(
        "--scale_ch_real",
        type=float,
        default=1.0,
        help="Scaling factor for real correlations",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Number of images to process (default: all)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    po.tools.set_seed(args.seed)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"metamer_synthesis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Get SLURM job information
    slurm_info = get_slurm_info()

    # Record job start time
    job_start_time = time.time()

    # Save run configuration with SLURM info
    config = vars(args)
    config["slurm"] = slurm_info
    config["hostname"] = os.uname()[1] if hasattr(os, "uname") else "unknown"
    config["start_time"] = datetime.now().isoformat()
    config["python_version"] = os.sys.version
    config["torch_version"] = torch.__version__

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )

    # Load RGB to OPC and OPC to RGB transformations
    rgb2opc = img_transforms.rgb2opc.to(device)
    opc2rgb = img_transforms.opc2rgb.to(device)

    # Get list of image files
    image_files = [
        f
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    # Limit number of images if specified
    if args.num_images is not None:
        image_files = image_files[: args.num_images]

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for idx, img_file in enumerate(image_files):
        print(f"Processing image {idx + 1}/{len(image_files)}: {img_file}")
        img_path = os.path.join(args.input_dir, img_file)
        img_name = os.path.splitext(img_file)[0]

        try:
            # Load and transform image
            pil_img = Image.open(img_path).convert("RGB")
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Convert to OPC space
            img_opc_tensor = img_transforms.color_transform_image(img_tensor, rgb2opc)

            # Rescale to [0,1] and get min/max values for later reconstruction
            img_opc_tensor_rescaled, min_val, max_val = img_transforms.rescale(
                img_opc_tensor
            )

            # Ensure min_val and max_val are on the right device
            min_val = min_val.to(device)
            max_val = max_val.to(device)
            # Create PS model
            model = PortillaSimoncelliCrossChannel(
                img_opc_tensor_rescaled.shape[-2:],
                scale_ch_covar=args.scale_ch_covar,
                scale_cor_mag=args.scale_ch_mag,
                scale_cor_real=args.scale_ch_real,
            ).to(device)
            model.eval()

            # Set up metamer synthesis
            store_progress = args.store_progress if args.store_progress > 0 else False

            # Initialize metamer with noise
            initial_image = (
                torch.rand_like(img_opc_tensor_rescaled) * 0.01
                + img_opc_tensor_rescaled.mean()
            )

            # Create ChannelMetamerCTF instance
            metamer = ChannelMetamerCTF(
                image=img_opc_tensor_rescaled,
                model=model,
                loss_function=optim.l2_channelwise,
                initial_image=initial_image,
                range_penalty_lambda=args.range_penalty,
                coarse_to_fine=args.coarse_to_fine,
            )

            # Run synthesis
            start_time = time.time()
            metamer.synthesize(
                max_iter=args.max_iter,
                store_progress=store_progress,
                change_scale_criterion=None,
                ctf_iters_to_check=4,
                stop_criterion=args.stop_criterion,
            )
            synthesis_time = time.time() - start_time

            print(
                f"Synthesis completed in {synthesis_time:.2f}s ({len(metamer.losses)} iterations)"
            )
            print(f"Final loss: {metamer.losses[-1]:.6e}")

            # Save results for this image
            save_metamer_progress(
                metamer, img_name, output_dir, min_val, max_val, rgb2opc, opc2rgb
            )

        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
            import traceback

            traceback.print_exc()

    # Record job end time and duration
    job_end_time = time.time()
    job_duration = job_end_time - job_start_time

    # Update config with final timing information
    config["end_time"] = datetime.now().isoformat()
    config["duration_seconds"] = job_duration
    config["duration_formatted"] = (
        f"{job_duration // 3600:02.0f}:{(job_duration % 3600) // 60:02.0f}:{job_duration % 60:02.0f}"
    )

    # Save updated config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Processing complete. Results saved to {output_dir}")
    print(f"Total job duration: {config['duration_formatted']}")


if __name__ == "__main__":
    main()
