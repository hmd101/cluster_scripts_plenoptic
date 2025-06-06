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
import numpy as np
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
    """Extension of MetamerCTF that tracks per-channel losses during synthesis."""

    def __init__(
        self,
        image: Tensor,
        model: torch.nn.Module,
        loss_function=optim.l2_channelwise,
        range_penalty_lambda: float = 0.1,
        allowed_range: tuple[float, float] = (0, 1),  # default
        initial_image: Tensor | None = None,
        coarse_to_fine: Literal["together", "separate"] = "together",
    ):
        # Initialize parent class first
        super().__init__(
            image=image,
            model=model,
            loss_function=loss_function,
            range_penalty_lambda=range_penalty_lambda,
            allowed_range=allowed_range,
            initial_image=initial_image,
            coarse_to_fine=coarse_to_fine,
        )

        # Initialize tracking variables
        self._channel_losses = [[] for _ in range(self.image.shape[1])]
        self._saved_representation_errors = []
        self._opc_range_stats = []

    def objective_function(
        self, metamer_representation=None, target_representation=None
    ):
        """Compute the per-channel losses and overall loss."""
        # Ensure tracking variable exists
        if not hasattr(self, "_channel_losses") or self._channel_losses is None:
            self._channel_losses = [[] for _ in range(self.image.shape[1])]

        # Get representations if not provided
        if metamer_representation is None:
            metamer_representation = self.model(self.metamer)
        if target_representation is None:
            target_representation = self.target_representation

        # Compute squared differences per feature
        squared_diff = (metamer_representation - target_representation).pow(2)

        # For each channel, compute mean over its features
        n_channels = self.image.shape[1]  # Use image shape directly
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

    def _optimizer_step(
        self, pbar, change_scale_criterion=None, ctf_iters_to_check=None
    ):
        """Override to track representation errors and OPC range stats."""
        # Ensure tracking variables exist
        if not hasattr(self, "_saved_representation_errors"):
            self._saved_representation_errors = []
        if not hasattr(self, "_opc_range_stats"):
            self._opc_range_stats = []

        # Store OPC range stats and representation error if requested
        if self.store_progress and ((len(self.losses) % self.store_progress) == 0):
            with torch.no_grad():
                # Compute detailed OPC range statistics
                metamer_data = self.metamer.detach()

                # Basic statistics
                stats = {
                    "iteration": len(self.losses),
                    "min": metamer_data.min().item(),
                    "max": metamer_data.max().item(),
                    "mean": metamer_data.mean().item(),
                    "std": metamer_data.std().item(),
                    "median": metamer_data.median().item(),
                    # Count values outside the [0,1] range
                    "below_zero": (metamer_data < 0).sum().item(),
                    "above_one": (metamer_data > 1).sum().item(),
                    "total_outside_range": (
                        (metamer_data < 0).sum() + (metamer_data > 1).sum()
                    ).item(),
                    # Percentages outside range
                    "percent_outside_range": (
                        ((metamer_data < 0).sum() + (metamer_data > 1).sum())
                        / metamer_data.numel()
                        * 100
                    ).item(),
                    # Per-channel statistics
                    "channel_mins": [
                        metamer_data[:, c].min().item()
                        for c in range(metamer_data.shape[1])
                    ],
                    "channel_maxs": [
                        metamer_data[:, c].max().item()
                        for c in range(metamer_data.shape[1])
                    ],
                    "channel_means": [
                        metamer_data[:, c].mean().item()
                        for c in range(metamer_data.shape[1])
                    ],
                    "channel_outside_range": [
                        (
                            (
                                (metamer_data[:, c] < 0).sum()
                                + (metamer_data[:, c] > 1).sum()
                            )
                            / metamer_data[:, c].numel()
                            * 100
                        ).item()
                        for c in range(metamer_data.shape[1])
                    ],
                }
                self._opc_range_stats.append(stats)

                # Track representation error
                current_rep = self.model(self.metamer)
                rep_error = current_rep - self.target_representation
                self._saved_representation_errors.append(
                    rep_error.detach().clone().cpu()
                )

        # Call parent method for the optimizer step
        return super()._optimizer_step(pbar, change_scale_criterion, ctf_iters_to_check)

    @property
    def channel_losses(self):
        """Get the per-channel losses over iterations."""
        if not hasattr(self, "_channel_losses") or self._channel_losses is None:
            return []
        return [torch.tensor(losses) for losses in self._channel_losses]

    @property
    def saved_representation_errors(self):
        """Saved representation errors during synthesis."""
        if (
            not hasattr(self, "_saved_representation_errors")
            or not self._saved_representation_errors
        ):
            return None
        return torch.stack(self._saved_representation_errors)

    @property
    def opc_range_stats(self):
        """Statistics about pixel values in OPC space during synthesis."""
        if not hasattr(self, "_opc_range_stats"):
            return []
        return self._opc_range_stats

    def plot_representation_error(
        self,
        batch_idx: int = 0,
        iteration: int | None = None,
        ylim: tuple[float, float] | None | Literal[False] = None,
        ax: plt.Axes | None = None,
        as_rgb: bool = False,
    ) -> list[plt.Axes]:
        """Plot all representation errors including cross-channel statistics."""
        # Get representation error
        if iteration is None:
            # Use the most recent representation error
            with torch.no_grad():
                representation_error = (
                    self.model(self.metamer) - self.target_representation
                )
        elif (
            hasattr(self, "_saved_representation_errors")
            and self._saved_representation_errors
        ):
            representation_error = self._saved_representation_errors[iteration]
            if representation_error.device != self.metamer.device:
                representation_error = representation_error.to(self.metamer.device)
        else:
            # If no saved errors, compute for current state
            with torch.no_grad():
                representation_error = (
                    self.model(self.metamer) - self.target_representation
                )

        # Get the representation dictionary
        rep = {
            k: v[0, 0]
            for k, v in self.model.convert_to_dict(
                representation_error[batch_idx].unsqueeze(0).mean(1, keepdim=True)
            ).items()
        }

        # Organize stats into groups
        stat_groups = {
            "Standard Statistics": [
                "pixel_statistics",
                "var_highpass_residual",
                "std_reconstructed",
                "skew_reconstructed",
                "kurtosis_reconstructed",
            ],
            "Correlation Statistics": [
                "auto_correlation_magnitude",
                "auto_correlation_reconstructed",
                "cross_orientation_correlation_magnitude",
                "cross_scale_correlation_magnitude",
                "cross_scale_correlation_real",
            ],
            "Cross-Channel Statistics": [
                "color_cross_channel_covariance_img",
                "color_cross_channel_covariance_mag",
                "color_cross_channel_covariance_real",
            ],
        }

        # Create figure with subplots
        if ax is None:
            fig = plt.figure(figsize=(15, 10))
        else:
            fig = ax.figure

        # Calculate total number of subplots needed
        available_stats = [
            stat for group in stat_groups.values() for stat in group if stat in rep
        ]
        total_plots = len(available_stats)
        n_rows = (total_plots + 2) // 3  # Round up to nearest multiple of 3
        n_cols = 3
        axes = []
        current_plot = 0

        # Plot each group of statistics
        for group_name, stats in stat_groups.items():
            for stat in stats:
                if stat in rep:
                    ax = fig.add_subplot(n_rows, n_cols, current_plot + 1)
                    # Get data and create stem plot
                    data = rep[stat].flatten()
                    data_np = data.detach().cpu().numpy()

                    # Create stem plot
                    ax.stem(
                        np.arange(len(data_np)),
                        data_np,
                        basefmt="gray",
                        markerfmt="bo",
                        linefmt="b-",
                    )

                    # Set title and styling
                    ax.set_title(f"{stat}", fontsize=8)
                    ax.grid(True, alpha=0.3)
                    if ylim is not None and ylim is not False:
                        ax.set_ylim(ylim)
                    axes.append(ax)
                    current_plot += 1

        plt.tight_layout()
        return axes

    def plot_opc_range_statistics(self, output_dir=None):
        """Plot statistics about pixel values in OPC space during synthesis."""
        if not hasattr(self, "_opc_range_stats") or not self._opc_range_stats:
            raise ValueError("No OPC range statistics available.")

        # Extract data
        iterations = [stat["iteration"] for stat in self._opc_range_stats]
        mins = [stat["min"] for stat in self._opc_range_stats]
        maxs = [stat["max"] for stat in self._opc_range_stats]
        means = [stat["mean"] for stat in self._opc_range_stats]
        percent_outside = [
            stat["percent_outside_range"] for stat in self._opc_range_stats
        ]

        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Plot min, max, mean values
        ax1.plot(iterations, mins, "b-", label="Min")
        ax1.plot(iterations, maxs, "r-", label="Max")
        ax1.plot(iterations, means, "g-", label="Mean")
        ax1.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax1.axhline(y=1, color="k", linestyle="--", alpha=0.5)
        ax1.set_ylabel("pixel Value")
        ax1.set_title("OPC Pixel Value range During Synthesis")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot percentage of pixels outside the [0,1] range
        ax2.plot(iterations, percent_outside, "k-")
        ax2.set_ylabel("% Pixels Outside [0,1] Range")
        ax2.set_title("Percentage of OPC Pixels outside Valid Range")
        ax2.grid(True, alpha=0.3)

        # Plot per-channel statistics
        n_channels = len(self._opc_range_stats[0]["channel_mins"])
        channel_names = (
            ["O1", "O2", "O3"]
            if n_channels == 3
            else [f"Ch{i + 1}" for i in range(n_channels)]
        )

        for c in range(n_channels):
            c_mins = [stat["channel_mins"][c] for stat in self._opc_range_stats]
            c_maxs = [stat["channel_maxs"][c] for stat in self._opc_range_stats]
            c_outside = [
                stat["channel_outside_range"][c] for stat in self._opc_range_stats
            ]

            ax3.plot(iterations, c_outside, label=f"{channel_names[c]}")

        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("% Pixels Outside [0,1] Range")
        ax3.set_title("Per-Channel OPC Range Violations")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if output directory is provided
        if output_dir:
            plt.savefig(os.path.join(output_dir, "opc_range_statistics.png"))
            plt.close()
            return None

        return fig


###############################################################
### Plots
###############################################################


def save_metamer_progress(
    metamer, img_name, output_dir, min_val, max_val, rgb2opc, opc2rgb
):
    """
    Save all outputs from the metamer synthesis process, including visualizations,
    intermediate progress, representation errors, loss curves, and metadata.

    Parameters
    ----------
    metamer : Metamer
        Metamer object containing the synthesized image, loss history, and optionally saved states.
    img_name : str
        Identifier name of the target image used to create directory structure.
    output_dir : str
        Path to the root directory where all outputs should be stored.
    min_val : float or torch.Tensor
        Minimum allowed value for OPC-rescaled images (used for range visualization).
    max_val : float or torch.Tensor
        Maximum allowed value for OPC-rescaled images (used for range visualization).
    rgb2opc : torch.Tensor
        Transformation matrix from RGB to OPC color space (unused in this function but may be saved).
    opc2rgb : torch.Tensor
        Transformation matrix from OPC to RGB color space, used for visualization.

    Notes
    -----
    - Saves PNG images of the original and final metamer in both OPC and RGB spaces.
    - Plots representation errors (final and optionally intermediate).
    - Plots per-channel and overall loss curves.
    - Generates OPC and RGB histograms.
    - Saves intermediate metamer progress frames (if available).
    - Stores all relevant synthesis metadata and model state in disk.
    """

    # Create directory for this image
    img_output_dir = os.path.join(output_dir, f"metamer_{img_name}")
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(os.path.join(img_output_dir, "progress"), exist_ok=True)

    # Create additional directory for OPC analysis
    opc_analysis_dir = os.path.join(img_output_dir, "opc_analysis")
    os.makedirs(opc_analysis_dir, exist_ok=True)

    # Save original image and final metamer in both OPC and RGB spaces
    with torch.no_grad():
        # Get original image and final metamer
        original_img_opc = metamer.image
        final_metamer_opc = metamer.metamer

        # Compute scalar allowed range from min/max tensors
        # allowed_range = (min_val.item(), max_val.item())
        allowed_range = (
            min_val.item() if isinstance(min_val, torch.Tensor) else min_val,
            max_val.item() if isinstance(max_val, torch.Tensor) else max_val,
        )

        # Move everything to CPU for visualization and saving
        original_img_opc_cpu = original_img_opc.cpu()
        final_metamer_opc_cpu = final_metamer_opc.cpu()
        # min_val_cpu = min_val.cpu()
        # max_val_cpu = max_val.cpu()
        opc2rgb_cpu = opc2rgb.cpu()
        allowed_range_cpu = torch.tensor(allowed_range).cpu()

        # Convert OPC to RGB for visualization
        # WITH rescaling
        # original_img_rgb = img_transforms.color_transform_image(
        #     img_transforms.inverse_rescale(
        #         original_img_opc_cpu, min_val_cpu, max_val_cpu
        #     ),
        #     opc2rgb_cpu,
        # )

        # Convert OPC to RGB for visualization
        # WITHOUT rescaling
        original_img_rgb = img_transforms.color_transform_image(
            original_img_opc_cpu,
            # min_val_cpu,
            # max_val_cpu,
            # min_val,
            # max_val,
            opc2rgb_cpu,
        )

        # Convert OPC to RGB for visualization
        # WITH rescaling
        # final_metamer_rgb = img_transforms.color_transform_image(
        #     img_transforms.inverse_rescale(
        #         final_metamer_opc_cpu, min_val_cpu, max_val_cpu
        #     ),
        #     opc2rgb_cpu,
        # )

        # Convert OPC to RGB for visualization
        # WITH rescaling
        final_metamer_rgb = img_transforms.color_transform_image(
            final_metamer_opc_cpu,
            # min_val,
            # max_val,
            opc2rgb_cpu,
        )

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

    # Add representation error plots
    print("Checking for saved representation errors...")
    print(
        f"Has _saved_representation_errors attribute: {hasattr(metamer, '_saved_representation_errors')}"
    )
    if hasattr(metamer, "_saved_representation_errors"):
        print(
            f"Number of saved representation errors: {len(metamer._saved_representation_errors) if metamer._saved_representation_errors else 0}"
        )
        print(
            f"Type of _saved_representation_errors: {type(metamer._saved_representation_errors)}"
        )

    if (
        hasattr(metamer, "_saved_representation_errors")
        and metamer._saved_representation_errors
    ):
        try:
            # Create plot for the final representation error
            print(
                f"Creating representation error plots from {len(metamer._saved_representation_errors)} saved errors"
            )
            plt.figure(figsize=(15, 10))
            axes = metamer.plot_representation_error()
            plt.savefig(os.path.join(img_output_dir, "representation_error_final.png"))
            plt.close()

            # Optionally, create plots for a few intermediate points
            if len(metamer._saved_representation_errors) > 1:
                # Plot at 25%, 50%, and 75% of progress
                points = [
                    len(metamer._saved_representation_errors) // 4,
                    len(metamer._saved_representation_errors) // 2,
                    3 * len(metamer._saved_representation_errors) // 4,
                ]

                for i, point in enumerate(points):
                    plt.figure(figsize=(15, 10))
                    metamer.plot_representation_error(iteration=point)
                    plt.savefig(
                        os.path.join(
                            img_output_dir, f"representation_error_{i + 1}.png"
                        )
                    )
                    plt.close()
        except Exception as e:
            print(
                f"Warning: Failed to create representation error plots from saved data: {e}"
            )
    else:
        # Fallback: Compute representation error directly
        try:
            print(
                "No saved representation errors found. Computing current representation error."
            )
            plt.figure(figsize=(15, 10))
            # Compute representation error directly from current state
            with torch.no_grad():
                current_rep = metamer.model(metamer.metamer)
                rep_error = current_rep - metamer.target_representation

                # Modified plot_representation_error logic for direct usage
                batch_idx = 0
                # Get the representation dictionary
                rep = {
                    k: v[0, 0]
                    for k, v in metamer.model.convert_to_dict(
                        rep_error[batch_idx].unsqueeze(0).mean(1, keepdim=True)
                    ).items()
                }

                # Organize stats into groups
                stat_groups = {
                    "Standard Statistics": [
                        "pixel_statistics",
                        "var_highpass_residual",
                        "std_reconstructed",
                        "skew_reconstructed",
                        "kurtosis_reconstructed",
                    ],
                    "Correlation Statistics": [
                        "auto_correlation_magnitude",
                        "auto_correlation_reconstructed",
                        "cross_orientation_correlation_magnitude",
                        "cross_scale_correlation_magnitude",
                        "cross_scale_correlation_real",
                    ],
                    "Cross-Channel Statistics": [
                        "color_cross_channel_covariance_img",
                        "color_cross_channel_covariance_mag",
                        "color_cross_channel_covariance_real",
                    ],
                }

                # Calculate total number of subplots needed
                total_plots = sum(
                    len(stats)
                    for stats in stat_groups.values()
                    if any(s in rep for s in stats)
                )
                n_rows = (total_plots + 2) // 3  # Round up to nearest multiple of 3
                n_cols = 3
                current_plot = 0

                # Plot each group of statistics
                for group_name, stats in stat_groups.items():
                    for stat in stats:
                        if stat in rep:
                            ax = plt.subplot(n_rows, n_cols, current_plot + 1)
                            # Get data and create stem plot
                            data = rep[stat].flatten()
                            ax.stem(
                                data.detach().cpu().numpy(),
                                basefmt="gray",
                                markerfmt="bo",
                                linefmt="b-",
                            )
                            # Set title and styling
                            ax.set_title(f"{stat}", fontsize=8)
                            ax.grid(True, alpha=0.3)
                            current_plot += 1

                plt.tight_layout()
                plt.savefig(
                    os.path.join(img_output_dir, "representation_error_current.png")
                )
                plt.close()
                print("Successfully created current representation error plot")
        except Exception as e:
            print(f"Warning: Failed to create current representation error plot: {e}")
            import traceback

            traceback.print_exc()  # Print the full exception traceback for debugging

    # Save channel losses
    channel_names = (
        ["O1", "O2", "O3"]
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

    # Add OPC range statistics visualization if available
    if hasattr(metamer, "_opc_range_stats") and metamer._opc_range_stats:
        try:
            # Generate OPC range plots
            metamer.plot_opc_range_statistics(output_dir=img_output_dir)

            # Save the raw statistics as JSON
            # with open(os.path.join(opc_analysis_dir, "opc_range_stats.json"), "w") as f:
            with open(
                os.path.join(opc_analysis_dir, "opc_summary.txt"), "w", encoding="utf-8"
            ) as f:
                json.dump(metamer._opc_range_stats, f, indent=2)

            ###############################################################
            ### OPC and RGB Histograms
            ###############################################################
            print("Generating per-channel OPC and RGB histograms...")

            with torch.no_grad():
                final_metamer_opc = metamer.metamer.detach().cpu()
                original_img_opc = metamer.image.detach().cpu()
                final_metamer_rgb = img_transforms.color_transform_image(
                    final_metamer_opc, opc2rgb_cpu
                ).cpu()
                original_img_rgb = img_transforms.color_transform_image(
                    original_img_opc, opc2rgb_cpu
                ).cpu()

                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                bins = 100

                def plot_channel_hist(
                    ax, tensor, title, channel_names, colors, allowed_range
                ):
                    for i, (name, color) in enumerate(zip(channel_names, colors)):
                        values = tensor[0, i].flatten().numpy()
                        ax.hist(values, bins=bins, color=color, alpha=0.5, label=name)
                    ax.axvline(
                        x=0, color="gray", linestyle="--", linewidth=1, alpha=0.6
                    )
                    ax.axvline(
                        x=1, color="gray", linestyle="--", linewidth=1, alpha=0.6
                    )
                    ax.axvline(
                        x=allowed_range[0],
                        color="red",
                        linestyle="--",
                        linewidth=1,
                        label="Allowed Min",
                    )
                    ax.axvline(
                        x=allowed_range[1],
                        color="red",
                        linestyle="--",
                        linewidth=1,
                        label="Allowed Max",
                    )
                    ax.set_title(title)
                    ax.set_xlabel("Pixel Value")
                    ax.set_ylabel("Frequency")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                # Define channel names and colors
                opc_channels = ["O₁", "O₂", "O₃"]
                opc_colors = [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                ]  # muted blue, orange, green
                rgb_channels = ["R", "G", "B"]
                rgb_colors = ["red", "green", "blue"]

                plot_channel_hist(
                    axes[0, 0],
                    original_img_opc,
                    "Input Image (OPC channels)",
                    opc_channels,
                    opc_colors,
                    allowed_range,
                )
                plot_channel_hist(
                    axes[0, 1],
                    original_img_rgb,
                    "Input Image (RGB channels)",
                    rgb_channels,
                    rgb_colors,
                    (0, 1),
                )
                plot_channel_hist(
                    axes[1, 0],
                    final_metamer_opc,
                    "Metamer Image (OPC channels)",
                    opc_channels,
                    opc_colors,
                    allowed_range,
                )
                plot_channel_hist(
                    axes[1, 1],
                    final_metamer_rgb,
                    "Metamer Image (RGB channels)",
                    rgb_channels,
                    rgb_colors,
                    (0, 1),
                )

                plt.tight_layout()
                plt.savefig(
                    os.path.join(opc_analysis_dir, "opc_rgb_histograms_per_channel.png")
                )
                plt.close()

        except Exception as e:
            print(f"Warning: Failed to generate OPC analysis: {e}")
            import traceback

            traceback.print_exc()

    # Add OPC data to the stored metamer state
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
            # new: Add OPC specific data
            "opc_range_stats": metamer._opc_range_stats
            if hasattr(metamer, "_opc_range_stats")
            else None,
            "opc_final_values": {
                "min": metamer.metamer.min().item(),
                "max": metamer.metamer.max().item(),
                "outside_range_percent": (
                    ((metamer.metamer < 0).sum() + (metamer.metamer > 1).sum())
                    / metamer.metamer.numel()
                    * 100
                ).item(),
            },
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

                # # Convert metamer to RGB
                # ## with rescaling
                # metamer_rgb = img_transforms.color_transform_image(
                #     img_transforms.inverse_rescale(
                #         metamer_opc, min_val_cpu, max_val_cpu
                #     ),
                #     opc2rgb_cpu,
                # )

                # Convert metamer to RGB
                ## WITHOUT rescaling
                metamer_rgb = img_transforms.color_transform_image(
                    metamer_opc,
                    # min_val,
                    # max_val,
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
    """
    Entry point for running color metamer synthesis over a set of input images.

    This script loads and preprocesses input images, performs metamer synthesis using a
    Portilla-Simoncelli-based model extended with cross-channel statistics, and saves all
    outputs including synthesis results, progress visualizations, loss curves, and metadata.

    Command-line Arguments
    ----------------------
    --input_dir : str
        Directory containing input RGB images.
    --output_dir : str
        Directory where synthesis results and logs will be stored.
    --max_iter : int, optional
        Maximum number of optimization iterations (default: 5000).
    --store_progress : int, optional
        Frequency of saving progress frames (set to 0 to disable, default: 50).
    --img_size : int, optional
        Size to which input images are resized (default: 256).
    --coarse_to_fine : {'together', 'separate'}, optional
        Strategy for multi-scale optimization (default: 'together').
    --stop_criterion : float, optional
        Threshold for early stopping based on loss change (default: 1e-6).
    --range_penalty : float, optional
        Penalty weight for pixel values falling outside allowed OPC range.
    --scale_ch_covar : float, optional
        Scaling factor for channel covariance terms (default: 10.0).
    --scale_ch_mag : float, optional
        Scaling factor for magnitude correlations (default: 1.0).
    --scale_ch_real : float, optional
        Scaling factor for real-valued correlations (default: 1.0).
    --num_images : int, optional
        Number of images to process (default: all).
    --device : str, optional
        Compute device (e.g., 'cuda', 'cpu'); if unspecified, auto-detected.
    --seed : int, optional
        Random seed for reproducibility (default: 1).

    Notes
    -----
    - Generates a timestamped output directory containing per-image results.
    - Logs synthesis configurations, SLURM job info (if available), and timing metadata.
    - For each image, performs conversion to opponent color space, initializes a metamer,
      and runs gradient-based optimization.
    - Delegates result saving to `save_metamer_progress`, including visual outputs and serialized state.
    """

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

            # # Rescale to [0,1] and get min/max values for later reconstruction
            # img_opc_tensor_rescaled, min_val, max_val = img_transforms.rescale(
            #     img_opc_tensor
            # )

            # For Dynamic penality range based on image
            max_val = img_opc_tensor.max().item()
            min_val = img_opc_tensor.min().item()

            # Ensure min_val and max_val are on the right device
            # min_val = min_val.to(device)
            # max_val = max_val.to(device)
            # Create PS model
            model = PortillaSimoncelliCrossChannel(
                # img_opc_tensor_rescaled.shape[-2:], # uncomment if rescaling
                img_opc_tensor.shape[-2:],
                scale_ch_covar=args.scale_ch_covar,
                scale_cor_mag=args.scale_ch_mag,
                scale_cor_real=args.scale_ch_real,
            ).to(device)
            model.eval()

            # Set up metamer synthesis
            store_progress = args.store_progress if args.store_progress > 0 else False

            # Initialize metamer with noise: REscaled
            # initial_image = (
            #     torch.rand_like(img_opc_tensor_rescaled) * 0.01
            #     + img_opc_tensor_rescaled.mean()
            # )

            ## comment below if rescaling
            initial_image = (
                torch.rand_like(img_opc_tensor) * 0.01 + img_opc_tensor.mean()
            )

            #  scalar range for dynamic allowed_range

            allowed_range = (min_val, max_val)

            # Create ChannelMetamerCTF instance
            metamer = ChannelMetamerCTF(
                # image=img_opc_tensor_rescaled,
                image=img_opc_tensor,  # comment if rescaling
                model=model,
                loss_function=optim.l2_channelwise,
                initial_image=initial_image,
                allowed_range=allowed_range,  # comment  if allowed range should be default (0,1)
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
