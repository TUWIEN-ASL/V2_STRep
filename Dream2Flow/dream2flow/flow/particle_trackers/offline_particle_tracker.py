from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.functional import grid_sample

from dream2flow.camera import CameraCalibration
from dream2flow.flow.geometry import depth_to_world_points
from dream2flow.flow.object_flow import ObjectFlow
from dream2flow.flow.particle_trackers.base import (
    ParticleTracker,
    ParticleTrackingResult,
)


@dataclass
class OfflineParticleTrackerConfig:
    visualize_tracking: bool = False
    pad_value: int = 0
    tracks_leave_trace: int = 30
    visualization_fps: int = 24
    depth_patch_radius: int = 2       # 5x5
    depth_temporal_gate_m: float = 0.08
    min_depth_candidates: int = 3

    # Depth-mask interpolation threshold.
    # The original code used == 1, which can reject valid pixels after
    # bilinear interpolation near object boundaries.
    depth_valid_threshold: float = 0.8

    # Jump rejection.
    enable_jump_filter: bool = True

    # Maximum allowed displacement between adjacent frames.
    max_pixel_step: float = 80.0
    max_3d_step_m: float = 0.08

    # True:
    #   once a particle jumps, invalidate it for all remaining frames.
    #
    # False:
    #   invalidate only the frame where the jump occurred.
    terminate_track_after_jump: bool = False
    save_intermediate_results: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.depth_valid_threshold <= 1.0:
            raise ValueError(
                "depth_valid_threshold must lie in [0, 1]."
            )

        if self.max_pixel_step <= 0.0:
            raise ValueError(
                "max_pixel_step must be positive."
            )

        if self.max_3d_step_m <= 0.0:
            raise ValueError(
                "max_3d_step_m must be positive."
            )


class OfflineParticleTracker(ParticleTracker):
    def __init__(
        self,
        config: OfflineParticleTrackerConfig,
        device: str = "cuda",
    ) -> None:
        self.config = config
        self._device = device

        self._offline_cotracker = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline",
        ).to(self._device)

        self._query_points: torch.Tensor | None = None

    def initialize_tracking(
        self,
        query_points: torch.Tensor,
    ) -> None:
        if (
            query_points.dim() != 2
            or query_points.shape[1] != 3
        ):
            raise ValueError(
                "query_points must have shape (N, 3), "
                f"got {tuple(query_points.shape)}."
            )

        self._query_points = (
            query_points
            .unsqueeze(0)
            .float()
            .to(self._device)
        )

    def _get_tracked_depth_values(
        self,
        depth_images: torch.Tensor,
        valid_estimated_depth_mask: torch.Tensor,
        tracks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample depth values at tracked pixel coordinates.

        Args:
            depth_images:
                Shape (T, 1, H, W).

            valid_estimated_depth_mask:
                Shape (T, 1, H, W).

            tracks:
                Shape (T, N, 2), in pixel coordinates (x, y).

        Returns:
            interpolated_depth_values:
                Shape (T, N).

            interpolated_valid_mask:
                Shape (T, N).
        """
        _, _, height, width = depth_images.shape

        if height < 2 or width < 2:
            raise ValueError(
                "Depth images must have width and height >= 2."
            )

        normalized_tracks = (
            tracks.clone().unsqueeze(1)
        )

        # grid_sample with align_corners=True expects:
        #
        # x_normalized = 2*x/(W-1) - 1
        # y_normalized = 2*y/(H-1) - 1
        normalized_tracks[..., 0] = (
            normalized_tracks[..., 0]
            / float(width - 1)
        ) * 2.0 - 1.0

        normalized_tracks[..., 1] = (
            normalized_tracks[..., 1]
            / float(height - 1)
        ) * 2.0 - 1.0

        data_for_interpolation = torch.cat(
            (
                depth_images,
                valid_estimated_depth_mask.float(),
            ),
            dim=1,
        )

        interpolated_data = grid_sample(
            data_for_interpolation,
            normalized_tracks,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        interpolated_depth_values = (
            interpolated_data[:, 0]
            .squeeze(1)
        )

        interpolated_valid_values = (
            interpolated_data[:, 1]
            .squeeze(1)
        )

        interpolated_valid_mask = (
            interpolated_valid_values
            > self.config.depth_valid_threshold
        )

        return (
            interpolated_depth_values,
            interpolated_valid_mask,
        )

    def _reject_large_jumps(
        self,
        pixel_tracks: torch.Tensor,
        positions_world: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Reject excessive frame-to-frame particle jumps.

        A jump is detected when either:

            pixel displacement > max_pixel_step

        or:

            3D displacement > max_3d_step_m

        The point coordinates are not clamped or modified. The corresponding
        observation is simply marked invalid.

        Args:
            pixel_tracks:
                Shape (T, N, 2).

            positions_world:
                Shape (T, N, 3).

            valid_mask:
                Shape (T, N).

        Returns:
            filtered_valid_mask:
                Shape (T, N).

            jump_mask:
                Combined pixel/3D jump mask, shape (T, N).

            pixel_jump_mask:
                Pixel jump mask, shape (T, N).

            metric_jump_mask:
                3D jump mask, shape (T, N).
        """
        if (
            pixel_tracks.ndim != 3
            or pixel_tracks.shape[-1] != 2
        ):
            raise ValueError(
                "pixel_tracks must have shape (T, N, 2), "
                f"got {tuple(pixel_tracks.shape)}."
            )

        if (
            positions_world.ndim != 3
            or positions_world.shape[-1] != 3
        ):
            raise ValueError(
                "positions_world must have shape (T, N, 3), "
                f"got {tuple(positions_world.shape)}."
            )

        if positions_world.shape[:2] != pixel_tracks.shape[:2]:
            raise ValueError(
                "pixel_tracks and positions_world must have "
                "matching T and N dimensions."
            )

        if valid_mask.shape != pixel_tracks.shape[:2]:
            raise ValueError(
                "valid_mask must have shape "
                f"{tuple(pixel_tracks.shape[:2])}, "
                f"got {tuple(valid_mask.shape)}."
            )

        filtered_valid_mask = (
            valid_mask.bool().clone()
        )

        frame_count, particle_count = (
            filtered_valid_mask.shape
        )

        pixel_jump_mask = torch.zeros(
            (frame_count, particle_count),
            dtype=torch.bool,
            device=pixel_tracks.device,
        )

        metric_jump_mask = torch.zeros(
            (frame_count, particle_count),
            dtype=torch.bool,
            device=pixel_tracks.device,
        )

        if frame_count < 2:
            jump_mask = (
                pixel_jump_mask
                | metric_jump_mask
            )

            return (
                filtered_valid_mask,
                jump_mask,
                pixel_jump_mask,
                metric_jump_mask,
            )

        # Frame-to-frame displacement.
        pixel_displacement = (
            pixel_tracks[1:]
            - pixel_tracks[:-1]
        )

        metric_displacement = (
            positions_world[1:]
            - positions_world[:-1]
        )

        pixel_step = torch.linalg.norm(
            pixel_displacement,
            dim=-1,
        )

        metric_step = torch.linalg.norm(
            metric_displacement,
            dim=-1,
        )

        finite_pixel_pair = (
            torch.isfinite(pixel_tracks[:-1])
            .all(dim=-1)
            & torch.isfinite(pixel_tracks[1:])
            .all(dim=-1)
        )

        finite_metric_pair = (
            torch.isfinite(positions_world[:-1])
            .all(dim=-1)
            & torch.isfinite(positions_world[1:])
            .all(dim=-1)
        )

        pair_valid = (
            filtered_valid_mask[:-1]
            & filtered_valid_mask[1:]
            & finite_pixel_pair
            & finite_metric_pair
        )

        pixel_jump_pair = (
            pair_valid
            & (
                pixel_step
                > float(self.config.max_pixel_step)
            )
        )

        metric_jump_pair = (
            pair_valid
            & (
                metric_step
                > float(self.config.max_3d_step_m)
            )
        )

        # A transition from frame t-1 to t is assigned to frame t.
        pixel_jump_mask[1:] = pixel_jump_pair
        metric_jump_mask[1:] = metric_jump_pair

        jump_mask = (
            pixel_jump_mask
            | metric_jump_mask
        )

        if self.config.terminate_track_after_jump:
            # Once a particle jumps, invalidate it from the first jump
            # frame onward.
            terminated_mask = (
                torch.cumsum(
                    jump_mask.to(torch.int32),
                    dim=0,
                )
                > 0
            )

            filtered_valid_mask &= (
                ~terminated_mask
            )

        else:
            # Only reject the individual jump frames.
            filtered_valid_mask &= (
                ~jump_mask
            )

        return (
            filtered_valid_mask,
            jump_mask,
            pixel_jump_mask,
            metric_jump_mask,
        )

    def track_particles(
        self,
        output_dir: str,
        camera: CameraCalibration,
        camera_name: str,
        rgb_images: torch.Tensor,
        depth_images: torch.Tensor,
        valid_estimated_depth_mask: torch.Tensor,
    ) -> ParticleTrackingResult:
        if self._query_points is None:
            raise RuntimeError(
                "Tracking cannot be initialized "
                "without query points."
            )

        batched_rgb_images = (
            rgb_images.unsqueeze(0)
        )

        if batched_rgb_images.max() <= 1.5:
            batched_rgb_images = batched_rgb_images * 255.0

        (
            predicted_tracks,
            predicted_visibilities,
        ) = self._offline_cotracker(
            video=batched_rgb_images,
            queries=self._query_points,
        )

        non_batched_predicted_tracks = (
            predicted_tracks.squeeze(0)
        )

        non_batched_predicted_visibilities = (
            predicted_visibilities
            .squeeze(0)
            .bool()
        )

        (
            tracked_depth_values,
            valid_track_mask,
        ) = self._get_tracked_depth_values(
            depth_images=depth_images.unsqueeze(1),
            valid_estimated_depth_mask=(
                valid_estimated_depth_mask.unsqueeze(1)
            ),
            tracks=non_batched_predicted_tracks,
        )

        # A point must be valid in both the depth map and CoTracker.
        valid_track_mask = (
            valid_track_mask.bool()
            & non_batched_predicted_visibilities
        )

        intrinsics, extrinsics = (
            camera.get_camera_calibration(
                camera_name
            )
        )

        intrinsics = intrinsics.to(
            depth_images.device
        )

        extrinsics = extrinsics.to(
            depth_images.device
        )

        particle_position_trajectory = (
            depth_to_world_points(
                tracked_depth_values,
                intrinsics,
                extrinsics,
                pixel_coords=(
                    non_batched_predicted_tracks
                ),
            )
        )

        # Reject non-finite reconstructed 3D points.
        valid_track_mask &= torch.isfinite(
            particle_position_trajectory
        ).all(dim=-1)

        if self.config.enable_jump_filter:
            (
                valid_track_mask,
                jump_mask,
                pixel_jump_mask,
                metric_jump_mask,
            ) = self._reject_large_jumps(
                pixel_tracks=(
                    non_batched_predicted_tracks
                ),
                positions_world=(
                    particle_position_trajectory
                ),
                valid_mask=valid_track_mask,
            )

            print(
                "Jump filter:"
            )

            print(
                "  pixel jump samples:",
                int(
                    pixel_jump_mask
                    .sum()
                    .item()
                ),
            )

            print(
                "  3D jump samples:",
                int(
                    metric_jump_mask
                    .sum()
                    .item()
                ),
            )

            print(
                "  combined jump samples:",
                int(
                    jump_mask
                    .sum()
                    .item()
                ),
            )

            print(
                "  affected particles:",
                int(
                    jump_mask
                    .any(dim=0)
                    .sum()
                    .item()
                ),
            )

            print(
                "  valid ratio after filtering:",
                valid_track_mask
                .float()
                .mean(dim=1)
                .detach()
                .cpu(),
            )

            if self.config.save_intermediate_results:
                diagnostics_dir = Path(output_dir)
                diagnostics_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                torch.save(
                    jump_mask.detach().cpu(),
                    diagnostics_dir
                    / "particle_jump_mask.pt",
                )

                torch.save(
                    pixel_jump_mask.detach().cpu(),
                    diagnostics_dir
                    / "particle_pixel_jump_mask.pt",
                )

                torch.save(
                    metric_jump_mask.detach().cpu(),
                    diagnostics_dir
                    / "particle_3d_jump_mask.pt",
                )

                torch.save(
                    valid_track_mask.detach().cpu(),
                    diagnostics_dir
                    / "filtered_track_valid_mask.pt",
                )

        flow = ObjectFlow(
            position=particle_position_trajectory,
            valid_mask=valid_track_mask,
        )

        flow.initialize_color()

        # Use the filtered mask in both visualization and returned result.
        filtered_visibilities = (
            valid_track_mask.unsqueeze(0)
        )

        if self.config.visualize_tracking:
            from cotracker.utils.visualizer import (
                Visualizer,
            )

            visualizer = Visualizer(
                save_dir=str(Path(output_dir)),
                pad_value=self.config.pad_value,
                linewidth=3,
                fps=self.config.visualization_fps,
                mode="rainbow",
                tracks_leave_trace=(
                    self.config.tracks_leave_trace
                ),
            )

            visualizer.visualize(
                batched_rgb_images,
                predicted_tracks,
                filtered_visibilities,
                filename="track_vis_filtered",
            )

            # Also save the original CoTracker visibility for comparison.
            visualizer.visualize(
                batched_rgb_images,
                predicted_tracks,
                predicted_visibilities,
                filename="track_vis_raw",
            )

        return ParticleTrackingResult(
            flow=flow,
            particle_pixel_trajectories=(
                non_batched_predicted_tracks
            ),
            predicted_visibilities=(
                valid_track_mask
            ),
        )
