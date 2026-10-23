from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from dream2flow.flow.generators.base import ObjectFlowGenerator
from dream2flow.flow.depth import DepthExtractor
from dream2flow.flow.object_flow_result import ObjectFlowResult
from dream2flow.flow.geometry import depth_to_world_points
from dream2flow.flow.particle_trackers import OfflineParticleTracker, OfflineParticleTrackerConfig
from dream2flow.flow.region_selectors import GroundedRegionSelector, GroundedRegionSelectorConfig
from dream2flow.camera import CameraCalibration


@dataclass
class VideoFlowGeneratorConfig:
    save_intermediate_results: bool = True
    particle_ratio: float = 0.05
    max_num_particles: int = 512
    # Keep initial CoTracker query points away from the SAM2 silhouette.
    # An erosion of 5 pixels corresponds to an 11x11 erosion kernel.
    mask_erosion_pixels: int = 5

    # Trim CoTracker QUERY CANDIDATES only along the vertical image axis.
    # 0.10 removes the top 10% and bottom 10% of the eroded object's image-y
    # extent. IMPORTANT: this does NOT modify the object mask used for depth
    # calibration.
    vertical_trim_fraction: float = 0.10

    # Fixed-camera static region used only to remove frame-wise global depth
    # scale drift.  Absolute metric scale still comes from frame-0 RealSense
    # calibration on the object interior.  Coordinates are [x1,y1,x2,y2].
    static_calibration_roi: tuple[int, int, int, int] | None = None

    tracker: OfflineParticleTrackerConfig = field(default_factory=OfflineParticleTrackerConfig)
    region_selector: GroundedRegionSelectorConfig = field(default_factory=GroundedRegionSelectorConfig)


class VideoFlowGenerator(ObjectFlowGenerator):
    """
    Generates 3D object flow using the same high-level sequence as the
    reference video-based particle trajectory generator.
    """

    def __init__(
        self,
        depth_extractor: DepthExtractor,
        device: str = "cpu",
        config: VideoFlowGeneratorConfig | None = None,
        region_selector: GroundedRegionSelector | None = None,
        particle_tracker: OfflineParticleTracker | None = None,
    ) -> None:
        self.device = device
        self.config = config or VideoFlowGeneratorConfig()
        self._depth_extractor = depth_extractor
        self._region_selector = region_selector or GroundedRegionSelector(self.config.region_selector, device=device)
        self._particle_tracker = particle_tracker or OfflineParticleTracker(self.config.tracker, device=device)

    def _save_rgb_if_needed(self, output_dir: Path, start_image: torch.Tensor) -> None:
        if not self.config.save_intermediate_results:
            return
        rgb_save_file_path = output_dir / "camera_rgb.png"
        if rgb_save_file_path.exists():
            return
        rgb_img = start_image.detach().cpu().numpy()
        cv2.imwrite(str(rgb_save_file_path), cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    def _save_camera_if_needed(self, output_dir: Path, camera: CameraCalibration) -> None:
        if not self.config.save_intermediate_results:
            return
        camera_calibration_info_file_path = output_dir / "camera_calibration_info.json"
        if not camera_calibration_info_file_path.exists():
            camera.save(str(camera_calibration_info_file_path))

    def _erode_mask(self, mask: torch.Tensor, erosion_pixels: int) -> torch.Tensor:
        """Erode a binary mask inward by approximately `erosion_pixels`."""
        if erosion_pixels < 0:
            raise ValueError("erosion_pixels must be >= 0")
        if erosion_pixels == 0:
            return mask.bool()

        kernel_size = 2 * int(erosion_pixels) + 1
        kernel = torch.ones((kernel_size, kernel_size), device=mask.device)
        eroded_mask = torch.nn.functional.conv2d(
            mask.float().unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=erosion_pixels,
        ).squeeze() == kernel.sum()
        return eroded_mask

    def _fps(self, object_points: torch.Tensor, ratio: float) -> torch.Tensor:
        from torch_geometric.nn import fps

        return fps(object_points, ratio=ratio)

    def _get_tracked_particle_image_coordinates(
        self,
        output_dir: Path,
        object_name: str,
        initial_rgb_img_torch: torch.Tensor,
        initial_depth_arr_torch: torch.Tensor,
        valid_estimated_depth_mask: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        object_bbox: tuple[float, float, float, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tracked_object_mask = self._region_selector.extract_region(
            output_dir=str(output_dir),
            image=initial_rgb_img_torch,
            object_name=object_name,
            bbox=object_bbox,
        )
        valid_real_depth_mask = (
            torch.isfinite(initial_depth_arr_torch)
            & (initial_depth_arr_torch > 0)
        )

        # This is the ORIGINAL eroded/valid object mask from the baseline.
        # Keep it unchanged because it is returned below and later used for
        # frame-0 depth scale calibration.
        calibration_object_mask = (
            self._erode_mask(
                tracked_object_mask,
                erosion_pixels=self.config.mask_erosion_pixels,
            )
            & valid_estimated_depth_mask
            & valid_real_depth_mask
        )

        if int(calibration_object_mask.sum().item()) == 0:
            raise RuntimeError(
                "The grounded object mask is empty after erosion and depth filtering."
            )

        # --------------------------------------------------------------
        # Vertical image-space trim for TRACKING ONLY.
        #
        # Do NOT use this trimmed mask for depth calibration.  It is only
        # used to decide which initial query points are sent to CoTracker.
        # Image coordinates follow (x, y), so this trims rows / y / v.
        # --------------------------------------------------------------
        trim = float(self.config.vertical_trim_fraction)
        if not (0.0 <= trim < 0.5):
            raise ValueError(
                f"vertical_trim_fraction must be in [0, 0.5), got {trim}"
            )

        tracking_candidate_mask = calibration_object_mask.clone()

        y_indices = torch.nonzero(
            calibration_object_mask,
            as_tuple=False,
        )[:, 0]
        y_min = int(y_indices.min().item())
        y_max = int(y_indices.max().item())
        object_height_px = y_max - y_min + 1
        trim_pixels = int(round(trim * object_height_px))

        y_keep_min = y_min + trim_pixels
        y_keep_max = y_max - trim_pixels

        if trim_pixels > 0:
            row_ids = torch.arange(
                calibration_object_mask.shape[0],
                device=calibration_object_mask.device,
            ).view(-1, 1)
            vertical_keep_mask = (
                (row_ids >= y_keep_min)
                & (row_ids <= y_keep_max)
            )
            tracking_candidate_mask &= vertical_keep_mask

        if int(tracking_candidate_mask.sum().item()) == 0:
            raise RuntimeError(
                "No tracking candidates remain after vertical image-space trim."
            )

        if self.config.save_intermediate_results:
            # Baseline calibration mask: unchanged by trim.
            eroded_mask_uint8 = (
                calibration_object_mask.detach().cpu().numpy().astype(np.uint8) * 255
            )
            cv2.imwrite(
                str(output_dir / "region_mask_eroded.png"),
                eroded_mask_uint8,
            )

            # Tracking-only candidate mask after image-y trimming.
            trimmed_mask_uint8 = (
                tracking_candidate_mask.detach().cpu().numpy().astype(np.uint8) * 255
            )
            cv2.imwrite(
                str(output_dir / "region_mask_tracking_vertical_trimmed.png"),
                trimmed_mask_uint8,
            )

            print(
                f"Tracking mask erosion: {self.config.mask_erosion_pixels} px; "
                f"calibration-mask pixels: {int(calibration_object_mask.sum().item())}"
            )
            print("Vertical image-space query trim:")
            print(f"  fraction top/bottom: {trim:.3f} each")
            print(f"  eroded-mask y range: {y_min} .. {y_max}")
            print(f"  trim pixels: {trim_pixels} each side")
            print(f"  kept y range: {y_keep_min} .. {y_keep_max}")
            print(
                "  tracking-candidate pixels before/after: "
                f"{int(calibration_object_mask.sum().item())} / "
                f"{int(tracking_candidate_mask.sum().item())}"
            )

        # FPS and CoTracker query selection use ONLY the trimmed candidate mask.
        mask_indices = torch.nonzero(tracking_candidate_mask)
        mask_pixel_coords = torch.stack(
            (mask_indices[:, 1], mask_indices[:, 0]),
            dim=1,
        )
        object_points = depth_to_world_points(
            depth=initial_depth_arr_torch[tracking_candidate_mask].unsqueeze(0),
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            pixel_coords=mask_pixel_coords.unsqueeze(0),
        ).squeeze(0)

        particle_ratio = min(
            self.config.particle_ratio,
            self.config.max_num_particles / object_points.shape[0],
        )
        kept_object_points_idx = self._fps(object_points, ratio=particle_ratio)
        kept_object_pixel_coords = mask_pixel_coords[kept_object_points_idx]

        if self.config.save_intermediate_results:
            # Save an RGB overlay of the ACTUAL query points for easy checking.
            rgb = initial_rgb_img_torch.detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for xy in kept_object_pixel_coords.detach().cpu().numpy():
                x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
                cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
            cv2.imwrite(
                str(output_dir / "tracking_query_points_vertical_trimmed.png"),
                overlay,
            )
            print(
                "  final CoTracker query points: "
                f"{int(kept_object_pixel_coords.shape[0])}"
            )

        # CRITICAL: return the UNTRIMMED calibration mask, exactly as in the
        # baseline, so frame-0 depth calibration is unchanged.
        return kept_object_pixel_coords, calibration_object_mask

    def _get_calibrated_depth_frames(
        self,
        depth_frames: torch.Tensor,
        initial_depth_arr_torch: torch.Tensor,
        object_mask: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Absolute frame-0 metric calibration + fixed-camera drift removal.

        1) Robust scale-only frame-0 calibration against the real initial depth
           on the eroded object interior (Dream2Flow/NovaFlow-style metric
           anchoring, but using a robust median ratio).

        2) If `static_calibration_roi` is set, estimate a *relative* scale for
           every video frame so that the static ROI matches calibrated frame 0.
           This preserves the absolute object scale from step 1 while removing
           temporal global-scale drift visible on the table.
        """
        calibration_mask = (
            object_mask
            & torch.isfinite(initial_depth_arr_torch)
            & (initial_depth_arr_torch > 0.0)
            & torch.isfinite(depth_frames[0])
            & (depth_frames[0] > 0.0)
        )

        if depth_valid_mask is not None:
            calibration_mask &= depth_valid_mask

        calibrated = self._depth_extractor.calibrate_depth(
            extracted_depth=depth_frames,
            ground_truth_depth=initial_depth_arr_torch,
            calibration_mask=calibration_mask,
        )

        relative_scales = None
        if self.config.static_calibration_roi is not None:
            calibrated, relative_scales = (
                self._depth_extractor.stabilize_depth_with_static_roi(
                    depth_frames=calibrated,
                    roi=self.config.static_calibration_roi,
                )
            )

        return calibrated, relative_scales


    @staticmethod
    def _sample_depth_along_tracks(
        depth_frames: torch.Tensor,
        pixel_trajectories: torch.Tensor,
    ) -> torch.Tensor:
        """Sample one depth value for every tracked pixel at every frame.

        This intentionally matches OfflineParticleTracker._get_tracked_depth_values
        so the corrected flow is reconstructed from the same depth samples that
        produced the original Dream2Flow ObjectFlow.

        Args:
            depth_frames: [T, H, W]
            pixel_trajectories: [T, N, 2] in (x, y) pixel coordinates

        Returns:
            sampled_depth: [T, N]
        """
        if depth_frames.ndim != 3:
            raise ValueError(
                f"depth_frames must have shape [T,H,W], got {tuple(depth_frames.shape)}"
            )
        if pixel_trajectories.ndim != 3 or pixel_trajectories.shape[-1] != 2:
            raise ValueError(
                "pixel_trajectories must have shape [T,N,2], got "
                f"{tuple(pixel_trajectories.shape)}"
            )
        if depth_frames.shape[0] != pixel_trajectories.shape[0]:
            raise ValueError(
                "Depth-frame count and trajectory-frame count must match: "
                f"{depth_frames.shape[0]} vs {pixel_trajectories.shape[0]}"
            )

        _, height, width = depth_frames.shape
        normalized_tracks = pixel_trajectories.clone().unsqueeze(1)

        # Match Dream2Flow's OfflineParticleTracker exactly.
        # normalized_tracks[..., 0] = (
        #     normalized_tracks[..., 0] / width
        # ) * 2 - 1
        # normalized_tracks[..., 1] = (
        #     normalized_tracks[..., 1] / height
        # ) * 2 - 1

        normalized_tracks[..., 0] = (
            normalized_tracks[..., 0] / (width - 1)
        ) * 2 - 1

        normalized_tracks[..., 1] = (
            normalized_tracks[..., 1] / (height - 1)
        ) * 2 - 1

        sampled = torch.nn.functional.grid_sample(
            depth_frames.unsqueeze(1),
            normalized_tracks,
            align_corners=True,
        )

        # [T, 1, 1, N] -> [T, N]
        return sampled[:, 0, 0, :]


    def generate(
        self,
        output_dir: str,
        start_image: torch.Tensor,
        camera: CameraCalibration,
        camera_name: str,
        instruction: str,
        object_name: str,
        video_frames: torch.Tensor,
        initial_depth: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
        object_bbox: tuple[float, float, float, float] | None = None,
        video_path: str = "rgb.mp4",
        **kwargs,
    ) -> ObjectFlowResult:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        initial_depth = initial_depth.to(dtype=torch.float32, device=self.device)

        self._save_rgb_if_needed(output_dir_path, start_image)
        self._save_camera_if_needed(output_dir_path, camera)

        depth_frames = self._depth_extractor.extract_depth_from_file(output_dir, video_path)
        depth_frames = depth_frames.to(dtype=torch.float32, device=self.device)
        if self.config.save_intermediate_results:
            depth_save_path = output_dir_path / "depth_frames.pt"
            if not depth_save_path.exists():
                torch.save(depth_frames.detach().cpu(), depth_save_path)

        video_frames = video_frames.to(self.device)
        intrinsics, extrinsics = camera.get_camera_calibration(camera_name)
        intrinsics = intrinsics.to(self.device)
        extrinsics = extrinsics.to(self.device)

        tracked_object_pixel_coords, tracked_region_mask = self._get_tracked_particle_image_coordinates(
            output_dir=output_dir_path,
            object_name=object_name,
            initial_rgb_img_torch=start_image,
            initial_depth_arr_torch=initial_depth,
            valid_estimated_depth_mask=(
                (depth_frames[0] > 0)
                & (initial_depth.squeeze() > 0)
            ),
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            object_bbox=object_bbox,
        )

        tracker_query_points = torch.stack(
            (
                torch.zeros_like(tracked_object_pixel_coords[:, 0]),
                tracked_object_pixel_coords[:, 0],
                tracked_object_pixel_coords[:, 1],
            ),
            dim=1,
        )
        self._particle_tracker.initialize_tracking(tracker_query_points)

        calibrated_depth_frames, relative_scales = self._get_calibrated_depth_frames(
            depth_frames=depth_frames,
            initial_depth_arr_torch=initial_depth,
            object_mask=tracked_region_mask,
            depth_valid_mask=depth_valid_mask,
        )

        if self.config.save_intermediate_results:
            torch.save(
                calibrated_depth_frames.detach().cpu(),
                output_dir_path / "depth_frames_stabilized.pt",
            )
            if relative_scales is not None:
                torch.save(
                    relative_scales.detach().cpu(),
                    output_dir_path / "depth_static_roi_relative_scales.pt",
                )

        restructured_video_frames = video_frames.float().permute(0, 3, 1, 2)
        tracking_result = self._particle_tracker.track_particles(
            output_dir=output_dir,
            camera=camera,
            camera_name=camera_name,
            rgb_images=restructured_video_frames,
            depth_images=calibrated_depth_frames,
            valid_estimated_depth_mask=depth_frames > 0,
        )
        flow = tracking_result.flow
        if flow is None:
            raise RuntimeError("Particle tracker did not return a 3D flow")


        # Keep the particle tracker's calibrated/stabilized 3D flow exactly as
        # returned.  Do NOT apply a second per-particle additive depth anchor:
        # it does not correct temporal depth-layer failures and would overwrite
        # the tracker's jump-filtered 3D result.

        tracks_path = output_dir_path / "tracks_2d.pt"
        if self.config.save_intermediate_results and not tracks_path.exists():
            torch.save(tracking_result.particle_pixel_trajectories.detach().cpu(), tracks_path)

        result = ObjectFlowResult(
            flow=flow,
            initial_pixel_coords=tracked_object_pixel_coords,
            region_mask=tracked_region_mask,
            pixel_trajectories=tracking_result.particle_pixel_trajectories,
        )
        return result







    # def generate(
    #     self,
    #     output_dir: str,
    #     start_image: torch.Tensor,
    #     camera: CameraCalibration,
    #     camera_name: str,
    #     instruction: str,
    #     object_name: str,
    #     video_frames: torch.Tensor,
    #     initial_depth: torch.Tensor,
    #     depth_valid_mask: torch.Tensor | None = None,
    #     object_bbox: tuple[float, float, float, float] | None = None,
    #     video_path: str = "rgb.mp4",
    #     **kwargs,
    # ) -> ObjectFlowResult:
    #     output_dir_path = Path(output_dir)
    #     output_dir_path.mkdir(parents=True, exist_ok=True)
    #     initial_depth = initial_depth.to(dtype=torch.float32, device=self.device)

    #     self._save_rgb_if_needed(output_dir_path, start_image)
    #     self._save_camera_if_needed(output_dir_path, camera)

    #     depth_frames = self._depth_extractor.extract_depth_from_file(output_dir, video_path)
    #     depth_frames = depth_frames.to(dtype=torch.float32, device=self.device)
    #     if self.config.save_intermediate_results:
    #         depth_save_path = output_dir_path / "depth_frames.pt"
    #         if not depth_save_path.exists():
    #             torch.save(depth_frames.detach().cpu(), depth_save_path)

    #     video_frames = video_frames.to(self.device)
    #     intrinsics, extrinsics = camera.get_camera_calibration(camera_name)
    #     intrinsics = intrinsics.to(self.device)
    #     extrinsics = extrinsics.to(self.device)

    #     tracked_object_pixel_coords, tracked_region_mask = self._get_tracked_particle_image_coordinates(
    #         output_dir=output_dir_path,
    #         object_name=object_name,
    #         initial_rgb_img_torch=start_image,
    #         initial_depth_arr_torch=initial_depth,
    #         valid_estimated_depth_mask=depth_frames[0] > 0,
    #         camera_intrinsics=intrinsics,
    #         camera_extrinsics=extrinsics,
    #         object_bbox=object_bbox,
    #     )

    #     tracker_query_points = torch.stack(
    #         (
    #             torch.zeros_like(tracked_object_pixel_coords[:, 0]),
    #             tracked_object_pixel_coords[:, 0],
    #             tracked_object_pixel_coords[:, 1],
    #         ),
    #         dim=1,
    #     )
    #     self._particle_tracker.initialize_tracking(tracker_query_points)

    #     calibrated_depth_frames = self._get_calibrated_depth_frames(
    #         depth_frames=depth_frames,
    #         initial_depth_arr_torch=initial_depth,
    #         object_mask=tracked_region_mask,
    #         depth_valid_mask=depth_valid_mask,
    #     )
    #     # calibrated_depth_frames[0] = initial_depth

    #     restructured_video_frames = video_frames.float().permute(0, 3, 1, 2)
    #     tracking_result = self._particle_tracker.track_particles(
    #         output_dir=output_dir,
    #         camera=camera,
    #         camera_name=camera_name,
    #         rgb_images=restructured_video_frames,
    #         depth_images=calibrated_depth_frames,
    #         valid_estimated_depth_mask=depth_frames > 0,
    #     )
    #     flow = tracking_result.flow
    #     if flow is None:
    #         raise RuntimeError("Particle tracker did not return a 3D flow")

    #     tracks_path = output_dir_path / "tracks_2d.pt"
    #     if self.config.save_intermediate_results and not tracks_path.exists():
    #         torch.save(tracking_result.particle_pixel_trajectories.detach().cpu(), tracks_path)

    #     result = ObjectFlowResult(
    #         flow=flow,
    #         initial_pixel_coords=tracked_object_pixel_coords,
    #         region_mask=tracked_region_mask,
    #         pixel_trajectories=tracking_result.particle_pixel_trajectories,
    #     )
    #     return result
