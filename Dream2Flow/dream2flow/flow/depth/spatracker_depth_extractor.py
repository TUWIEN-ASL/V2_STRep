from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from dream2flow.flow.depth.base import DepthExtractor, DepthExtractorConfig


@dataclass
class SpaTrackerDepthExtractorConfig(DepthExtractorConfig):
    pass


class SpaTrackerDepthExtractor(DepthExtractor):
    @staticmethod
    def _add_spatracker_repo_to_path() -> None:
        workspace_root = Path(__file__).resolve().parents[3]

        candidate_roots = [
            workspace_root / "deps" / "SpaTrackerV2",
        ]

        for spatracker_root in candidate_roots:
            if spatracker_root.is_dir():
                spatracker_path = str(spatracker_root)

                if spatracker_path not in sys.path:
                    sys.path.insert(0, spatracker_path)

                return

        expected_locations = ", ".join(
            str(path) for path in candidate_roots
        )

        raise ImportError(
            "Could not find the SpaTrackerV2 repository. "
            "Install it in one of these locations and try again: "
            f"{expected_locations}"
        )

    def __init__(
        self,
        config: SpaTrackerDepthExtractorConfig,
        device: str,
    ) -> None:
        super().__init__(config, device)

        self._add_spatracker_repo_to_path()

        from models.SpaTrackV2.models.predictor import Predictor
        from models.SpaTrackV2.models.utils import get_points_on_a_grid
        from models.SpaTrackV2.models.vggt4track.models.vggt_moe import (
            VGGT4Track,
        )
        from models.SpaTrackV2.models.vggt4track.utils.load_fn import (
            preprocess_image,
        )

        self.device_obj = torch.device(self.device)

        # -------------------------------------------------------------
        # VGGT geometry model
        #
        # This model is needed first. Keep it on the GPU initially.
        # -------------------------------------------------------------
        self.vggt4track_model = VGGT4Track.from_pretrained(
            "Yuxihenry/SpatialTrackerV2_Front"
        )

        self.vggt4track_model.eval()
        self.vggt4track_model.requires_grad_(False)
        self.vggt4track_model.to(self.device_obj)

        # -------------------------------------------------------------
        # SpaTracker tracking model
        #
        # Keep this model on CPU until the VGGT stage is complete.
        # This prevents both large models occupying GPU memory together.
        # -------------------------------------------------------------
        self.tracking_model = Predictor.from_pretrained(
            "Yuxihenry/SpatialTrackerV2-Offline"
        )

        self.tracking_model.spatrack.track_num = 756      
        self.tracking_model.eval()
        self.tracking_model.requires_grad_(False)
        self.tracking_model.to("cpu")

        self._preprocess_image_fn = preprocess_image
        self._grid_point_get_fn = get_points_on_a_grid

    def _autocast_context(self):
        """
        Use BF16 mixed precision on CUDA.

        BF16 substantially lowers activation memory while preserving
        a larger numerical range than FP16.
        """
        if self.device_obj.type == "cuda":
            return torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )

        return nullcontext()

    def _print_cuda_memory(self, label: str) -> None:
        """
        Print current CUDA memory usage for debugging.
        """
        if self.device_obj.type != "cuda":
            return

        allocated_gib = torch.cuda.memory_allocated(
            self.device_obj
        ) / (1024**3)

        reserved_gib = torch.cuda.memory_reserved(
            self.device_obj
        ) / (1024**3)

        peak_gib = torch.cuda.max_memory_allocated(
            self.device_obj
        ) / (1024**3)

        print(
            f"{label}: "
            f"allocated={allocated_gib:.2f} GiB, "
            f"reserved={reserved_gib:.2f} GiB, "
            f"peak={peak_gib:.2f} GiB"
        )

    def extract_depth_from_file(
        self,
        scene_dir: str,
        video_path: str,
    ) -> torch.Tensor:
        resolved_video_path = Path(video_path)

        if not resolved_video_path.is_absolute():
            resolved_video_path = Path(scene_dir) / video_path

        cap = cv2.VideoCapture(str(resolved_video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video file: {resolved_video_path}"
            )

        frames_list = []

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame)
        finally:
            cap.release()

        if not frames_list:
            raise RuntimeError(
                f"No frames were read from video: {resolved_video_path}"
            )

        frames_np = np.stack(frames_list, axis=0)

        video_frames = torch.from_numpy(frames_np)
        video_frames = video_frames.permute(0, 3, 1, 2)

        return self.extract_depth_from_video(video_frames)

    def extract_depth_from_video(
        self,
        video_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate video geometry with VGGT and then refine the depth using
        SpaTracker.

        GPU-memory strategy:

        1. VGGT is placed on GPU while SpaTracker remains on CPU.
        2. VGGT outputs are copied to CPU.
        3. VGGT is moved back to CPU.
        4. SpaTracker is then moved to GPU.
        5. Both forward passes use inference mode and BF16 autocast.
        """

        if video_frames.ndim != 4:
            raise ValueError(
                "video_frames must have shape [T, C, H, W], "
                f"but received {tuple(video_frames.shape)}"
            )

        if self.device_obj.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device_obj)

        # OpenCV frames are uint8. Convert to normalized float32 before
        # applying SpaTracker/VGGT preprocessing.
        video_frames = video_frames.to(dtype=torch.float32)

        # SpaTracker preprocessing returns [T, C, H, W].
        # Add the batch dimension to obtain [1, T, C, H, W].
        video_frames = self._preprocess_image_fn(video_frames)[None]

        video_frames = video_frames.to(
            device=self.device_obj,
            non_blocking=True,
        )


        print(
            "VGGT input:",
            f"shape={tuple(video_frames.shape)},",
            f"dtype={video_frames.dtype},",
            f"device={video_frames.device}",
        )

        # =============================================================
        # Stage 1: VGGT geometry estimation
        # =============================================================

        # Make sure only VGGT occupies GPU model memory during this stage.
        self.tracking_model.to("cpu")

        self.vggt4track_model.to(self.device_obj)
        self.vggt4track_model.eval()

        if self.device_obj.type == "cuda":
            torch.cuda.empty_cache()

        self._print_cuda_memory("Before VGGT")

        geometry_predictions = None

        try:
            with torch.inference_mode():
                with self._autocast_context():
                    geometry_predictions = self.vggt4track_model(
                        video_frames / 255.0
                    )

                # Move the required outputs directly to CPU.
                #
                # Calling float() before numpy() ensures that BF16 tensors
                # are safely converted into NumPy-compatible float32.
                extrs = (
                    geometry_predictions["poses_pred"]
                    .squeeze()
                    .float()
                    .cpu()
                    .numpy()
                )

                intrs = (
                    geometry_predictions["intrs"]
                    .squeeze()
                    .float()
                    .cpu()
                    .numpy()
                )

                depth_tensor = (
                    geometry_predictions["points_map"][..., 2]
                    .squeeze()
                    .float()
                    .cpu()
                    .numpy()
                )

                unc_metric = (
                    geometry_predictions["unc_metric"]
                    .squeeze()
                    .float()
                    .cpu()
                    .numpy()
                    > 0.5
                )

            self._print_cuda_memory("After VGGT")

        finally:
            # Delete the potentially large VGGT prediction dictionary.
            if geometry_predictions is not None:
                del geometry_predictions

            # VGGT is no longer needed for this call.
            # Move its parameters away from the GPU before loading
            # SpaTracker onto the GPU.
            self.vggt4track_model.to("cpu")

            if self.device_obj.type == "cuda":
                torch.cuda.empty_cache()

        self._print_cuda_memory("After moving VGGT to CPU")

        # Remove the batch dimension:
        # [1, T, C, H, W] -> [T, C, H, W].
        video_tensor = video_frames.squeeze(0)

        # The original batched view is no longer needed.
        del video_frames

        frame_h, frame_w = video_tensor.shape[-2:]

        grid_pts = self._grid_point_get_fn(
            10,
            (frame_h, frame_w),
            device="cpu",
        )

        query_xyt = torch.cat(
            [
                torch.zeros_like(grid_pts[:, :, :1]),
                grid_pts,
            ],
            dim=2,
        )[0].numpy()

        # =============================================================
        # Stage 2: SpaTracker refinement
        # =============================================================

        self.tracking_model.to(self.device_obj)
        self.tracking_model.eval()

        if self.device_obj.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device_obj)

        self._print_cuda_memory("Before SpaTracker")

        outputs = None

        try:
            with torch.inference_mode():
                with self._autocast_context():
                    outputs = self.tracking_model.forward(
                        video_tensor,
                        depth=depth_tensor,
                        intrs=intrs,
                        extrs=extrs,
                        queries=query_xyt,
                        fps=1,
                        full_point=False,
                        iters_track=4,
                        query_no_BA=True,
                        fixed_cam=False,
                        stage=1,
                        unc_metric=unc_metric,
                        support_frame=len(video_tensor) - 1,
                        replace_ratio=0.2,
                    )

            self._print_cuda_memory("After SpaTracker")

            point_map = outputs[2]
            conf_depth = outputs[3]

            depth_save = point_map[:, 2, ...].clone()

            depth_save[conf_depth < 0.5] = 0

            depth_save = torch.nn.functional.interpolate(
                depth_save.unsqueeze(1),
                size=(
                    self.config.resize_height,
                    self.config.resize_width,
                ),
                mode="nearest",
            ).squeeze(1)

        finally:
            if outputs is not None:
                del outputs

            del video_tensor

            # Leave the tracker on CPU after finishing, preventing it from
            # keeping GPU memory occupied between calls.
            self.tracking_model.to("cpu")

            if self.device_obj.type == "cuda":
                torch.cuda.empty_cache()

        return depth_save
