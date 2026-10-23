from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class DepthExtractorConfig:
    resize_height: int = 720
    resize_width: int = 1280
    use_affine_calibration: bool = False


class DepthExtractor(ABC):
    def __init__(self, config: DepthExtractorConfig, device: str) -> None:
        self.config = config
        self.device = device

    @abstractmethod
    def extract_depth_from_file(self, scene_dir: str, video_path: str) -> torch.Tensor:
        pass

    @abstractmethod
    def extract_depth_from_video(self, video_frames: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def _robust_positive_median_ratio(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        min_samples: int = 10,
    ) -> torch.Tensor:
        """Return a robust median(numerator / denominator).

        Invalid/non-positive samples are discarded and a MAD gate removes
        extreme ratio outliers.  This is the scale-only calibration used for
        metric depth because it is substantially more stable than fitting an
        affine scale+shift on a narrow depth range (for example a table ROI).
        """
        valid = (
            torch.isfinite(numerator)
            & torch.isfinite(denominator)
            & (numerator > 0.0)
            & (denominator > 0.0)
        )

        ratios = numerator[valid] / denominator[valid]
        ratios = ratios[torch.isfinite(ratios) & (ratios > 0.0)]

        if ratios.numel() < min_samples:
            raise RuntimeError(
                f"Too few valid depth ratios: {ratios.numel()} < {min_samples}"
            )

        median_ratio = torch.median(ratios)
        absolute_deviation = torch.abs(ratios - median_ratio)
        mad = torch.median(absolute_deviation)

        if mad > 1e-8:
            robust_sigma = 1.4826 * mad
            inliers = absolute_deviation < 3.5 * robust_sigma
            filtered = ratios[inliers]
            if filtered.numel() >= min_samples:
                ratios = filtered

        return torch.median(ratios)

    def calibrate_depth(
        self,
        extracted_depth: torch.Tensor,
        ground_truth_depth: torch.Tensor,
        calibration_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calibrate the full sequence from the real frame-0 depth.

        Default (recommended here):
            z_real ~= scale * z_pred

        The scale is the robust median of z_real / z_pred over the supplied
        frame-0 calibration mask.  A scale+shift fit is left available for
        compatibility, but scale-only is the default.
        """
        if extracted_depth.ndim == 4:
            extracted_depth = extracted_depth.squeeze(1)

        ground_truth_depth = ground_truth_depth.squeeze()
        predicted_depth_0 = extracted_depth[0]

        valid_mask = (
            calibration_mask.bool()
            & torch.isfinite(predicted_depth_0)
            & torch.isfinite(ground_truth_depth)
            & (predicted_depth_0 > 0.0)
            & (ground_truth_depth > 0.0)
        )

        predicted = predicted_depth_0[valid_mask]
        target = ground_truth_depth[valid_mask]

        if predicted.numel() < 10:
            raise RuntimeError(
                f"Too few valid depth calibration pixels: {predicted.numel()}"
            )

        if self.config.use_affine_calibration:
            matrix = torch.stack(
                [predicted, torch.ones_like(predicted)],
                dim=-1,
            )
            scale_shift = torch.linalg.lstsq(matrix, target).solution
            scale = scale_shift[0]
            shift = scale_shift[1]
        else:
            scale = self._robust_positive_median_ratio(
                numerator=target,
                denominator=predicted,
                min_samples=10,
            )
            shift = torch.tensor(
                0.0,
                device=extracted_depth.device,
                dtype=extracted_depth.dtype,
            )

        calibrated_initial = scale * predicted + shift
        error_before = torch.abs(predicted - target)
        error_after = torch.abs(calibrated_initial - target)

        print()
        print("Depth frame-0 metric calibration:")
        print("  valid pixels:", int(predicted.numel()))
        print("  scale:", float(scale))
        print("  shift:", float(shift))
        print(
            "  median abs error before [mm]:",
            float(torch.median(error_before) * 1000.0),
        )
        print(
            "  median abs error after  [mm]:",
            float(torch.median(error_after) * 1000.0),
        )
        print(
            "  mean abs error before [mm]:",
            float(torch.mean(error_before) * 1000.0),
        )
        print(
            "  mean abs error after  [mm]:",
            float(torch.mean(error_after) * 1000.0),
        )

        calibrated_depth = scale * extracted_depth + shift
        calibrated_depth = torch.where(
            extracted_depth > 0.0,
            calibrated_depth,
            torch.zeros_like(calibrated_depth),
        )
        return calibrated_depth

    def stabilize_depth_with_static_roi(
        self,
        depth_frames: torch.Tensor,
        roi: tuple[int, int, int, int],
        min_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove frame-wise global scale drift using a static image ROI.

        This is intended for a *fixed camera*.  Frame 0 remains unchanged.
        For each later frame t we robustly estimate

            relative_scale[t] = median(D_0 / D_t)

        over a user-provided static region and apply that one scalar to the
        entire frame.  Thus the absolute metric calibration established at
        frame 0 is preserved, while slow/flickering global scale drift in the
        monocular video depth is corrected.

        Args:
            depth_frames: already frame-0-calibrated depths [T,H,W].
            roi: [x_min, y_min, x_max, y_max], with max coordinates used as
                 Python-slice exclusive bounds.
            min_samples: minimum valid ROI pixels per frame.

        Returns:
            stabilized_depth_frames: [T,H,W]
            relative_scales: [T], with relative_scales[0] == 1.
        """
        if depth_frames.ndim != 3:
            raise ValueError(
                f"depth_frames must be [T,H,W], got {tuple(depth_frames.shape)}"
            )

        frame_count, height, width = depth_frames.shape
        x1, y1, x2, y2 = (int(v) for v in roi)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(1, min(x2, width))
        y2 = max(1, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid static calibration ROI after clamping: {(x1, y1, x2, y2)}"
            )

        reference_roi = depth_frames[0, y1:y2, x1:x2]
        relative_scales = torch.ones(
            frame_count,
            dtype=depth_frames.dtype,
            device=depth_frames.device,
        )

        before_errors_mm = []
        after_errors_mm = []

        for t in range(frame_count):
            current_roi = depth_frames[t, y1:y2, x1:x2]

            try:
                relative_scale = self._robust_positive_median_ratio(
                    numerator=reference_roi,
                    denominator=current_roi,
                    min_samples=min_samples,
                )
            except RuntimeError as exc:
                if t == 0:
                    raise
                # A bad generated frame/ROI should not destroy the sequence.
                # Fall back to the previous reliable global scale.
                print(
                    f"[Static ROI] frame {t}: {exc}; "
                    "reusing previous relative scale."
                )
                relative_scale = relative_scales[t - 1]

            relative_scales[t] = relative_scale

            valid = (
                torch.isfinite(reference_roi)
                & torch.isfinite(current_roi)
                & (reference_roi > 0.0)
                & (current_roi > 0.0)
            )
            if valid.any():
                before = torch.abs(current_roi[valid] - reference_roi[valid])
                after = torch.abs(
                    relative_scale * current_roi[valid] - reference_roi[valid]
                )
                before_errors_mm.append(torch.median(before) * 1000.0)
                after_errors_mm.append(torch.median(after) * 1000.0)

        # By construction frame 0 should be exactly 1; force it to avoid tiny
        # floating point deviations.
        relative_scales[0] = 1.0

        stabilized = depth_frames * relative_scales[:, None, None]
        stabilized = torch.where(
            depth_frames > 0.0,
            stabilized,
            torch.zeros_like(stabilized),
        )

        print()
        print("Static-ROI temporal depth stabilization:")
        print("  ROI [x1 y1 x2 y2]:", (x1, y1, x2, y2))
        print("  reference frame: 0")
        print("  relative scale min:", float(relative_scales.min()))
        print("  relative scale median:", float(relative_scales.median()))
        print("  relative scale max:", float(relative_scales.max()))

        if before_errors_mm:
            before_tensor = torch.stack(before_errors_mm)
            after_tensor = torch.stack(after_errors_mm)
            print(
                "  temporal ROI median abs change before [mm], median/max:",
                float(torch.median(before_tensor)),
                "/",
                float(torch.max(before_tensor)),
            )
            print(
                "  temporal ROI median abs change after  [mm], median/max:",
                float(torch.median(after_tensor)),
                "/",
                float(torch.max(after_tensor)),
            )

        return stabilized, relative_scales
