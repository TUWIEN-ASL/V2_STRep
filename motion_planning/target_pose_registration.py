from __future__ import annotations

"""Demo/current rigid-entity registration utilities.

The LoFTR RGB-D path supports two rotation policies:

* ``full_6d``: the original unconstrained rigid registration followed by
  optional full-6DoF trimmed ICP. This remains the default and is used for
  ``object_pose_6d`` targets.
* ``yaw_only``: use the full rigid RANSAC only to reject bad RGB-D LoFTR
  correspondences, then refit the selected correspondences with translation
  plus a single rotation about ``world_up``. Full-6DoF ICP is intentionally
  skipped so that roll/pitch cannot be reintroduced. This is intended for the
  initial controlled-entity registration when the object is placed with the
  same gravity alignment as in the demonstration.

The LoFTR entry point comes first, followed by shared helpers and saved outputs.
"""

from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

Array = np.ndarray
_LOFTR_CACHE: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class CandidateDiagnostics:
    method: str
    accepted: bool
    rejection_reason: Optional[str]
    converged: bool
    iterations: int
    correspondence_count: int
    initial_rms_m: float
    final_rms_m: float
    selected_yaw_hypothesis_deg: float
    source_overlap: float
    target_overlap: float
    forward_trimmed_rms_m: float
    reverse_trimmed_rms_m: float
    bidirectional_trimmed_rms_m: float
    score: float
    ransac_fitness: Optional[float] = None
    ransac_inlier_rmse_m: Optional[float] = None
    icp_fitness: Optional[float] = None
    icp_inlier_rmse_m: Optional[float] = None
    residual_translation_m: Optional[float] = None
    residual_rotation_deg: Optional[float] = None
    world_up_tilt_deg: Optional[float] = None
    rgb_matcher: Optional[str] = None
    rgb_rotation_preprocess_deg: Optional[float] = None
    rgb_raw_match_count: Optional[int] = None
    rgb_masked_match_count: Optional[int] = None
    rgb_valid_depth_match_count: Optional[int] = None
    rgb_ransac_inlier_count: Optional[int] = None
    rgb_ransac_inlier_ratio: Optional[float] = None
    rgb_reprojection_rms_px: Optional[float] = None
    rgb_spatial_coverage: Optional[float] = None


@dataclass(frozen=True)
class RegistrationDiagnostics:
    method: str
    accepted: bool
    fallback_used: bool
    selected_candidate_method: str
    converged: bool
    iterations: int
    correspondence_count: int
    initial_rms_m: float
    final_rms_m: float
    selected_yaw_hypothesis_deg: float
    source_overlap: float
    target_overlap: float
    bidirectional_trimmed_rms_m: float
    max_correspondence_distance_m: float
    trim_fraction: float
    minimum_correspondences: int
    minimum_source_overlap: float
    minimum_target_overlap: float
    maximum_accepted_rms_m: float
    maximum_bidirectional_rms_m: float
    demo_point_count: int
    current_point_count: int
    candidates: tuple[CandidateDiagnostics, ...]


@dataclass(frozen=True)
class RegisteredTargetPoses:
    transform_world_target_demo: Array
    transform_world_target_current: Array
    transform_world_target_current_pca_diagnostic: Array
    demo_points_world: Array
    current_points_world: Array
    demo_points_target: Array
    registered_demo_points_world: Array
    demo_mask: Array
    current_mask: Array
    demo_pca_eigenvalues_ascending: Array
    current_pca_eigenvalues_ascending: Array
    registration: RegistrationDiagnostics
    rgb_match_demo_pixels: Array
    rgb_match_current_pixels: Array
    rgb_match_confidences: Array
    rgb_match_inlier_mask: Array


@dataclass(frozen=True)
class _RGBDView:
    rgb: Array
    depth: Array
    mask: Array
    bbox: Sequence[float]
    intrinsic_matrix: Array
    transform_world_camera: Array
    depth_scale_to_m: float


@dataclass(frozen=True)
class _RGBDMatches:
    demo_world: Array
    current_world: Array
    demo_pixels: Array
    current_pixels: Array
    confidences: Array
    matcher_name: str
    raw_count: int
    masked_count: int


# Current pipeline: LoFTR relative registration.


def estimate_registered_entity_poses_from_loftr(
    *,
    demo_rgb: Array,
    demo_depth: Array,
    demo_bbox: Sequence[float],
    demo_intrinsic_matrix: Array,
    transform_world_camera_demo: Array,
    demo_depth_scale_to_m: float,
    current_rgb: Array,
    current_depth: Array,
    current_bbox: Sequence[float],
    current_intrinsic_matrix: Array,
    transform_world_camera_current: Array,
    current_depth_scale_to_m: float,
    demo_entity_pose_world: Optional[Array] = None,
    demo_frame_transform: Optional[Callable[[Array], Array]] = None,
    sam2_model_path: str = "facebook/sam2-hiera-large",
    device: str = "cuda",
    minimum_depth_m: float = 0.05,
    maximum_depth_m: float = 3.0,
    depth_band_m: float = 0.0,
    point_stride: int = 1,
    loftr_pretrained: str = "indoor",
    loftr_device: str = "cuda",
    loftr_confidence_threshold: float = 0.2,
    loftr_maximum_long_side: int = 1024,
    loftr_crop_padding_px: int = 0,
    loftr_maximum_matches: int = 500,
    depth_policy: str = "min",
    depth_search_radius_px: int = 2,
    ransac_enabled: bool = True,
    ransac_3d_inlier_threshold_m: float = 0.02,
    ransac_iterations: int = 5000,
    ransac_seed: int = 0,
    ransac_minimum_inliers: int = 3,
    ransac_minimum_inlier_ratio: float = 0.0,
    mask_erode_px: int = 2,
    icp_refine: bool = True,
    icp_voxel_size_m: float = 0.005,
    icp_overlap_fraction: float = 0.65,
    icp_coarse_max_correspondence_distance_m: float = 0.05,
    icp_fine_max_correspondence_distance_m: float = 0.015,
    icp_coarse_max_iterations: int = 60,
    icp_fine_max_iterations: int = 100,
    icp_stop_rms_change_m: float = 1e-5,
    icp_accept_only_if_improved: bool = False,
    validation_max_correspondence_distance_m: float = 0.03,
    validation_trim_fraction: float = 0.65,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
    rotation_constraint: str = "full_6d",
    segmenter: Optional[Sam2BoxSegmenter] = None,
) -> RegisteredTargetPoses:
    """Estimate demo->current registration from LoFTR RGB-D pairs.

    ``rotation_constraint='full_6d'`` preserves the original behavior:
    bbox-crop LoFTR -> RGB-D -> rigid RANSAC/Kabsch -> optional full-6DoF ICP.

    ``demo_frame_transform`` optionally selects source axes after registration,
    before transferring the source frame and expressing its cloud locally.

    ``rotation_constraint='yaw_only'`` uses the full rigid RANSAC only for
    correspondence rejection, then refits those inliers with translation plus
    rotation about ``world_up``. Full-6DoF ICP is skipped in this mode so that
    spurious roll/pitch cannot tilt the transferred controlled-object trajectory.
    """
    rotation_constraint = _rotation_constraint_name(rotation_constraint)

    demo_rgb = np.asarray(demo_rgb, dtype=np.uint8)
    current_rgb = np.asarray(current_rgb, dtype=np.uint8)
    demo_depth = resize_depth_to_rgb(demo_depth, demo_rgb.shape[:2])
    current_depth = resize_depth_to_rgb(current_depth, current_rgb.shape[:2])

    if segmenter is None:
        segmenter = Sam2BoxSegmenter(sam2_model_path, device)
    demo_mask = segmenter.segment(demo_rgb, demo_bbox)
    current_mask = segmenter.segment(current_rgb, current_bbox)

    # Full segmented clouds are kept for outputs / canonical-frame diagnostics.
    demo_points_world, current_points_world = _masked_cloud_pair(
        (demo_depth, demo_mask, demo_intrinsic_matrix,
         transform_world_camera_demo, demo_depth_scale_to_m),
        (current_depth, current_mask, current_intrinsic_matrix,
         transform_world_camera_current, current_depth_scale_to_m),
        minimum_depth_m=minimum_depth_m, maximum_depth_m=maximum_depth_m,
        depth_band_m=depth_band_m, point_stride=point_stride,
    )
    if len(demo_points_world) < 3 or len(current_points_world) < 3:
        raise RuntimeError(
            f"Segmented RGB-D cloud too small: demo={len(demo_points_world)}, "
            f"current={len(current_points_world)}."
        )

    demo_view = _RGBDView(
        demo_rgb, demo_depth, demo_mask, demo_bbox,
        demo_intrinsic_matrix, transform_world_camera_demo, demo_depth_scale_to_m,
    )
    current_view = _RGBDView(
        current_rgb, current_depth, current_mask, current_bbox,
        current_intrinsic_matrix, transform_world_camera_current, current_depth_scale_to_m,
    )
    matches = _prepare_rgbd_correspondences(
        demo_view, current_view,
        loftr_pretrained=loftr_pretrained,
        loftr_device=loftr_device,
        loftr_confidence_threshold=loftr_confidence_threshold,
        loftr_maximum_long_side=loftr_maximum_long_side,
        loftr_crop_padding_px=loftr_crop_padding_px,
        loftr_maximum_matches=loftr_maximum_matches,
        minimum_depth_m=minimum_depth_m,
        maximum_depth_m=maximum_depth_m,
        depth_policy=depth_policy,
        depth_search_radius_px=depth_search_radius_px,
    )

    sparse_relative_world, inlier_mask, sparse_rms, selected_yaw_deg = (
        _fit_sparse_relative_pose(
            matches,
            ransac_enabled=ransac_enabled,
            ransac_3d_inlier_threshold_m=ransac_3d_inlier_threshold_m,
            ransac_iterations=ransac_iterations,
            ransac_seed=ransac_seed,
            ransac_minimum_inliers=ransac_minimum_inliers,
            ransac_minimum_inlier_ratio=ransac_minimum_inlier_ratio,
            rotation_constraint=rotation_constraint,
            world_up=world_up,
        )
    )

    # Dense ICP follows the standalone feasibility script: erode masks first.
    demo_icp_mask = _erode_mask_for_icp(demo_mask, mask_erode_px)
    current_icp_mask = _erode_mask_for_icp(current_mask, mask_erode_px)
    demo_dense_world, current_dense_world = _masked_cloud_pair(
        (demo_depth, demo_icp_mask, demo_intrinsic_matrix,
         transform_world_camera_demo, demo_depth_scale_to_m),
        (current_depth, current_icp_mask, current_intrinsic_matrix,
         transform_world_camera_current, current_depth_scale_to_m),
        minimum_depth_m=minimum_depth_m, maximum_depth_m=maximum_depth_m,
        depth_band_m=depth_band_m, point_stride=point_stride,
    )
    demo_dense_icp = _semantic_voxel_downsample(demo_dense_world, icp_voxel_size_m)
    current_dense_icp = _semantic_voxel_downsample(current_dense_world, icp_voxel_size_m)
    if len(demo_dense_icp) < 3 or len(current_dense_icp) < 3:
        raise RuntimeError(
            f"Dense ICP cloud too small after mask erosion/voxelization: "
            f"demo={len(demo_dense_icp)}, current={len(current_dense_icp)}."
        )

    # Keep dense validation in yaw-only mode, but never apply full-6DoF ICP.
    final_relative_world, coarse_history, fine_history, final_dense_score = (
        _refine_relative_pose(
            demo_dense_icp, current_dense_icp, sparse_relative_world,
            enabled=icp_refine and rotation_constraint == "full_6d",
            stages=(
                (icp_coarse_max_correspondence_distance_m, icp_coarse_max_iterations),
                (icp_fine_max_correspondence_distance_m, icp_fine_max_iterations),
            ),
            overlap_fraction=icp_overlap_fraction,
            stop_rms_change_m=icp_stop_rms_change_m,
            accept_only_if_improved=icp_accept_only_if_improved,
            score_distance_m=validation_max_correspondence_distance_m,
            score_trim_fraction=validation_trim_fraction,
        )
    )

    demo_pca_pose, demo_eigenvalues = compute_pca_target_pose(
        demo_points_world,
        world_up=world_up,
        camera_position_world=np.asarray(transform_world_camera_demo)[:3, 3],
    )
    current_pca_pose, current_eigenvalues = compute_pca_target_pose(
        current_points_world,
        world_up=world_up,
        camera_position_world=np.asarray(transform_world_camera_current)[:3, 3],
    )
    transform_world_entity_demo = (
        np.asarray(demo_entity_pose_world, dtype=np.float64)
        if demo_entity_pose_world is not None
        else demo_pca_pose
    )
    if demo_frame_transform is not None:
        transform_world_entity_demo = demo_frame_transform(transform_world_entity_demo)
    transform_world_entity_current = final_relative_world @ transform_world_entity_demo
    demo_points_entity = world_points_to_target(transform_world_entity_demo, demo_points_world)
    registered_demo_points_world = transform_points(final_relative_world, demo_points_world)

    registration = _build_registration_diagnostics(
        matches, inlier_mask,
        sparse_relative_world=sparse_relative_world,
        final_relative_world=final_relative_world,
        sparse_rms=sparse_rms,
        selected_yaw_deg=selected_yaw_deg,
        rotation_constraint=rotation_constraint,
        coarse_history=coarse_history,
        fine_history=fine_history,
        final_dense_score=final_dense_score,
        demo=demo_view,
        current=current_view,
        loftr_pretrained=loftr_pretrained,
        ransac_minimum_inliers=ransac_minimum_inliers,
        validation_max_correspondence_distance_m=validation_max_correspondence_distance_m,
        validation_trim_fraction=validation_trim_fraction,
        demo_point_count=int(len(demo_points_world)),
        current_point_count=int(len(current_points_world)),
    )

    return RegisteredTargetPoses(
        transform_world_target_demo=transform_world_entity_demo,
        transform_world_target_current=transform_world_entity_current,
        transform_world_target_current_pca_diagnostic=current_pca_pose,
        demo_points_world=demo_points_world,
        current_points_world=current_points_world,
        demo_points_target=demo_points_entity,
        registered_demo_points_world=registered_demo_points_world,
        demo_mask=demo_mask,
        current_mask=current_mask,
        demo_pca_eigenvalues_ascending=demo_eigenvalues,
        current_pca_eigenvalues_ascending=current_eigenvalues,
        registration=registration,
        rgb_match_demo_pixels=matches.demo_pixels,
        rgb_match_current_pixels=matches.current_pixels,
        rgb_match_confidences=matches.confidences,
        rgb_match_inlier_mask=inlier_mask,
    )


def _prepare_rgbd_correspondences(
    demo: _RGBDView,
    current: _RGBDView,
    *,
    loftr_pretrained: str,
    loftr_device: str,
    loftr_confidence_threshold: float,
    loftr_maximum_long_side: int,
    loftr_crop_padding_px: int,
    loftr_maximum_matches: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
    depth_policy: str,
    depth_search_radius_px: int,
) -> _RGBDMatches:
    """Match image crops and retain paired pixels with valid world coordinates."""
    # LoFTR runs on bbox crops of the original RGB images. SAM2 is a post-filter.
    demo_crop, demo_origin, _ = _loftr_bbox_crop(
        demo.rgb, demo.mask, demo.bbox, padding_px=loftr_crop_padding_px
    )
    current_crop, current_origin, _ = _loftr_bbox_crop(
        current.rgb, current.mask, current.bbox, padding_px=loftr_crop_padding_px
    )
    crop_demo_pixels, crop_current_pixels, confidences, matcher_name = _match_loftr(
        demo_crop,
        current_crop,
        device=str(loftr_device),
        pretrained=str(loftr_pretrained),
        maximum_long_side=max(8, int(loftr_maximum_long_side)),
    )
    raw_match_count = int(len(confidences))

    demo_pixels = np.asarray(crop_demo_pixels, dtype=np.float64).copy()
    current_pixels = np.asarray(crop_current_pixels, dtype=np.float64).copy()
    if len(demo_pixels):
        demo_pixels[:, 0] += float(demo_origin[0])
        demo_pixels[:, 1] += float(demo_origin[1])
        current_pixels[:, 0] += float(current_origin[0])
        current_pixels[:, 1] += float(current_origin[1])

    confident = np.asarray(confidences, dtype=np.float64) >= float(
        loftr_confidence_threshold
    )
    demo_pixels = demo_pixels[confident]
    current_pixels = current_pixels[confident]
    confidences = np.asarray(confidences, dtype=np.float64)[confident]

    inside = _mask_contains_pixels(demo.mask, demo_pixels) & _mask_contains_pixels(
        current.mask, current_pixels
    )
    demo_pixels = demo_pixels[inside]
    current_pixels = current_pixels[inside]
    confidences = confidences[inside]

    if int(loftr_maximum_matches) > 0 and len(confidences) > int(loftr_maximum_matches):
        order = np.argsort(-confidences)[: int(loftr_maximum_matches)]
        demo_pixels = demo_pixels[order]
        current_pixels = current_pixels[order]
        confidences = confidences[order]
    masked_match_count = int(len(confidences))
    if masked_match_count < 3:
        raise RuntimeError(
            f"Only {masked_match_count} LoFTR matches remain after confidence + "
            "SAM2 filtering; need at least 3."
        )

    demo_sparse_all: list[Array] = []
    current_sparse_all: list[Array] = []
    depth_demo_pixels: list[list[float]] = []
    depth_current_pixels: list[list[float]] = []
    depth_confidences: list[float] = []

    for demo_pixel, current_pixel, confidence in zip(
        demo_pixels, current_pixels, confidences
    ):
        try:
            demo_z, _ = _semantic_local_depth(
                demo.depth,
                demo.mask,
                demo_pixel,
                depth_scale_to_m=demo.depth_scale_to_m,
                minimum_depth_m=minimum_depth_m,
                maximum_depth_m=maximum_depth_m,
                search_radius_px=depth_search_radius_px,
                policy=depth_policy,
            )
            current_z, _ = _semantic_local_depth(
                current.depth,
                current.mask,
                current_pixel,
                depth_scale_to_m=current.depth_scale_to_m,
                minimum_depth_m=minimum_depth_m,
                maximum_depth_m=maximum_depth_m,
                search_radius_px=depth_search_radius_px,
                policy=depth_policy,
            )
        except ValueError:
            # Automatic matcher: an isolated depth hole should drop this pair,
            # not abort the entire registration.
            continue

        demo_world = _semantic_backproject_original_ray_world(
            demo_pixel,
            demo_z,
            demo.intrinsic_matrix,
            demo.transform_world_camera,
        )
        current_world = _semantic_backproject_original_ray_world(
            current_pixel,
            current_z,
            current.intrinsic_matrix,
            current.transform_world_camera,
        )
        demo_sparse_all.append(demo_world)
        current_sparse_all.append(current_world)
        depth_demo_pixels.append([float(demo_pixel[0]), float(demo_pixel[1])])
        depth_current_pixels.append([float(current_pixel[0]), float(current_pixel[1])])
        depth_confidences.append(float(confidence))

    demo_sparse_all_array = np.asarray(demo_sparse_all, dtype=np.float64).reshape(-1, 3)
    current_sparse_all_array = np.asarray(current_sparse_all, dtype=np.float64).reshape(-1, 3)
    depth_demo_pixels_array = np.asarray(depth_demo_pixels, dtype=np.float64).reshape(-1, 2)
    depth_current_pixels_array = np.asarray(depth_current_pixels, dtype=np.float64).reshape(-1, 2)
    depth_confidences_array = np.asarray(depth_confidences, dtype=np.float64)
    valid_depth_count = int(len(demo_sparse_all_array))
    if valid_depth_count < 3:
        raise RuntimeError(
            f"Only {valid_depth_count} LoFTR pairs have valid RGB-D depth in both images."
        )

    return _RGBDMatches(
        demo_world=demo_sparse_all_array,
        current_world=current_sparse_all_array,
        demo_pixels=depth_demo_pixels_array,
        current_pixels=depth_current_pixels_array,
        confidences=depth_confidences_array,
        matcher_name=matcher_name,
        raw_count=raw_match_count,
        masked_count=masked_match_count,
    )


def _fit_sparse_relative_pose(
    matches: _RGBDMatches,
    *,
    ransac_enabled: bool,
    ransac_3d_inlier_threshold_m: float,
    ransac_iterations: int,
    ransac_seed: int,
    ransac_minimum_inliers: int,
    ransac_minimum_inlier_ratio: float,
    rotation_constraint: str,
    world_up: Sequence[float],
) -> tuple[Array, Array, float, float]:
    """Reject outliers with full rigid RANSAC, then apply the rotation policy."""
    valid_depth_count = int(len(matches.demo_world))
    if ransac_enabled:
        sparse_relative_world, inlier_mask = _rigid_ransac_world(
            matches.demo_world,
            matches.current_world,
            threshold_m=ransac_3d_inlier_threshold_m,
            iterations=ransac_iterations,
            seed=ransac_seed,
        )
    else:
        inlier_mask = np.ones(valid_depth_count, dtype=bool)
        sparse_relative_world = _estimate_rigid_transform_kabsch(
            matches.demo_world, matches.current_world
        )

    inlier_count = int(inlier_mask.sum())
    inlier_ratio = float(inlier_count / max(1, valid_depth_count))
    if inlier_count < int(ransac_minimum_inliers):
        raise RuntimeError(
            f"LoFTR RGB-D RANSAC retained {inlier_count} inliers; "
            f"minimum is {int(ransac_minimum_inliers)}."
        )
    if inlier_ratio < float(ransac_minimum_inlier_ratio):
        raise RuntimeError(
            f"LoFTR RGB-D RANSAC inlier ratio {inlier_ratio:.3f} is below "
            f"{float(ransac_minimum_inlier_ratio):.3f}."
        )

    demo_sparse = matches.demo_world[inlier_mask]
    current_sparse = matches.current_world[inlier_mask]

    # RANSAC above is deliberately full rigid even for yaw_only: it is used as
    # a permissive outlier selector. The final sparse transform is then refit
    # with the requested motion prior.
    if rotation_constraint == "yaw_only":
        sparse_relative_world = _estimate_yaw_only_rigid_transform(
            demo_sparse,
            current_sparse,
            world_up=world_up,
        )
    else:
        sparse_relative_world = _estimate_rigid_transform_kabsch(demo_sparse, current_sparse)

    sparse_prediction = transform_points(sparse_relative_world, demo_sparse)
    sparse_residual = np.linalg.norm(sparse_prediction - current_sparse, axis=1)
    sparse_rms = float(np.sqrt(np.mean(sparse_residual**2)))
    selected_yaw_deg = (
        _signed_yaw_about_world_up_deg(
            sparse_relative_world[:3, :3],
            world_up=world_up,
        )
        if rotation_constraint == "yaw_only"
        else 0.0
    )

    return sparse_relative_world, inlier_mask, sparse_rms, selected_yaw_deg


def _build_registration_diagnostics(
    matches: _RGBDMatches,
    inlier_mask: Array,
    *,
    sparse_relative_world: Array,
    final_relative_world: Array,
    sparse_rms: float,
    selected_yaw_deg: float,
    rotation_constraint: str,
    coarse_history: Sequence[Mapping[str, Any]],
    fine_history: Sequence[Mapping[str, Any]],
    final_dense_score: Mapping[str, float],
    demo: _RGBDView,
    current: _RGBDView,
    loftr_pretrained: str,
    ransac_minimum_inliers: int,
    validation_max_correspondence_distance_m: float,
    validation_trim_fraction: float,
    demo_point_count: int,
    current_point_count: int,
) -> RegistrationDiagnostics:
    """Summarize sparse matches, attempted refinement, and dense validation."""
    valid_depth_count = int(len(matches.demo_world))
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = float(inlier_count / max(1, valid_depth_count))
    if rotation_constraint == "yaw_only":
        method = "loftr_rgbd_ransac+yaw_only_refit"
        final_rms = sparse_rms
    elif np.allclose(final_relative_world, sparse_relative_world):
        method = "loftr_rgbd_ransac_kabsch"
        final_rms = sparse_rms
    else:
        method = "loftr_rgbd_ransac_kabsch+full6d_trimmed_icp"
        final_rms = float(
            fine_history[-1]["rms_m"]
            if fine_history
            else coarse_history[-1]["rms_m"]
            if coarse_history
            else sparse_rms
        )

    projected_sparse, valid_projection = _project_world_points(
        transform_points(sparse_relative_world, matches.demo_world),
        current.intrinsic_matrix,
        current.transform_world_camera,
    )
    projection_mask = inlier_mask & valid_projection
    reprojection_rms_px: Optional[float] = None
    if projection_mask.any():
        reprojection_rms_px = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            projected_sparse[projection_mask]
                            - matches.current_pixels[projection_mask]
                        )
                        ** 2,
                        axis=1,
                    )
                )
            )
        )
    spatial_coverage = _spatial_coverage_from_inliers(
        matches.demo_pixels,
        inlier_mask,
        demo.bbox,
        grid_rows=4,
        grid_cols=4,
    )

    candidate = CandidateDiagnostics(
        method=method,
        accepted=True,
        rejection_reason=None,
        converged=True,
        iterations=int(len(coarse_history) + len(fine_history)),
        correspondence_count=inlier_count,
        initial_rms_m=sparse_rms,
        final_rms_m=final_rms,
        selected_yaw_hypothesis_deg=selected_yaw_deg,
        **_candidate_metric_fields(final_dense_score),
        score=float(final_dense_score["bidirectional_trimmed_rms_m"]),
        rgb_matcher=f"{matches.matcher_name}_{loftr_pretrained}",
        rgb_raw_match_count=matches.raw_count,
        rgb_masked_match_count=matches.masked_count,
        rgb_valid_depth_match_count=valid_depth_count,
        rgb_ransac_inlier_count=inlier_count,
        rgb_ransac_inlier_ratio=inlier_ratio,
        rgb_reprojection_rms_px=reprojection_rms_px,
        rgb_spatial_coverage=spatial_coverage,
    )
    return RegistrationDiagnostics(
        method=(
            "bbox-crop LoFTR -> confidence + SAM2 mask filtering -> local RGB-D "
            + (
                "depth -> 3D RANSAC outlier filtering -> yaw-only refit about world_up"
                if rotation_constraint == "yaw_only"
                else "depth -> 3D RANSAC/Kabsch -> SAM2 dense full-6DoF trimmed ICP"
            )
        ),
        accepted=True,
        fallback_used=False,
        **_candidate_registration_fields(candidate),
        max_correspondence_distance_m=float(validation_max_correspondence_distance_m),
        trim_fraction=float(validation_trim_fraction),
        minimum_correspondences=int(ransac_minimum_inliers),
        minimum_source_overlap=0.0,
        minimum_target_overlap=0.0,
        maximum_accepted_rms_m=float("inf"),
        maximum_bidirectional_rms_m=float("inf"),
        demo_point_count=demo_point_count,
        current_point_count=current_point_count,
        candidates=(candidate,),
    )


# Shared RGB-D inputs, segmentation, coordinates, and fit summaries.


def _open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "object_pose_6d target registration requires Open3D. Install it in "
            "the environment that runs the full pipeline, for example: "
            "pip install open3d"
        ) from error
    return o3d


def normalize(vector: Array, *, name: str = "vector") -> Array:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"{name} has near-zero or invalid norm.")
    return vector / norm


def make_transform(rotation: Array, translation: Array) -> Array:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def transform_points(transform: Array, points: Array) -> Array:
    transform = np.asarray(transform, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation[None, :]


def world_points_to_target(transform_world_target: Array, points_world: Array) -> Array:
    rotation_world_target = transform_world_target[:3, :3]
    translation_world_target = transform_world_target[:3, 3]
    return (
        np.asarray(points_world, dtype=np.float64)
        - translation_world_target[None, :]
    ) @ rotation_world_target


def load_rgb(path: Path) -> Array:
    image = np.asarray(Image.open(path).convert("RGB"))
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB image must have shape (H,W,3), got {image.shape}.")
    return image


def clamp_bbox(bbox: Sequence[float], image_shape: tuple[int, int]) -> Array:
    height, width = image_shape
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = min(max(x1, 0.0), float(width - 1))
    y1 = min(max(y1, 0.0), float(height - 1))
    x2 = min(max(x2, x1 + 1.0), float(width))
    y2 = min(max(y2, y1 + 1.0), float(height))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def keep_largest_component(mask: Array) -> Array:
    binary = np.asarray(mask, dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if component_count <= 1:
        return binary.astype(bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


class Sam2BoxSegmenter:
    def __init__(self, model_path: str, device: str) -> None:
        try:
            from hydra import initialize_config_module
            from hydra.core.global_hydra import GlobalHydra
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as error:
            raise RuntimeError(
                "object_pose_6d target segmentation requires the user's SAM2 "
                "environment (hydra-core and sam2)."
            ) from error

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for SAM2, but torch.cuda.is_available() is False."
            )

        self.device = device
        GlobalHydra.instance().clear()
        initialize_config_module("sam2", version_base="1.2")
        self.predictor = SAM2ImagePredictor.from_pretrained(
            model_path,
            device=device,
        )

    def segment(self, image_rgb: Array, bbox: Sequence[float]) -> Array:
        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        box = clamp_bbox(bbox, image_rgb.shape[:2])
        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            self.predictor.set_image(image_rgb)
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
        mask = keep_largest_component(np.asarray(masks[0], dtype=bool))
        if int(mask.sum()) == 0:
            raise RuntimeError("SAM2 returned an empty target mask.")
        return mask


def resize_depth_to_rgb(depth: Array, image_shape: tuple[int, int]) -> Array:
    height, width = image_shape
    depth = np.asarray(depth, dtype=np.float64)
    if depth.shape == (height, width):
        return depth
    return cv2.resize(
        depth,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


def masked_depth_to_world_points(
    depth: Array,
    mask: Array,
    intrinsic_matrix: Array,
    transform_world_camera: Array,
    *,
    depth_scale_to_m: float,
    minimum_depth_m: float,
    maximum_depth_m: float,
    depth_band_m: float,
    point_stride: int,
) -> Array:
    depth = np.asarray(depth, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError(f"Depth and mask shapes differ: {depth.shape} vs {mask.shape}.")

    depth_m = depth * float(depth_scale_to_m)
    valid = (
        mask
        & np.isfinite(depth_m)
        & (depth_m >= minimum_depth_m)
        & (depth_m <= maximum_depth_m)
    )

    if depth_band_m > 0:
        selected_depths = depth_m[valid]
        if selected_depths.size == 0:
            raise RuntimeError("No valid masked depth is available.")
        median_depth = float(np.median(selected_depths))
        valid &= np.abs(depth_m - median_depth) <= depth_band_m

    rows, cols = np.nonzero(valid)
    stride = max(1, int(point_stride))
    rows = rows[::stride]
    cols = cols[::stride]
    if rows.size < 30:
        raise RuntimeError(
            f"Only {rows.size} valid segmented depth points remain; at least 30 are required."
        )

    z = depth_m[rows, cols]
    intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
    x = (cols.astype(np.float64) - intrinsic_matrix[0, 2]) * z / intrinsic_matrix[0, 0]
    y = (rows.astype(np.float64) - intrinsic_matrix[1, 2]) * z / intrinsic_matrix[1, 1]
    points_camera = np.column_stack((x, y, z))
    return transform_points(transform_world_camera, points_camera)


def numpy_to_point_cloud(points: Array) -> Any:
    o3d = _open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def clean_point_cloud(points: Array) -> Array:
    cloud = numpy_to_point_cloud(points)
    if len(cloud.points) >= 50:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=min(30, len(cloud.points) - 1),
            std_ratio=2.0,
        )
    result = np.asarray(cloud.points, dtype=np.float64)
    if result.shape[0] < 30:
        raise RuntimeError(
            f"Only {result.shape[0]} points remain after outlier filtering."
        )
    return result


def _masked_cloud_pair(
    demo: tuple[Array, Array, Array, Array, float],
    current: tuple[Array, Array, Array, Array, float],
    *,
    minimum_depth_m: float,
    maximum_depth_m: float,
    depth_band_m: float,
    point_stride: int,
) -> tuple[Array, Array]:
    """Build cleaned clouds in demo/current order from depth, mask, K, T, scale."""
    return tuple(
        clean_point_cloud(masked_depth_to_world_points(
            depth, mask, intrinsic, transform,
            depth_scale_to_m=scale,
            minimum_depth_m=minimum_depth_m,
            maximum_depth_m=maximum_depth_m,
            depth_band_m=depth_band_m,
            point_stride=point_stride,
        ))
        for depth, mask, intrinsic, transform, scale in (demo, current)
    )


def compute_pca_target_pose(
    points_world: Array,
    *,
    world_up: Sequence[float],
    camera_position_world: Sequence[float],
) -> tuple[Array, Array]:
    """Compute the user's canonical demo target frame."""
    points = np.asarray(points_world, dtype=np.float64)
    centroid = np.mean(points, axis=0)
    centered = points - centroid[None, :]
    covariance = centered.T @ centered / max(1, points.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    up = normalize(np.asarray(world_up, dtype=np.float64), name="world_up")
    camera_position = np.asarray(camera_position_world, dtype=np.float64)

    z_index = int(np.argmax(np.abs(eigenvectors.T @ up)))
    z_axis = normalize(eigenvectors[:, z_index], name="PCA z-axis")
    if float(np.dot(z_axis, up)) < 0:
        z_axis = -z_axis

    remaining_indices = [index for index in range(3) if index != z_index]
    toward_camera = camera_position - centroid
    toward_camera = toward_camera - float(np.dot(toward_camera, z_axis)) * z_axis
    if np.linalg.norm(toward_camera) < 1e-8:
        fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        toward_camera = fallback - float(np.dot(fallback, z_axis)) * z_axis
    toward_camera = normalize(toward_camera, name="camera direction")

    x_index = max(
        remaining_indices,
        key=lambda index: abs(float(np.dot(eigenvectors[:, index], toward_camera))),
    )
    x_axis = normalize(eigenvectors[:, x_index], name="PCA x-axis")
    if float(np.dot(x_axis, toward_camera)) > 0:
        x_axis = -x_axis

    y_axis = normalize(np.cross(z_axis, x_axis), name="PCA y-axis")
    x_axis = normalize(np.cross(y_axis, z_axis), name="orthogonalized PCA x-axis")
    rotation_world_target = np.column_stack((x_axis, y_axis, z_axis))
    return make_transform(rotation_world_target, centroid), eigenvalues


def _candidate_metric_fields(metrics: Mapping[str, float | int]) -> dict[str, float]:
    """Copy the shared overlap and distance metrics into a candidate."""
    return {
        name: float(metrics[name])
        for name in (
            "source_overlap",
            "target_overlap",
            "forward_trimmed_rms_m",
            "reverse_trimmed_rms_m",
            "bidirectional_trimmed_rms_m",
        )
    }


def _candidate_registration_fields(candidate: CandidateDiagnostics) -> dict[str, Any]:
    """Copy the selected candidate's fit summary without changing its values."""
    return {
        "selected_candidate_method": candidate.method,
        **{
            name: getattr(candidate, name)
            for name in (
                "converged",
                "iterations",
                "correspondence_count",
                "initial_rms_m",
                "final_rms_m",
                "selected_yaw_hypothesis_deg",
                "source_overlap",
                "target_overlap",
                "bidirectional_trimmed_rms_m",
            )
        },
    }


# LoFTR matching, sparse pose fitting, and dense refinement.


def _bbox_crop(
    image: Array,
    mask: Array,
    bbox: Sequence[float],
) -> tuple[Array, Array, tuple[int, int]]:
    x1f, y1f, x2f, y2f = clamp_bbox(bbox, image.shape[:2])
    x1, y1 = int(math.floor(x1f)), int(math.floor(y1f))
    x2, y2 = int(math.ceil(x2f)), int(math.ceil(y2f))
    crop = np.asarray(image[y1:y2, x1:x2]).copy()
    crop_mask = np.asarray(mask[y1:y2, x1:x2], dtype=bool).copy()
    if crop.size == 0 or not crop_mask.any():
        raise RuntimeError("Target crop or crop mask is empty.")
    return crop, crop_mask, (x1, y1)


def _resize_gray_for_matching(
    image_rgb: Array,
    *,
    maximum_long_side: int,
) -> tuple[Array, float, float]:
    """Resize a LoFTR crop while preserving enough resolution for small objects.

    Large crops are only downscaled to ``maximum_long_side`` as before.
    Very small bbox crops are upscaled so that their long side is at least 384
    pixels.  LoFTR coordinates are mapped back to the original crop using the
    returned scale factors, so this does not change the image geometry used by
    the downstream RGB-D registration.
    """
    gray = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    long_side = max(height, width)
    minimum_long_side = min(384, max(8, int(maximum_long_side)))

    if long_side < minimum_long_side:
        scale = float(minimum_long_side) / float(long_side)
    else:
        scale = min(1.0, float(maximum_long_side) / float(long_side))

    resized_width = max(8, int(round(width * scale)))
    resized_height = max(8, int(round(height * scale)))
    resized_width = max(8, (resized_width // 8) * 8)
    resized_height = max(8, (resized_height // 8) * 8)

    resized = cv2.resize(
        gray,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    return (
        resized,
        float(width) / float(resized_width),
        float(height) / float(resized_height),
    )


def _match_loftr(
    image0_rgb: Array,
    image1_rgb: Array,
    *,
    device: str,
    pretrained: str,
    maximum_long_side: int,
) -> tuple[Array, Array, Array, str]:
    try:
        from kornia.feature import LoFTR
    except ImportError as error:
        raise RuntimeError(
            "LoFTR matching requires Kornia. Install it with: pip install kornia"
        ) from error

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for LoFTR but is unavailable.")
    matcher_device = torch.device(device)
    image0, sx0, sy0 = _resize_gray_for_matching(
        image0_rgb, maximum_long_side=maximum_long_side
    )
    image1, sx1, sy1 = _resize_gray_for_matching(
        image1_rgb, maximum_long_side=maximum_long_side
    )
    tensor0 = torch.from_numpy(image0).to(matcher_device, dtype=torch.float32)[None, None] / 255.0
    tensor1 = torch.from_numpy(image1).to(matcher_device, dtype=torch.float32)[None, None] / 255.0
    cache_key = (str(pretrained), str(matcher_device))
    matcher = _LOFTR_CACHE.get(cache_key)
    if matcher is None:
        matcher = LoFTR(pretrained=pretrained).to(matcher_device).eval()
        _LOFTR_CACHE[cache_key] = matcher
    with torch.inference_mode():
        prediction = matcher({"image0": tensor0, "image1": tensor1})
    keypoints0 = prediction["keypoints0"].detach().cpu().numpy().astype(np.float64)
    keypoints1 = prediction["keypoints1"].detach().cpu().numpy().astype(np.float64)
    confidence = prediction["confidence"].detach().cpu().numpy().astype(np.float64)
    keypoints0[:, 0] *= sx0
    keypoints0[:, 1] *= sy0
    keypoints1[:, 0] *= sx1
    keypoints1[:, 1] *= sy1
    return keypoints0, keypoints1, confidence, "loftr"


def _mask_contains_pixels(mask: Array, pixels: Array) -> Array:
    mask = np.asarray(mask, dtype=bool)
    pixels = np.asarray(pixels, dtype=np.float64)
    x = np.rint(pixels[:, 0]).astype(int)
    y = np.rint(pixels[:, 1]).astype(int)
    valid = (
        (x >= 0)
        & (x < mask.shape[1])
        & (y >= 0)
        & (y < mask.shape[0])
    )
    result = np.zeros(pixels.shape[0], dtype=bool)
    indices = np.nonzero(valid)[0]
    result[indices] = mask[y[indices], x[indices]]
    return result


def _estimate_rigid_transform_kabsch(source: Array, target: Array) -> Array:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Kabsch inputs must have matching shape (N,3).")
    if source.shape[0] < 3:
        raise ValueError("At least three correspondences are required.")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return make_transform(rotation, translation)


def _project_world_points(
    points_world: Array,
    intrinsic_matrix: Array,
    transform_world_camera: Array,
) -> tuple[Array, Array]:
    transform_camera_world = np.linalg.inv(np.asarray(transform_world_camera, dtype=np.float64))
    points_camera = transform_points(transform_camera_world, points_world)
    z = points_camera[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    pixels = np.full((points_camera.shape[0], 2), np.nan, dtype=np.float64)
    pixels[valid, 0] = (
        intrinsic_matrix[0, 0] * points_camera[valid, 0] / z[valid]
        + intrinsic_matrix[0, 2]
    )
    pixels[valid, 1] = (
        intrinsic_matrix[1, 1] * points_camera[valid, 1] / z[valid]
        + intrinsic_matrix[1, 2]
    )
    return pixels, valid


def _spatial_coverage_from_inliers(
    pixels: Array,
    inlier_mask: Array,
    bbox: Sequence[float],
    *,
    grid_rows: int,
    grid_cols: int,
) -> float:
    selected = np.asarray(pixels, dtype=np.float64)[np.asarray(inlier_mask, dtype=bool)]
    if selected.shape[0] == 0:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    cols = np.clip(((selected[:, 0] - x1) / width * grid_cols).astype(int), 0, grid_cols - 1)
    rows = np.clip(((selected[:, 1] - y1) / height * grid_rows).astype(int), 0, grid_rows - 1)
    return float(np.unique(rows * grid_cols + cols).size / max(1, grid_rows * grid_cols))


def _semantic_local_depth(
    depth: Array,
    mask: Array,
    pixel: Sequence[float],
    *,
    depth_scale_to_m: float,
    minimum_depth_m: float,
    maximum_depth_m: float,
    search_radius_px: int,
    policy: str,
) -> tuple[float, tuple[int, int]]:
    """Return depth for one semantic pixel without moving its camera ray.

    ``policy="exact"`` uses the clicked/VLM pixel itself.
    ``policy="min"`` uses the minimum valid depth in a tiny RAW-depth window.
    ``policy="median"`` uses the valid depth closest to the local median.

    Important: the SAM2 mask is deliberately NOT used for semantic sparse depth
    lookup. Semantic landmarks are commonly corners/endpoints on segmentation
    boundaries. SAM2 is used only for the dense ICP clouds.

    The returned source pixel only identifies where the depth value came from.
    Back-projection must still use the ORIGINAL semantic pixel.
    """
    depth = np.asarray(depth, dtype=np.float64)
    # Keep mask in the function signature for backward compatibility with the
    # caller, but do not gate sparse landmark depth with it.
    mask = np.asarray(mask, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError(f"Depth and mask shapes differ: {depth.shape} vs {mask.shape}.")

    height, width = depth.shape
    u0 = int(round(float(pixel[0])))
    v0 = int(round(float(pixel[1])))
    if not (0 <= u0 < width and 0 <= v0 < height):
        raise ValueError(f"Semantic pixel {(u0, v0)} is outside image shape {depth.shape}.")

    depth_m = depth * float(depth_scale_to_m)
    policy = str(policy).lower().strip()

    def valid_value(u: int, v: int) -> bool:
        value = float(depth_m[v, u])
        return (
            np.isfinite(value)
            and minimum_depth_m <= value <= maximum_depth_m
        )

    if policy == "exact":
        if not valid_value(u0, v0):
            raise ValueError(
                f"No valid depth at exact semantic pixel {(u0, v0)}."
            )
        return float(depth_m[v0, u0]), (u0, v0)

    if policy not in {"min", "median"}:
        raise ValueError(
            f"Unsupported semantic depth policy {policy!r}; use exact, min, or median."
        )

    radius = max(0, int(search_radius_px))
    x1 = max(0, u0 - radius)
    x2 = min(width, u0 + radius + 1)
    y1 = max(0, v0 - radius)
    y2 = min(height, v0 + radius + 1)

    patch = depth_m[y1:y2, x1:x2]
    valid = (
        np.isfinite(patch)
        & (patch >= minimum_depth_m)
        & (patch <= maximum_depth_m)
    )
    if not bool(valid.any()):
        raise ValueError(
            f"No valid depth within radius {radius} around semantic pixel "
            f"{(u0, v0)}."
        )

    ys, xs = np.nonzero(valid)
    zs = patch[ys, xs].astype(np.float64)
    if policy == "min":
        index = int(np.argmin(zs))
    else:
        median = float(np.median(zs))
        index = int(np.argmin(np.abs(zs - median)))

    source_u = int(x1 + xs[index])
    source_v = int(y1 + ys[index])
    return float(zs[index]), (source_u, source_v)


def _semantic_backproject_original_ray_world(
    pixel: Sequence[float],
    depth_m: float,
    intrinsic_matrix: Array,
    transform_world_camera: Array,
) -> Array:
    """Back-project the ORIGINAL semantic pixel with the selected depth."""
    u = float(pixel[0])
    v = float(pixel[1])
    intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
    z = float(depth_m)
    x = (u - intrinsic_matrix[0, 2]) * z / intrinsic_matrix[0, 0]
    y = (v - intrinsic_matrix[1, 2]) * z / intrinsic_matrix[1, 1]
    point_camera = np.asarray([[x, y, z]], dtype=np.float64)
    return transform_points(transform_world_camera, point_camera)[0]


def _semantic_voxel_downsample(points: Array, voxel_size_m: float) -> Array:
    points = np.asarray(points, dtype=np.float64)
    voxel_size_m = float(voxel_size_m)
    if voxel_size_m <= 0.0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def _semantic_trimmed_icp(
    source_points: Array,
    target_points: Array,
    transform_target_source_initial: Array,
    *,
    overlap_fraction: float,
    maximum_correspondence_distance_m: float,
    maximum_iterations: int,
    stop_rms_change_m: float,
) -> tuple[Array, list[dict[str, Any]]]:
    """General full-6DoF point-to-point trimmed ICP.

    Source is the demo world cloud and target is the current world cloud.
    No axis, yaw, roll, pitch, or translation constraint is imposed.
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    transform = np.asarray(transform_target_source_initial, dtype=np.float64).copy()

    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("source_points must have shape (N,3).")
    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise ValueError("target_points must have shape (N,3).")
    if min(len(source_points), len(target_points)) < 3:
        raise ValueError("Trimmed ICP requires at least three points per cloud.")

    overlap_fraction = float(overlap_fraction)
    if not (0.05 <= overlap_fraction <= 1.0):
        raise ValueError("overlap_fraction must be in [0.05, 1.0].")

    tree = cKDTree(target_points)
    history: list[dict[str, Any]] = []
    previous_rms: Optional[float] = None

    for iteration in range(max(1, int(maximum_iterations))):
        transformed_source = transform_points(transform, source_points)
        distances, nearest = tree.query(transformed_source, k=1, workers=-1)
        valid = (
            np.isfinite(distances)
            & (distances <= float(maximum_correspondence_distance_m))
        )
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) < 3:
            break

        keep_count = max(
            3,
            int(np.ceil(len(valid_indices) * overlap_fraction)),
        )
        order = np.argsort(distances[valid_indices])
        keep = valid_indices[order[:keep_count]]

        source_matched = transformed_source[keep]
        target_matched = target_points[nearest[keep]]
        delta = _estimate_rigid_transform_kabsch(source_matched, target_matched)
        transform = delta @ transform

        source_after = transform_points(delta, source_matched)
        residual = np.linalg.norm(source_after - target_matched, axis=1)
        rms = float(np.sqrt(np.mean(residual**2)))
        history.append(
            {
                "iteration": int(iteration),
                "pairs_under_max_correspondence_distance": int(len(valid_indices)),
                "pairs_kept": int(len(keep)),
                "rms_m": rms,
            }
        )
        if (
            previous_rms is not None
            and abs(previous_rms - rms) < float(stop_rms_change_m)
        ):
            break
        previous_rms = rms

    return transform, history


def _semantic_dense_registration_score(
    demo_points_world: Array,
    current_points_world: Array,
    relative_transform_world: Array,
    *,
    maximum_correspondence_distance_m: float,
    trim_fraction: float,
) -> dict[str, float]:
    transformed_demo = transform_points(relative_transform_world, demo_points_world)

    forward_tree = cKDTree(current_points_world)
    forward_distance, _ = forward_tree.query(transformed_demo, k=1, workers=-1)
    reverse_tree = cKDTree(transformed_demo)
    reverse_distance, _ = reverse_tree.query(current_points_world, k=1, workers=-1)

    threshold = float(maximum_correspondence_distance_m)
    source_overlap = float(np.mean(forward_distance <= threshold))
    target_overlap = float(np.mean(reverse_distance <= threshold))

    def trimmed_rms(values: Array) -> float:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("inf")
        keep = max(1, int(np.ceil(values.size * float(trim_fraction))))
        selected = np.partition(values, keep - 1)[:keep]
        return float(np.sqrt(np.mean(selected**2)))

    forward_rms = trimmed_rms(forward_distance)
    reverse_rms = trimmed_rms(reverse_distance)
    return {
        "source_overlap": source_overlap,
        "target_overlap": target_overlap,
        "forward_trimmed_rms_m": forward_rms,
        "reverse_trimmed_rms_m": reverse_rms,
        "bidirectional_trimmed_rms_m": 0.5 * (forward_rms + reverse_rms),
    }


def _refine_relative_pose(
    demo_points: Array,
    current_points: Array,
    initial: Array,
    *,
    enabled: bool,
    stages: Sequence[tuple[float, int]],
    overlap_fraction: float,
    stop_rms_change_m: float,
    accept_only_if_improved: bool,
    score_distance_m: float,
    score_trim_fraction: float,
) -> tuple[Array, list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Run coarse/fine point-to-point ICP and retain attempted-fit histories."""
    score_options = dict(
        maximum_correspondence_distance_m=score_distance_m,
        trim_fraction=score_trim_fraction,
    )
    initial_score = _semantic_dense_registration_score(
        demo_points, current_points, initial, **score_options
    )
    final = initial
    histories: list[list[dict[str, Any]]] = [[], []]
    if enabled:
        candidate = initial
        for index, (distance, iterations) in enumerate(stages):
            candidate, histories[index] = _semantic_trimmed_icp(
                demo_points, current_points, candidate,
                overlap_fraction=overlap_fraction,
                maximum_correspondence_distance_m=distance,
                maximum_iterations=iterations,
                stop_rms_change_m=stop_rms_change_m,
            )
        candidate_score = _semantic_dense_registration_score(
            demo_points, current_points, candidate, **score_options
        )
        if (
            not accept_only_if_improved
            or candidate_score["bidirectional_trimmed_rms_m"]
            <= initial_score["bidirectional_trimmed_rms_m"]
        ):
            final = candidate
    final_score = _semantic_dense_registration_score(
        demo_points, current_points, final, **score_options
    )
    return final, histories[0], histories[1], final_score


def _erode_mask_for_icp(mask: Array, radius_px: int) -> Array:
    """Erode a SAM2 mask before dense ICP to avoid silhouette/depth-edge noise."""
    mask = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_px))
    if radius == 0:
        return mask.copy()
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    # Never turn a valid segmentation into an unusable empty cloud.
    return eroded if eroded.any() else mask.copy()


def _loftr_bbox_crop(
    image_rgb: Array,
    mask: Array,
    bbox: Sequence[float],
    *,
    padding_px: int,
) -> tuple[Array, tuple[int, int], tuple[int, int, int, int]]:
    """Crop an RGB bbox (optionally padded) and return crop + full-image origin."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    p = max(0, int(padding_px))
    expanded = [x1 - p, y1 - p, x2 + p, y2 + p]
    crop, _, origin = _bbox_crop(image_rgb, mask, expanded)
    ox, oy = origin
    box = (ox, oy, ox + int(crop.shape[1]), oy + int(crop.shape[0]))
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        raise RuntimeError(f"LoFTR bbox crop is too small: {crop.shape[:2]}")
    return crop, origin, box


def _rigid_ransac_world(
    source_world: Array,
    target_world: Array,
    *,
    threshold_m: float,
    iterations: int,
    seed: int,
) -> tuple[Array, Array]:
    """3-point fixed-scale rigid RANSAC followed by all-inlier Kabsch refit."""
    source = np.asarray(source_world, dtype=np.float64)
    target = np.asarray(target_world, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("RANSAC inputs must have matching shape (N,3).")
    if source.shape[0] < 3:
        raise RuntimeError("Need at least three RGB-D LoFTR pairs for 3D RANSAC.")

    rng = np.random.default_rng(int(seed))
    best_mask: Optional[Array] = None
    best_key: Optional[tuple[int, float]] = None

    for _ in range(max(1, int(iterations))):
        ids = rng.choice(source.shape[0], size=3, replace=False)
        a = source[ids]
        b = target[ids]
        if np.linalg.norm(np.cross(a[1] - a[0], a[2] - a[0])) < 1e-6:
            continue
        if np.linalg.norm(np.cross(b[1] - b[0], b[2] - b[0])) < 1e-6:
            continue
        transform = _estimate_rigid_transform_kabsch(a, b)
        errors = np.linalg.norm(transform_points(transform, source) - target, axis=1)
        mask = np.isfinite(errors) & (errors <= float(threshold_m))
        count = int(mask.sum())
        if count < 3:
            continue
        rms = float(np.sqrt(np.mean(errors[mask] ** 2)))
        key = (count, -rms)
        if best_key is None or key > best_key:
            best_key = key
            best_mask = mask

    if best_mask is None or int(best_mask.sum()) < 3:
        raise RuntimeError(
            "3D RANSAC could not find a rigid LoFTR hypothesis with at least 3 inliers."
        )

    # Refit/reclassify a few times using every current inlier.
    inliers = best_mask
    transform = _estimate_rigid_transform_kabsch(source[inliers], target[inliers])
    for _ in range(3):
        errors = np.linalg.norm(transform_points(transform, source) - target, axis=1)
        updated = np.isfinite(errors) & (errors <= float(threshold_m))
        if int(updated.sum()) < 3 or np.array_equal(updated, inliers):
            break
        inliers = updated
        transform = _estimate_rigid_transform_kabsch(source[inliers], target[inliers])

    transform = _estimate_rigid_transform_kabsch(source[inliers], target[inliers])
    return transform, np.asarray(inliers, dtype=bool)


def _world_up_tangent_basis(world_up: Sequence[float]) -> tuple[Array, Array, Array]:
    """Return an orthonormal (x, y, up) basis with x/y perpendicular to up."""
    up = normalize(np.asarray(world_up, dtype=np.float64), name="world_up")

    candidates = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    x_axis: Optional[Array] = None
    for candidate in candidates:
        projected = candidate - up * float(np.dot(candidate, up))
        norm = float(np.linalg.norm(projected))
        if norm > 1e-8:
            x_axis = projected / norm
            break
    if x_axis is None:
        raise ValueError("Could not construct a tangent basis for world_up.")

    y_axis = normalize(np.cross(up, x_axis), name="world_up_y_axis")
    x_axis = normalize(np.cross(y_axis, up), name="world_up_x_axis")
    return x_axis, y_axis, up


def _estimate_yaw_only_rigid_transform(
    source_world: Array,
    target_world: Array,
    *,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> Array:
    """Estimate translation + rotation only about ``world_up``.

    The rotation is estimated in the 2D tangent plane orthogonal to world_up.
    Translation is then estimated as the component-wise median of the residual
    vectors, which makes the result less sensitive to a few bad depth matches.
    """
    source = np.asarray(source_world, dtype=np.float64)
    target = np.asarray(target_world, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Yaw-only registration inputs must have matching shape (N,3).")
    if source.shape[0] < 2:
        raise RuntimeError("Need at least two correspondences for yaw-only registration.")

    basis_x, basis_y, up = _world_up_tangent_basis(world_up)
    basis = np.column_stack((basis_x, basis_y, up))

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center[None, :]
    target_centered = target - target_center[None, :]

    # Express centered points in the tangent (x,y) coordinates.
    source_xy = np.column_stack(
        (source_centered @ basis_x, source_centered @ basis_y)
    )
    target_xy = np.column_stack(
        (target_centered @ basis_x, target_centered @ basis_y)
    )

    covariance_2d = source_xy.T @ target_xy
    u, _, vt = np.linalg.svd(covariance_2d)
    rotation_2d = vt.T @ u.T
    if float(np.linalg.det(rotation_2d)) < 0.0:
        vt[-1, :] *= -1.0
        rotation_2d = vt.T @ u.T

    rotation_local = np.eye(3, dtype=np.float64)
    rotation_local[:2, :2] = rotation_2d
    rotation_world = basis @ rotation_local @ basis.T

    rotated_source = source @ rotation_world.T
    translation_samples = target - rotated_source
    translation = np.median(translation_samples, axis=0)
    return make_transform(rotation_world, translation)


def _rotation_constraint_name(value: str) -> str:
    mode = str(value).strip().lower()
    aliases = {
        "full": "full_6d",
        "full6d": "full_6d",
        "full_6d": "full_6d",
        "yaw": "yaw_only",
        "yaw_only": "yaw_only",
    }
    if mode not in aliases:
        raise ValueError(
            "rotation_constraint must be 'full_6d' or 'yaw_only', "
            f"got {value!r}."
        )
    return aliases[mode]


def _signed_yaw_about_world_up_deg(
    rotation_world: Array,
    *,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> float:
    """Return the signed yaw of a world rotation about ``world_up``."""
    basis_x, basis_y, _ = _world_up_tangent_basis(world_up)
    rotated_x = np.asarray(rotation_world, dtype=np.float64) @ basis_x
    x_component = float(np.dot(rotated_x, basis_x))
    y_component = float(np.dot(rotated_x, basis_y))
    return math.degrees(math.atan2(y_component, x_component))


# Saved poses and visualizations.


def save_mask_and_overlay(
    output_dir: Path,
    prefix: str,
    image_rgb: Array,
    mask: Array,
    bbox: Sequence[float],
) -> None:
    """Save the segmentation mask over the input image for visual inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = np.asarray(image_rgb, dtype=np.uint8).copy()
    mask_color = np.asarray((0, 255, 0), dtype=np.uint8)
    alpha = 0.35
    overlay[mask] = (
        (1.0 - alpha) * overlay[mask] + alpha * mask_color
    ).astype(np.uint8)
    x1, y1, x2, y2 = clamp_bbox(bbox, image_rgb.shape[:2]).astype(int)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
    Image.fromarray(overlay).save(output_dir / f"{prefix}_mask_overlay.png")


def save_rgb_match_visualization(
    path: Path,
    demo_rgb: Array,
    current_rgb: Array,
    demo_pixels: Array,
    current_pixels: Array,
    inlier_mask: Array,
) -> None:
    demo = np.asarray(demo_rgb, dtype=np.uint8)
    current = np.asarray(current_rgb, dtype=np.uint8)
    height = max(demo.shape[0], current.shape[0])
    width = demo.shape[1] + current.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: demo.shape[0], : demo.shape[1]] = demo
    canvas[: current.shape[0], demo.shape[1] :] = current
    demo_pixels = np.asarray(demo_pixels, dtype=np.float64)
    current_pixels = np.asarray(current_pixels, dtype=np.float64)
    inlier_mask = np.asarray(inlier_mask, dtype=bool)
    for index, (p0, p1) in enumerate(zip(demo_pixels, current_pixels)):
        start = (int(round(p0[0])), int(round(p0[1])))
        end = (
            int(round(p1[0])) + demo.shape[1],
            int(round(p1[1])),
        )
        is_inlier = bool(inlier_mask[index]) if index < inlier_mask.size else False
        color = (0, 255, 0) if is_inlier else (255, 0, 0)
        thickness = 2 if is_inlier else 1
        cv2.line(canvas, start, end, color=color, thickness=thickness)
        cv2.circle(canvas, start, radius=3, color=color, thickness=-1)
        cv2.circle(canvas, end, radius=3, color=color, thickness=-1)
    Image.fromarray(canvas).save(path)


def save_registered_target_pose_result(
    output_dir: Path,
    result: RegisteredTargetPoses,
    *,
    demo_rgb: Array,
    current_rgb: Array,
    demo_bbox: Sequence[float],
    current_bbox: Sequence[float],
    entity_name: str = "target",
) -> None:
    """Save final poses, key fit metrics, and three registration visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    entity_name = str(entity_name).strip().replace(" ", "_") or "target"

    for prefix, image, mask, bbox in (
        ("demo", demo_rgb, result.demo_mask, demo_bbox),
        ("current", current_rgb, result.current_mask, current_bbox),
    ):
        save_mask_and_overlay(output_dir, prefix, image, mask, bbox)

    # Depth-valid LoFTR/RGB-D candidates: green=RANSAC inlier, red=outlier.
    save_rgb_match_visualization(
        output_dir / "rgb_guided_matches.png",
        demo_rgb,
        current_rgb,
        result.rgb_match_demo_pixels,
        result.rgb_match_current_pixels,
        result.rgb_match_inlier_mask,
    )
    summary = {
        "entity_name": entity_name,
        "frame_convention": (
            "Object-to-world transforms; translation in meters. Demo axes come "
            "from the selected pose method; current axes follow the estimated "
            "demo-to-current rotation."
        ),
        "T_world_target_demo": result.transform_world_target_demo.tolist(),
        "T_world_target_current": result.transform_world_target_current.tolist(),
        "registration": {
            name: getattr(result.registration, name)
            for name in (
                "method",
                "accepted",
                "fallback_used",
                "selected_candidate_method",
                "converged",
                "correspondence_count",
                "initial_rms_m",
                "final_rms_m",
                "source_overlap",
                "target_overlap",
            )
        },
    }
    (output_dir / f"{entity_name}_pose_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
