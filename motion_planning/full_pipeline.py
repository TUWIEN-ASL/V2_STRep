from __future__ import annotations

"""End-to-end orchestration helpers for the motion-planning pipeline.

This module intentionally keeps file/configuration handling separate from the
geometry and optimization modules.  The public entry point is
``scripts/run_full_pipeline.py``.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from os.path import relpath
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .current_adaptation import (
    CurrentAdaptation,
    CurrentAdaptationValidationError,
    parse_current_adaptation,
)
from .demo_trajectory import prepare_demo_trajectory
from .grasp_candidates import (
    coerce_grasp_pose_array,
    grasp_pose_error,
    grasp_poses_to_world,
)
from .grasp_attachment import (
    GraspAttachment,
    build_grasp_attachment,
    object_trajectory_to_grasp_trajectory,
)
from .object_flow_pose import (
    estimate_object_pose_trajectory,
    load_object_flow,
    normalize,
    project_axis_to_plane,
    points_world_to_object,
)
from .reference_builder import (
    MotionReference,
    build_motion_reference,
    save_motion_reference,
)
from .scene_geometry import (
    TargetGrounding,
    CameraModel,
    backproject_pixel_to_world,
    build_scene_geometry,
    parse_target_grounding,
)
from .target_pose_registration import (
    RegisteredTargetPoses,
    Sam2BoxSegmenter,
    compute_pca_target_pose,
    estimate_registered_entity_poses_from_loftr,
    load_rgb,
    save_registered_target_pose_result,
)
from .trajectory_optimizer import (
    TrajectoryOptimizationResult,
    TrajectoryOptimizerConfig,
    result_summary,
    run_multi_start_trajectory_optimization,
    run_trajectory_optimization,
    save_optimization_result,
)


FloatArray = NDArray[np.float64]
Pixel = tuple[int, int]

# Standard Panda nominal configuration used only as a numerical warm start when
# pose-driven grasp optimization has neither current.q nor current.q_candidates.
# This matches the nominal seed used by grasp2world.py.
PANDA_NOMINAL_Q = np.array(
[1.182322788634764, 
1.3800003763124726, 
-1.798522808253073, 
-2.3118736096177184, 
1.3347757708709436, 
1.582170154471965, 
-0.01415557294859602],
    dtype=np.float64,
)


class PipelineInputError(ValueError):
    """Raised when the requested run cannot be constructed from its inputs."""

    def __init__(self, message: str, *, missing: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.missing = tuple(missing)


@dataclass(frozen=True)
class CameraBundle:
    intrinsic_matrix: FloatArray
    transform_world_camera: FloatArray
    depth_scale_to_m: float
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    trajectory: Path
    registration: Path
    grasps: Path


@dataclass(frozen=True)
class SceneInputs:
    demo_depth: FloatArray
    current_depth: FloatArray
    video_structure_payload: Mapping[str, Any]
    demo_grounding_payload: Mapping[str, Any]
    current_adaptation_payload: Mapping[str, Any]
    initial_adaptation: CurrentAdaptation


@dataclass(frozen=True)
class ControlledObjectState:
    object_pose_world: FloatArray
    grasp_pose_world: FloatArray
    attachment: GraspAttachment
    registration: Optional[RegisteredTargetPoses]


@dataclass(frozen=True)
class TargetGeometryInputs:
    demo_grounding: TargetGrounding
    current_grounding: TargetGrounding
    demo_points_world: Optional[FloatArray]
    current_points_world: Optional[FloatArray]


@dataclass(frozen=True)
class GraspCandidateInputs:
    raw_grasp_poses: Optional[FloatArray]
    q_start_candidates: FloatArray
    grasp_scores: Optional[FloatArray]
    q_seed_source: str


def create_output_tree(
    root: Path, *, overwrite: bool = False, preserve: Sequence[Path] = (),
) -> PipelinePaths:
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {root}. "
                "Use --overwrite or choose a new run directory."
            )
        for path in root.iterdir():
            if path in preserve:
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()

    paths = PipelinePaths(
        root=root,
        trajectory=root / "trajectory",
        registration=root / "object_pose_registration",
        grasps=root / "vlm_grasps",
    )
    for directory in asdict(paths).values():
        Path(directory).mkdir(parents=True, exist_ok=True)
    return paths


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def resolve_path(value: Any, base_dir: Path) -> Optional[Path]:
    if value is None or value == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()




def loftr_relative_registration_options(
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Shared LoFTR, RGB-D, and refinement options for entity registration."""
    return {
        "minimum_depth_m": float(mapping.get("minimum_depth_m", 0.05)),
        "maximum_depth_m": float(mapping.get("maximum_depth_m", 3.0)),
        "depth_band_m": float(mapping.get("depth_band_m", 0.0)),
        "point_stride": int(mapping.get("point_stride", 1)),
        "loftr_pretrained": str(mapping.get("loftr_pretrained", "indoor")),
        "loftr_device": str(
            mapping.get("loftr_device", mapping.get("device", "cuda"))
        ),
        "loftr_confidence_threshold": float(
            mapping.get("loftr_confidence_threshold", 0.2)
        ),
        "loftr_maximum_long_side": int(
            mapping.get("loftr_maximum_long_side", 1024)
        ),
        "loftr_crop_padding_px": int(mapping.get("loftr_crop_padding_px", 0)),
        "loftr_maximum_matches": int(mapping.get("loftr_maximum_matches", 500)),
        "depth_policy": str(mapping.get("depth_policy", "min")),
        "depth_search_radius_px": int(mapping.get("depth_search_radius_px", 2)),
        "ransac_enabled": bool(mapping.get("ransac_enabled", True)),
        "ransac_3d_inlier_threshold_m": float(
            mapping.get("ransac_3d_inlier_threshold_m", 0.02)
        ),
        "ransac_iterations": int(mapping.get("ransac_iterations", 5000)),
        "ransac_seed": int(mapping.get("ransac_seed", 0)),
        "ransac_minimum_inliers": int(mapping.get("ransac_minimum_inliers", 3)),
        "ransac_minimum_inlier_ratio": float(
            mapping.get("ransac_minimum_inlier_ratio", 0.0)
        ),
        "mask_erode_px": int(mapping.get("mask_erode_px", 2)),
        "icp_refine": bool(mapping.get("icp_refine", True)),
        "icp_voxel_size_m": float(mapping.get("icp_voxel_size_m", 0.005)),
        "icp_overlap_fraction": float(mapping.get("icp_overlap_fraction", 0.65)),
        "icp_coarse_max_correspondence_distance_m": float(
            mapping.get("icp_coarse_max_correspondence_distance_m", 0.05)
        ),
        "icp_fine_max_correspondence_distance_m": float(
            mapping.get("icp_fine_max_correspondence_distance_m", 0.015)
        ),
        "icp_coarse_max_iterations": int(
            mapping.get("icp_coarse_max_iterations", 60)
        ),
        "icp_fine_max_iterations": int(
            mapping.get("icp_fine_max_iterations", 100)
        ),
        "icp_stop_rms_change_m": float(
            mapping.get("icp_stop_rms_change_m", 1e-5)
        ),
        "icp_accept_only_if_improved": bool(
            mapping.get("icp_accept_only_if_improved", False)
        ),
        "validation_max_correspondence_distance_m": float(
            mapping.get("validation_max_correspondence_distance_m", 0.03)
        ),
        "validation_trim_fraction": float(
            mapping.get("validation_trim_fraction", 0.65)
        ),
    }


def merge_mappings(*values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if isinstance(value, Mapping):
            result.update(value)
    return result


def dotted_get(payload: Any, dotted_key: str) -> Any:
    current = payload
    for component in dotted_key.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(dotted_key)
        current = current[component]
    return current


def _walk_mappings(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        yield payload
        for value in payload.values():
            yield from _walk_mappings(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _walk_mappings(value)


def _find_first_key(payload: Any, keys: Sequence[str]) -> tuple[Any, Optional[str]]:
    for mapping in _walk_mappings(payload):
        for key in keys:
            if key in mapping:
                return mapping[key], key
    return None, None


def _coerce_intrinsics(value: Any) -> FloatArray:
    if isinstance(value, Mapping):
        if all(key in value for key in ("fx", "fy", "cx", "cy")):
            return np.array(
                [
                    [float(value["fx"]), 0.0, float(value["cx"])],
                    [0.0, float(value["fy"]), float(value["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        for key in ("K", "data", "matrix", "intrinsic_matrix"):
            if key in value:
                return _coerce_intrinsics(value[key])
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (9,):
        array = array.reshape(3, 3)
    if array.shape != (3, 3):
        raise ValueError(f"Camera intrinsics must be (3,3) or length 9, got {array.shape}.")
    return array


def _coerce_transform(value: Any) -> FloatArray:
    if isinstance(value, Mapping):
        for key in ("matrix", "data", "transform", "T_world_camera"):
            if key in value:
                return _coerce_transform(value[key])
        translation = value.get("translation", value.get("position"))
        quaternion = value.get("quaternion", value.get("orientation"))
        if translation is not None and quaternion is not None:
            if isinstance(translation, Mapping):
                translation = [translation[key] for key in ("x", "y", "z")]
            if isinstance(quaternion, Mapping):
                quaternion = [quaternion[key] for key in ("x", "y", "z", "w")]
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = Rotation.from_quat(
                np.asarray(quaternion, dtype=np.float64)
            ).as_matrix()
            transform[:3, 3] = np.asarray(translation, dtype=np.float64)
            return transform
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (16,):
        array = array.reshape(4, 4)
    if array.shape != (4, 4):
        raise ValueError(f"T_world_camera must be (4,4) or length 16, got {array.shape}.")
    return array


def _load_array_file(path: Path, *, key: Optional[str] = None) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as bundle:
            if key is not None:
                if key not in bundle:
                    raise KeyError(f"{path} has no key {key!r}; keys={list(bundle.keys())}.")
                return np.asarray(bundle[key])
            keys = list(bundle.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"{path} contains multiple arrays {keys}; specify an explicit key."
                )
            return np.asarray(bundle[keys[0]])
    if suffix == ".json":
        payload = load_json(path)
        return dotted_get(payload, key) if key else payload
    raise ValueError(f"Unsupported camera/array file type: {path}")


def load_camera_bundle(camera_config: Mapping[str, Any], base_dir: Path) -> CameraBundle:
    assumptions: list[str] = []
    bundle_path = resolve_path(camera_config.get("bundle"), base_dir)
    intrinsics_path = resolve_path(camera_config.get("intrinsics"), base_dir)
    transform_path = resolve_path(camera_config.get("T_world_camera"), base_dir)
    intrinsics_key = camera_config.get("intrinsics_key")
    transform_key = camera_config.get("T_world_camera_key")

    payload: Any = None
    if bundle_path is not None:
        if not bundle_path.exists():
            raise PipelineInputError(
                f"Camera bundle does not exist: {bundle_path}",
                missing=("camera.bundle",),
            )
        if bundle_path.suffix.lower() == ".npz":
            with np.load(bundle_path, allow_pickle=False) as bundle:
                payload = {key: np.asarray(bundle[key]) for key in bundle.keys()}
        elif bundle_path.suffix.lower() == ".json":
            payload = load_json(bundle_path)
        else:
            raise PipelineInputError(
                "camera.bundle must be a .json or .npz file."
            )

    if intrinsics_path is not None:
        if not intrinsics_path.exists():
            raise PipelineInputError(
                f"Intrinsics file does not exist: {intrinsics_path}",
                missing=("camera.intrinsics",),
            )
        intrinsics_value = _load_array_file(intrinsics_path, key=intrinsics_key)
    elif payload is not None and intrinsics_key:
        intrinsics_value = dotted_get(payload, str(intrinsics_key))
    elif payload is not None:
        intrinsics_value, found_key = _find_first_key(
            payload,
            ("K", "intrinsic_matrix", "camera_matrix", "camera_intrinsics", "intrinsics"),
        )
        if found_key is None:
            # Some calibration files directly store fx/fy/cx/cy in a nested mapping.
            intrinsics_value = None
            for mapping in _walk_mappings(payload):
                if all(key in mapping for key in ("fx", "fy", "cx", "cy")):
                    intrinsics_value = mapping
                    break
    else:
        intrinsics_value = None

    if transform_path is not None:
        if not transform_path.exists():
            raise PipelineInputError(
                f"Camera transform file does not exist: {transform_path}",
                missing=("camera.T_world_camera",),
            )
        transform_value = _load_array_file(transform_path, key=transform_key)
    elif payload is not None and transform_key:
        transform_value = dotted_get(payload, str(transform_key))
    elif payload is not None:
        transform_value, found_key = _find_first_key(
            payload,
            (
                "T_world_camera",
                "transform_world_camera",
                "world_T_camera",
                "camera_extrinsics",
                "extrinsic_matrix",
                "extrinsics",
            ),
        )
        if found_key in {"camera_extrinsics", "extrinsic_matrix", "extrinsics"}:
            assumptions.append(
                f"Assumed calibration key {found_key!r} is T_world_camera, not its inverse."
            )
    else:
        transform_value = None

    missing = []
    if intrinsics_value is None:
        missing.append("camera intrinsics K")
    if transform_value is None:
        missing.append("T_world_camera")
    if missing:
        raise PipelineInputError(
            "Camera calibration is incomplete: " + ", ".join(missing) + ".",
            missing=tuple(missing),
        )

    depth_scale = float(camera_config.get("depth_scale_to_m", 1.0))
    camera = CameraModel(
        intrinsic_matrix=_coerce_intrinsics(intrinsics_value),
        transform_world_camera=_coerce_transform(transform_value),
        depth_scale_to_m=depth_scale,
    )
    return CameraBundle(
        intrinsic_matrix=camera.intrinsic_matrix,
        transform_world_camera=camera.transform_world_camera,
        depth_scale_to_m=depth_scale,
        assumptions=tuple(assumptions),
    )


def camera_model(bundle: CameraBundle) -> CameraModel:
    return CameraModel(
        intrinsic_matrix=bundle.intrinsic_matrix,
        transform_world_camera=bundle.transform_world_camera,
        depth_scale_to_m=bundle.depth_scale_to_m,
    )


def _bbox_from_section(section: Mapping[str, Any], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    bbox = section.get("bbox")
    if bbox is None:
        raise PipelineInputError(
            "Scene grounding needs controlled_entity.bbox for automatic object reconstruction.",
            missing=("controlled_entity.bbox",),
        )
    array = np.asarray(bbox, dtype=np.int64)
    if array.shape != (4,):
        raise ValueError(f"bbox must be [x1,y1,x2,y2], got {array.shape}.")
    height, width = image_shape
    x1, y1, x2, y2 = (int(value) for value in array)
    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _positive_pixels(section: Mapping[str, Any]) -> list[Pixel]:
    for key in ("positive_pixels", "seed_pixels", "pixels"):
        value = section.get(key)
        if value:
            array = np.asarray(value, dtype=np.int64)
            if array.ndim == 1 and array.shape == (2,):
                array = array[None, :]
            if array.ndim == 2 and array.shape[1] == 2:
                return [(int(row[0]), int(row[1])) for row in array]
    return []


def choose_grasp_pixel(payload: Mapping[str, Any], image_shape: tuple[int, int]) -> Pixel:
    region = payload.get("allowed_grasp_region")
    if not isinstance(region, Mapping):
        region = payload.get("controlled_entity")
    if not isinstance(region, Mapping):
        raise PipelineInputError(
            "Scene grounding needs allowed_grasp_region or controlled_entity.",
            missing=("allowed_grasp_region",),
        )

    positives = _positive_pixels(region)
    if positives:
        return positives[0]

    x1, y1, x2, y2 = _bbox_from_section(region, image_shape)
    return ((x1 + x2 - 1) // 2, (y1 + y2 - 1) // 2)


def _valid_depth_m(depth: FloatArray, scale: float) -> FloatArray:
    return np.asarray(depth, dtype=np.float64) * scale


def reconstruct_controlled_object_points_from_scene(
    depth: FloatArray,
    grounding_payload: Mapping[str, Any],
    camera: CameraModel,
    *,
    depth_band_m: float = 0.04,
    bbox_stride: int = 2,
    minimum_points: int = 50,
    grasp_patch_radius: int = 2,
) -> tuple[
    FloatArray,
    float,
    tuple[int, int, int, int],
]:
    """Extract a depth-gated controlled-object point cloud in world coordinates."""
    depth_array = np.asarray(depth, dtype=np.float64)
    if depth_array.ndim != 2:
        raise ValueError(f"Initial depth must have shape (H,W), got {depth_array.shape}.")

    controlled = grounding_payload.get("controlled_entity")
    if not isinstance(controlled, Mapping):
        raise PipelineInputError(
            "Scene grounding is missing controlled_entity.",
            missing=("controlled_entity",),
        )
    bbox = _bbox_from_section(controlled, depth_array.shape)
    seeds = _positive_pixels(controlled)
    if not seeds:
        x1, y1, x2, y2 = bbox
        seeds = [((x1 + x2 - 1) // 2, (y1 + y2 - 1) // 2)]

    seed_depths = []
    for pixel in seeds:
        try:
            _, depth_m = backproject_pixel_to_world(
                depth_array,
                pixel,
                camera,
                patch_radius=grasp_patch_radius,
            )
        except ValueError:
            continue
        seed_depths.append(depth_m)
    if not seed_depths:
        raise PipelineInputError(
            "No valid depth was found around controlled_entity positive pixels."
        )
    seed_depth_m = float(np.median(seed_depths))

    x1, y1, x2, y2 = bbox
    us = np.arange(x1, x2, max(1, int(bbox_stride)), dtype=np.int64)
    vs = np.arange(y1, y2, max(1, int(bbox_stride)), dtype=np.int64)
    grid_u, grid_v = np.meshgrid(us, vs)
    raw_depth = depth_array[grid_v, grid_u]
    depth_m = _valid_depth_m(raw_depth, camera.depth_scale_to_m)
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= camera.minimum_depth_m)
        & (depth_m <= camera.maximum_depth_m)
        & (np.abs(depth_m - seed_depth_m) <= depth_band_m)
    )
    selected_u = grid_u[valid].astype(np.float64)
    selected_v = grid_v[valid].astype(np.float64)
    selected_z = depth_m[valid]
    if selected_z.size < minimum_points:
        raise PipelineInputError(
            f"Only {selected_z.size} controlled-object depth points survived bbox/depth filtering; "
            f"need at least {minimum_points}. Improve the object bbox/positive pixels, increase "
            "reconstruction.object_depth_band_m, or provide an explicit T_world_object."
        )

    K = camera.intrinsic_matrix
    points_camera = np.column_stack(
        (
            (selected_u - K[0, 2]) * selected_z / K[0, 0],
            (selected_v - K[1, 2]) * selected_z / K[1, 1],
            selected_z,
        )
    )
    rotation_world_camera = camera.transform_world_camera[:3, :3]
    translation_world_camera = camera.transform_world_camera[:3, 3]
    points_world = (
        points_camera @ rotation_world_camera.T
        + translation_world_camera[None, :]
    )
    return points_world, seed_depth_m, bbox


def reconstruct_target_object_points_from_scene(
    depth: FloatArray,
    grounding_payload: Mapping[str, Any],
    camera: CameraModel,
    *,
    depth_band_m: float = 0.04,
    bbox_stride: int = 2,
    minimum_points: int = 50,
    patch_radius: int = 2,
) -> FloatArray:
    """Reconstruct an object_pose_6d target point cloud from Prompt grounding."""
    target_geometry = grounding_payload.get("target_geometry")
    if not isinstance(target_geometry, Mapping):
        raise PipelineInputError(
            "object_pose_6d requires current_grounding.target_geometry."
        )
    bbox = target_geometry.get("target_object_bbox")
    positives = target_geometry.get("target_object_positive_pixels")
    if bbox is None:
        raise PipelineInputError(
            "object_pose_6d requires target_object_bbox.",
            missing=("target_geometry.target_object_bbox",),
        )
    proxy = {
        "controlled_entity": {
            "bbox": bbox,
            "positive_pixels": positives or [],
        }
    }
    points_world, _, _ = reconstruct_controlled_object_points_from_scene(
        depth,
        proxy,
        camera,
        depth_band_m=depth_band_m,
        bbox_stride=bbox_stride,
        minimum_points=minimum_points,
        grasp_patch_radius=patch_radius,
    )
    return points_world


def load_transform(path: Path, *, allow_trajectory: bool = False) -> FloatArray:
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    valid = value.shape == (4, 4) or (
        allow_trajectory and value.ndim == 3 and value.shape[1:] == (4, 4)
    )
    if not valid:
        expected = "(4,4) or (T,4,4)" if allow_trajectory else "(4,4)"
        raise ValueError(f"{path} must contain {expected}, got {value.shape}.")
    return value


def load_sam3d_controlled_rotation(
    path: Path,
    camera: CameraBundle,
    origin_world: FloatArray,
    image_shape: tuple[int, int],
    world_up: Sequence[float],
) -> tuple[FloatArray, dict[str, Any]]:
    """Name/sign fitted SAM3D axes; retain tilt and the pipeline's object origin."""
    with np.load(path, allow_pickle=False) as pose:
        if str(pose["translation_units"].item()) != "meters":
            raise PipelineInputError("Use a metric, RGB-D-refined SAM3D pose.")
        if (tuple(pose["image_size_hw"]) != tuple(image_shape)
                or not np.allclose(pose["intrinsics_pixels"], camera.intrinsic_matrix,
                                   rtol=1e-5, atol=1e-4)):
            raise PipelineInputError("SAM3D pose camera does not match the demo camera.")
        quaternion = np.asarray(pose["rotation"], dtype=np.float64).reshape(4)
    if not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) == 0:
        raise PipelineInputError("SAM3D pose has an invalid quaternion.")

    # SAM3D uses row-vector wxyz rotations and left/up/forward camera axes.
    rotation_row = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    native = camera.transform_world_camera[:3, :3] @ np.diag([-1., -1., 1.]) @ rotation_row.T
    up = normalize(world_up, name="world_up")
    up_scores = np.abs(native.T @ up)
    z_index = int(np.argmax(up_scores))
    z_axis = normalize(native[:, z_index], name="SAM3D z")
    if np.dot(z_axis, up) < 0:
        z_axis = -z_axis
    remaining = [index for index in range(3) if index != z_index]
    toward_camera = project_axis_to_plane(camera.transform_world_camera[:3, 3] - origin_world, z_axis)
    overhead = toward_camera is None
    if overhead:
        # Camera lies on object Z: use a fixed native axis as the remaining gauge.
        toward_camera = native[:, remaining[0]]
    view_scores = np.abs(native[:, remaining].T @ toward_camera)
    x_index = remaining[int(np.argmax(view_scores))]
    x_axis = native[:, x_index]
    if np.dot(x_axis, toward_camera) > 0:
        x_axis = -x_axis
    y_axis = normalize(np.cross(z_axis, x_axis), name="SAM3D y")
    x_axis = normalize(np.cross(y_axis, z_axis), name="SAM3D x")
    semantic = np.column_stack((x_axis, y_axis, z_axis))
    mapping = np.rint(native.T @ semantic).astype(int)
    return semantic, {
        "method": "sam3d",
        "source_pose": str(path.resolve()),
        "origin_policy": "retain_demo_trajectory_origin",
        "native_to_semantic_basis": mapping.tolist(),
        "rotation_world_semantic": semantic.tolist(),
        "camera_gauge_undefined": overhead,
    }


def transform_trajectory_to_pose7(transforms: FloatArray) -> FloatArray:
    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"Expected (N,4,4), got {transforms.shape}.")
    rotations = []
    for matrix in transforms[:, :3, :3]:
        u, _, vt = np.linalg.svd(matrix)
        projected = u @ vt
        if np.linalg.det(projected) < 0.0:
            u[:, -1] *= -1.0
            projected = u @ vt
        rotations.append(projected)
    quaternions = Rotation.from_matrix(np.stack(rotations)).as_quat()
    for index in range(1, quaternions.shape[0]):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
            quaternions[index] *= -1.0
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return np.concatenate((transforms[:, :3, 3], quaternions), axis=1)


def file_sha256(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def snapshot_inputs(paths: Iterable[Path], output_dir: Path, *, copy_files: bool) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    used_names: set[str] = set()
    for path in paths:
        path = path.resolve()
        if not path.exists() or not path.is_file():
            continue
        name = path.name
        if name in used_names:
            name = f"{path.parent.name}_{name}"
        used_names.add(name)
        destination = output_dir / name
        if copy_files:
            shutil.copy2(path, destination)
        records.append(
            {
                "source": str(path),
                "snapshot": str(destination) if copy_files else None,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return records


def object_flow_pose_options(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Forward configured track/pose filters; omitted options use estimator defaults."""
    names = (
        "temporal_median_window",
        "max_translation_step_m",
        "max_rotation_step_deg",
        "global_track_filter",
        "track_min_valid_ratio",
        "track_mad_scale",
        "minimum_stable_tracks",
        "stable_track_huber_delta_m",
        "stable_track_irls_iterations",
        "temporal_refinement",
        "temporal_translation_observation_sigma_m",
        "temporal_translation_acceleration_sigma_m",
        "temporal_rotation_observation_sigma_deg",
        "temporal_rotation_acceleration_sigma_deg",
        "adaptive_step_limit",
        "adaptive_step_limit_mad_scale",
        "adaptive_step_limit_percentile",
    )
    return {
        name: mapping[f"object_flow_{name}"]
        for name in names
        if f"object_flow_{name}" in mapping
    }


def phase_weight_pairs(value: Any) -> tuple[tuple[str, float], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        pairs = value.items()
    else:
        pairs = value
    result = []
    for selector, weight in pairs:
        result.append((str(selector), float(weight)))
    return tuple(result)


def optimizer_config_from_mapping(value: Mapping[str, Any]) -> TrajectoryOptimizerConfig:
    # Retain the pipeline's weight defaults; share all other optimizer defaults.
    defaults = TrajectoryOptimizerConfig(
        axis_scale=8.0,
        attractor_tracking_weight=0.1,
        joint_smoothness_scale=1.0,
        acceleration_scale=0.15,
    )
    fields = {key: value.get(key, default) for key, default in asdict(defaults).items()}
    fields["phase_tracking_weights"] = phase_weight_pairs(
        value.get("phase_tracking_weights")
    )
    return TrajectoryOptimizerConfig(**fields)


def required_path(config: Mapping[str, Any], key: str, base_dir: Path) -> Path:
    path = resolve_path(config.get(key), base_dir)
    if path is None:
        raise PipelineInputError(f"Missing required path: {key}", missing=(key,))
    if not path.exists():
        raise PipelineInputError(f"Input does not exist: {path}", missing=(key,))
    return path


def resolve_first_path(
    config: Mapping[str, Any],
    keys: Sequence[str],
    base_dir: Path,
) -> tuple[Optional[Path], Optional[str]]:
    for key in keys:
        path = resolve_path(config.get(key), base_dir)
        if path is not None:
            return path, key
    return None, None


def required_first_path(
    config: Mapping[str, Any],
    keys: Sequence[str],
    base_dir: Path,
    *,
    label: str,
) -> Path:
    path, key = resolve_first_path(config, keys, base_dir)
    if path is None:
        raise PipelineInputError(
            f"Missing required path: {label}",
            missing=(label,),
        )
    if not path.exists():
        raise PipelineInputError(
            f"Input does not exist: {path}",
            missing=(key or label,),
        )
    return path


def collect_config_input_paths(config: Mapping[str, Any], base_dir: Path) -> list[Path]:
    result: list[Path] = []
    sections = [
        config.get("camera", {}),
        config.get("demo_camera", {}),
        config.get("current_camera", {}),
        config.get("demo", {}),
        config.get("current", {}),
        config.get("robot", {}),
    ]
    path_keys = {
        "bundle",
        "intrinsics",
        "T_world_camera",
        "initial_depth",
        "rgb",
        "initial_rgb",
        "scene_grounding",
        "video_grounding",
        "analysis",
        "adaptation",
        "object_flow",
        "sam3d_pose",
        "sam3d_target_pose",
        "object_trajectory",
        "grasp_trajectory",
        "T_world_object",
        "T_world_grasp",
        "q",
        "q_candidates",
        "q_start_candidates",
        "grasp_scores",
        "graspgen_scores",
        "grasp_pose",
        "grasp_poses",
        "sphere_json",
        "initial_trajectory",
    }
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        for key, value in section.items():
            if key not in path_keys:
                continue
            path = resolve_path(value, base_dir)
            if path is not None:
                result.append(path)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(result))


def validate_minimum_config(config: Mapping[str, Any], base_dir: Path) -> list[str]:
    missing: list[str] = []

    demo = config.get("demo")
    if not isinstance(demo, Mapping):
        missing.append("demo")
    else:
        demo_depth = resolve_path(demo.get("initial_depth"), base_dir)
        demo_analysis, _ = resolve_first_path(
            demo,
            ("analysis", "video_grounding"),
            base_dir,
        )
        if demo_depth is None or not demo_depth.exists():
            missing.append("demo.initial_depth")
        if demo_analysis is None or not demo_analysis.exists():
            missing.append("demo.analysis (or demo.video_grounding)")

    current = config.get("current")
    if not isinstance(current, Mapping):
        missing.append("current")
        current = {}
    else:
        current_depth = resolve_path(current.get("initial_depth"), base_dir)
        current_adaptation, _ = resolve_first_path(
            current,
            ("adaptation", "scene_grounding"),
            base_dir,
        )
        if current_depth is None or not current_depth.exists():
            missing.append("current.initial_depth")
        if current_adaptation is None or not current_adaptation.exists():
            missing.append("current.adaptation (or current.scene_grounding)")

    if isinstance(current, Mapping):
        q_candidate_path = resolve_path(
            current.get("q_candidates", current.get("q_start_candidates")),
            base_dir,
        )
        q_single_path = resolve_path(current.get("q"), base_dir)
        grasp_pose_path, _ = resolve_first_path(
            current,
            ("grasp_poses", "grasp_pose"),
            base_dir,
        )
        has_q_seed = any(
            path is not None and path.exists()
            for path in (q_candidate_path, q_single_path)
        )
        has_grasp_pose = grasp_pose_path is not None and grasp_pose_path.exists()
        if not has_q_seed and not has_grasp_pose:
            # Pose-driven mode can fall back to PANDA_NOMINAL_Q when neither
            # current.q nor current.q_candidates is supplied. Legacy fixed-q
            # mode still requires an explicit joint input.
            missing.append(
                "current start state: provide current.grasp_poses, "
                "current.q_candidates, or current.q"
            )

    demo = config.get("demo", {})
    if isinstance(demo, Mapping):
        motion_paths = [
            resolve_path(demo.get("grasp_trajectory"), base_dir),
            resolve_path(demo.get("object_trajectory"), base_dir),
            resolve_path(demo.get("object_flow"), base_dir),
        ]
        if not any(path is not None and path.exists() for path in motion_paths):
            missing.append(
                "demo motion: provide demo.grasp_trajectory, demo.object_trajectory, "
                "or demo.object_flow"
            )

    reconstruction = config.get("reconstruction", {})
    frame_method = str(reconstruction.get("controlled_object_frame_method", "pca"))
    if frame_method not in ("pca", "sam3d"):
        missing.append("reconstruction.controlled_object_frame_method must be 'pca' or 'sam3d'")
    if frame_method == "sam3d":
        pose_path = resolve_path(demo.get("sam3d_pose"), base_dir) if isinstance(demo, Mapping) else None
        if pose_path is None or not pose_path.is_file():
            missing.append("demo.sam3d_pose: a refined pose.npz for the demonstration image")
        if current.get("T_world_object") and not reconstruction.get("same_scene_identity_registration", False):
            missing.append("SAM3D currently requires automatic current-object registration; clear current.T_world_object")

    target_frame_method = reconstruction.get("target_object_frame_method", "pca")
    if target_frame_method not in ("pca", "sam3d"):
        missing.append("reconstruction.target_object_frame_method must be 'pca' or 'sam3d'")
    if target_frame_method == "sam3d" and isinstance(demo, Mapping):
        analysis_path, _ = resolve_first_path(demo, ("analysis", "video_grounding"), base_dir)
        if analysis_path is not None and analysis_path.is_file():
            analysis = load_json(analysis_path)
            if analysis.get("video_structure", analysis).get("target_representation") == "object_pose_6d":
                pose_path = resolve_path(demo.get("sam3d_target_pose"), base_dir)
                if pose_path is None or not pose_path.is_file():
                    missing.append("demo.sam3d_target_pose: a refined pose.npz for the demo target")

    camera_sections = [
        merge_mappings(config.get("camera"), config.get("demo_camera")),
        merge_mappings(config.get("camera"), config.get("current_camera")),
    ]
    for label, camera_section in zip(("demo camera", "current camera"), camera_sections):
        try:
            load_camera_bundle(camera_section, base_dir)
        except Exception as error:  # Report all camera gaps at once.
            missing.append(f"{label}: {error}")
    return missing


def load_scene_inputs(
    *,
    demo_depth_path: Path,
    current_depth_path: Path,
    demo_analysis_path: Path,
    demo_grounding_path: Optional[Path],
    current_adaptation_path: Path,
) -> SceneInputs:
    """Load scene depths and prompt data, retaining payloads for later validation."""
    demo_depth = np.asarray(np.load(demo_depth_path, allow_pickle=False), dtype=np.float64)
    current_depth = np.asarray(np.load(current_depth_path, allow_pickle=False), dtype=np.float64)
    demo_analysis_payload = load_json(demo_analysis_path)
    if isinstance(demo_analysis_payload, Mapping) and "video_structure" in demo_analysis_payload:
        video_structure_payload = demo_analysis_payload["video_structure"]
        embedded_demo_grounding = demo_analysis_payload.get("demo_initial_grounding")
    else:
        video_structure_payload = demo_analysis_payload
        embedded_demo_grounding = None

    if not isinstance(video_structure_payload, Mapping):
        raise PipelineInputError("demo analysis is missing a valid video_structure object.")

    if demo_grounding_path is not None:
        if not demo_grounding_path.exists():
            raise PipelineInputError(
                f"Input does not exist: {demo_grounding_path}",
                missing=("demo.scene_grounding",),
            )
        demo_grounding_payload = load_json(demo_grounding_path)
    elif isinstance(embedded_demo_grounding, Mapping):
        demo_grounding_payload = embedded_demo_grounding
    else:
        raise PipelineInputError(
            "Provide demo.scene_grounding or use Prompt-1 output containing "
            "demo_initial_grounding.",
            missing=("demo.scene_grounding or demo_initial_grounding",),
        )

    if not isinstance(demo_grounding_payload, Mapping):
        raise PipelineInputError("Demonstration grounding must be a JSON object.")
    if demo_grounding_payload.get("is_usable") is False:
        raise PipelineInputError(
            "Demonstration grounding is unusable: "
            + str(demo_grounding_payload.get("failure_reason"))
        )

    current_adaptation_payload = load_json(current_adaptation_path)
    try:
        current_adaptation = parse_current_adaptation(
            current_adaptation_payload,
            video_structure_payload,
            total_frames=None,
            allow_legacy_direct_grounding=True,
        )
    except CurrentAdaptationValidationError as error:
        raise PipelineInputError(str(error)) from error

    if not current_adaptation.is_usable:
        raise PipelineInputError(
            "Current instruction cannot use this demonstration: "
            + str(current_adaptation.failure_reason)
        )

    return SceneInputs(
        demo_depth=demo_depth,
        current_depth=current_depth,
        video_structure_payload=video_structure_payload,
        demo_grounding_payload=demo_grounding_payload,
        current_adaptation_payload=current_adaptation_payload,
        initial_adaptation=current_adaptation,
    )


def prepare_grasp_candidates(
    *,
    q_single_path: Optional[Path],
    q_candidates_path: Optional[Path],
    grasp_poses_path: Optional[Path],
    grasp_poses_key: Optional[str],
    grasp_scores_path: Optional[Path],
) -> GraspCandidateInputs:
    """Load and validate the grasp poses, joint seeds and candidate scores.

    Supplied grasp poses define the desired start poses; joints provide warm starts.
    Without grasp poses, joint inputs define the starting configurations.
    """
    def load_current_array(path: Optional[Path], key: str) -> Optional[FloatArray]:
        if path is None:
            return None
        if not path.exists():
            raise PipelineInputError(
                f"Input does not exist: {path}", missing=(f"current.{key}",),
            )
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)

    raw_grasp_poses = load_current_array(
        grasp_poses_path, grasp_poses_key or "grasp_poses",
    )
    # Normalize shape now; convert to world coordinates once scene poses exist.
    grasp_pose_candidates_local = (
        None if raw_grasp_poses is None else coerce_grasp_pose_array(raw_grasp_poses)
    )
    using_pose_driven_grasp = grasp_pose_candidates_local is not None
    pose_candidate_count = (
        0 if grasp_pose_candidates_local is None else len(grasp_pose_candidates_local)
    )

    q_single_seed = load_current_array(q_single_path, "q")
    if q_single_seed is not None:
        if q_single_seed.ndim == 2 and q_single_seed.shape[0] == 1:
            q_single_seed = q_single_seed[0]
        if q_single_seed.ndim != 1:
            raise ValueError(
                f"current.q must be one-dimensional, got {q_single_seed.shape}."
            )
        if not np.isfinite(q_single_seed).all():
            raise ValueError("current.q contains NaN or infinity.")

    q_candidate_seeds = load_current_array(q_candidates_path, "q_candidates")
    if q_candidate_seeds is not None:
        if q_candidate_seeds.ndim == 1:
            q_candidate_seeds = q_candidate_seeds[None, :]
        if q_candidate_seeds.ndim != 2 or q_candidate_seeds.shape[0] == 0:
            raise ValueError(
                "current.q_candidates must have shape (K, dof), got "
                f"{q_candidate_seeds.shape}."
            )
        if not np.isfinite(q_candidate_seeds).all():
            raise ValueError("current.q_candidates contains NaN or infinity.")

    if q_candidate_seeds is not None:
        if using_pose_driven_grasp and len(q_candidate_seeds) != pose_candidate_count:
            raise ValueError(
                "When current.grasp_poses is supplied, current.q_candidates "
                "is optional but, if present, must contain one warm-start seed "
                f"per pose: {len(q_candidate_seeds)} versus {pose_candidate_count}."
            )
        q_start_candidates = q_candidate_seeds
        q_seed_source = "current.q_candidates"
    else:
        if q_single_seed is not None:
            seed = q_single_seed
            q_seed_source = "current.q"
        elif using_pose_driven_grasp:
            # The nominal configuration only seeds pose-driven optimization.
            seed = PANDA_NOMINAL_Q
            q_seed_source = "panda_nominal_fallback"
        else:
            raise PipelineInputError(
                "Provide current.grasp_poses, or provide legacy "
                "current.q_candidates/current.q.",
                missing=("current.grasp_poses or current.q_candidates/current.q",),
            )
        q_start_candidates = (
            np.repeat(seed[None, :], pose_candidate_count, axis=0)
            if using_pose_driven_grasp else seed[None, :]
        )

    candidate_count = int(q_start_candidates.shape[0])

    grasp_scores = load_current_array(grasp_scores_path, "grasp_scores")
    if grasp_scores is not None:
        grasp_scores = grasp_scores.reshape(-1)
        if len(grasp_scores) != candidate_count:
            raise ValueError(
                "current.grasp_scores length must match grasp candidates: "
                f"{len(grasp_scores)} versus {candidate_count}."
            )

    return GraspCandidateInputs(
        raw_grasp_poses=raw_grasp_poses,
        q_start_candidates=q_start_candidates,
        grasp_scores=grasp_scores,
        q_seed_source=q_seed_source,
    )


def prepare_demo_motion(
    demo_config: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    *,
    base_dir: Path,
    demo_camera: CameraBundle,
    demo_depth: FloatArray,
    demo_grounding_payload: Mapping[str, Any],
    world_up: Sequence[float],
    grasp_patch_radius: int,
    controlled_frame_method: str,
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Prepare demo motion; return object/grasp trajectories and frame metadata.

    Prefer an explicit grasp trajectory, then an object trajectory, then ObjectFlow.
    SAM3D changes object axes while preserving the demonstrated grasp trajectory.
    """
    demo_grasp_path = resolve_path(demo_config.get("grasp_trajectory"), base_dir)
    demo_object_path = resolve_path(demo_config.get("object_trajectory"), base_dir)
    demo_flow_path = resolve_path(demo_config.get("object_flow"), base_dir)

    demo_attachment: Optional[GraspAttachment] = None

    if demo_grasp_path is not None and demo_grasp_path.exists():
        demo_grasp_trajectory = load_transform(demo_grasp_path, allow_trajectory=True)
        if demo_grasp_trajectory.shape == (4, 4):
            raise PipelineInputError(
                "demo.grasp_trajectory contains only one pose. A full (T,4,4) trajectory is required."
            )
        if demo_object_path is None or not demo_object_path.exists():
            raise PipelineInputError(
                "When demo.grasp_trajectory is supplied directly, also provide "
                "demo.object_trajectory so Step 6 can define the demo effect attachment.",
                missing=("demo.object_trajectory",),
            )
        demo_object_trajectory = load_transform(demo_object_path, allow_trajectory=True)
        if demo_object_trajectory.shape == (4, 4):
            raise PipelineInputError("demo.object_trajectory must be a full (T,4,4) trajectory.")
        demo_attachment = GraspAttachment(
            np.linalg.inv(demo_object_trajectory[0]) @ demo_grasp_trajectory[0]
        )
    else:
        if demo_object_path is not None and demo_object_path.exists():
            demo_object_trajectory = load_transform(demo_object_path, allow_trajectory=True)
            if demo_object_trajectory.shape == (4, 4):
                raise PipelineInputError("demo.object_trajectory must be a full (T,4,4) trajectory.")
        elif demo_flow_path is not None and demo_flow_path.exists():
            positions_world, valid_mask = load_object_flow(demo_flow_path)
            object_result = estimate_object_pose_trajectory(
                positions_world,
                valid_mask,
                world_up=world_up,
                camera_position_world=(
                    demo_camera.transform_world_camera[:3, 3]
                ),
                inlier_threshold_m=float(
                    reconstruction.get("object_flow_inlier_threshold_m", 0.01)
                ),
                **object_flow_pose_options(reconstruction),
            )
            demo_object_trajectory = object_result.transforms
        else:  # Protected by validate_minimum_config.
            raise PipelineInputError("No demonstrated motion source was provided.")

        # The initial demo grounding supplies the canonical grasp point.  Use
        # the ObjectFlow/object-trajectory frame at t=0 for a consistent rigid
        # attachment over the full demonstration.
        grasp_pixel = choose_grasp_pixel(demo_grounding_payload, demo_depth.shape)
        grasp_point_world, _ = backproject_pixel_to_world(
            demo_depth,
            grasp_pixel,
            camera_model(demo_camera),
            patch_radius=grasp_patch_radius,
        )
        demo_attachment = build_grasp_attachment(
            demo_object_trajectory[0],
            grasp_point_world,
        )
        demo_grasp_trajectory = object_trajectory_to_grasp_trajectory(
            demo_object_trajectory,
            demo_attachment,
        )

    # Choose the object axes once; keep physical object/grasp translations and
    # the demonstrated grasp trajectory. T_grasp_effect carries the new object axes.
    frame_details: dict[str, Any] = {"method": controlled_frame_method}
    if controlled_frame_method == "sam3d":
        sam3d_path = required_path(demo_config, "sam3d_pose", base_dir)
        rotation, frame_details = load_sam3d_controlled_rotation(
            sam3d_path, demo_camera, demo_object_trajectory[0, :3, 3],
            demo_depth.shape, world_up,
        )
        change_of_basis = np.eye(4, dtype=np.float64)
        change_of_basis[:3, :3] = demo_object_trajectory[0, :3, :3].T @ rotation
        demo_object_trajectory = demo_object_trajectory @ change_of_basis
        demo_attachment = GraspAttachment(
            np.linalg.inv(demo_object_trajectory[0]) @ demo_grasp_trajectory[0]
        )

    if demo_object_trajectory.shape[0] != demo_grasp_trajectory.shape[0]:
        raise ValueError(
            "Demo object and grasp trajectories have different lengths: "
            f"{demo_object_trajectory.shape[0]} vs {demo_grasp_trajectory.shape[0]}."
        )

    prepared_demo = prepare_demo_trajectory(demo_grasp_trajectory)
    demo_grasp_trajectory = prepared_demo.poses

    return demo_object_trajectory, demo_grasp_trajectory, frame_details


def create_entity_registrar(
    *,
    scene_inputs: SceneInputs,
    demo_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
    base_dir: Path,
    demo_camera: CameraBundle,
    current_camera: CameraBundle,
    sam2_model_path: str,
    registration_device: str,
    world_up: Sequence[float],
    relative_pose_options: Mapping[str, Any],
    output_dir: Path,
    manifest: dict[str, Any],
) -> Callable[..., RegisteredTargetPoses]:
    """Create a registrar that lazily shares RGB images and SAM2 across entities."""
    relative_pose_segmenter: Optional[Sam2BoxSegmenter] = None
    registration_images = None

    def register_entity(
        entity_name: str,
        demo_bbox: Sequence[float],
        current_bbox: Sequence[float],
        rotation_constraint: str,
        *,
        demo_entity_pose_world: Optional[FloatArray] = None,
        demo_frame_transform: Optional[Callable[[FloatArray], FloatArray]] = None,
    ) -> RegisteredTargetPoses:
        nonlocal relative_pose_segmenter, registration_images
        # Load only when registration is needed, then reuse for the second entity.
        if registration_images is None:
            registration_images = tuple(
                load_rgb(
                    required_first_path(
                        section, ("rgb", "initial_rgb"), base_dir, label=f"{name}.rgb",
                    )
                )
                for name, section in (("demo", demo_config), ("current", current_config))
            )
        demo_rgb, current_rgb = registration_images
        if relative_pose_segmenter is None:
            relative_pose_segmenter = Sam2BoxSegmenter(sam2_model_path, registration_device)
        try:
            result = estimate_registered_entity_poses_from_loftr(
                demo_rgb=demo_rgb,
                demo_depth=scene_inputs.demo_depth,
                demo_bbox=demo_bbox,
                demo_intrinsic_matrix=demo_camera.intrinsic_matrix,
                transform_world_camera_demo=demo_camera.transform_world_camera,
                demo_depth_scale_to_m=demo_camera.depth_scale_to_m,
                current_rgb=current_rgb,
                current_depth=scene_inputs.current_depth,
                current_bbox=current_bbox,
                current_intrinsic_matrix=current_camera.intrinsic_matrix,
                transform_world_camera_current=current_camera.transform_world_camera,
                current_depth_scale_to_m=current_camera.depth_scale_to_m,
                demo_entity_pose_world=demo_entity_pose_world,
                demo_frame_transform=demo_frame_transform,
                sam2_model_path=sam2_model_path,
                device=registration_device,
                world_up=world_up,
                rotation_constraint=rotation_constraint,
                segmenter=relative_pose_segmenter,
                **relative_pose_options,
            )
        except Exception as error:
            label = "controlled-object" if entity_name == "controlled_object" else "object_pose_6d"
            raise PipelineInputError(
                f"Failed {label} LoFTR relative registration "
                f"(rotation_constraint={rotation_constraint!r}): {error}"
            ) from error

        registration_dir = output_dir / entity_name
        save_registered_target_pose_result(
            registration_dir, result,
            demo_rgb=demo_rgb,
            current_rgb=current_rgb,
            demo_bbox=demo_bbox,
            current_bbox=current_bbox,
            entity_name=entity_name,
        )
        manifest[f"{entity_name}_pose_estimation"] = {
            "method": (
                "bbox-crop LoFTR + SAM2 mask filtering + RGB-D 3D RANSAC + "
                f"{rotation_constraint} refit"
            ),
            "rotation_constraint": rotation_constraint,
            "artifacts": str(registration_dir),
            "registration": asdict(result.registration),
        }
        return result

    return register_entity


def prepare_controlled_object(
    scene_inputs: SceneInputs,
    demo_object_trajectory: FloatArray,
    demo_grasp_trajectory: FloatArray,
    *,
    current_camera: CameraBundle,
    current_object_path: Optional[Path],
    current_grasp_path: Optional[Path],
    same_scene_identity_registration: bool,
    controlled_entity_rotation_constraint: str,
    grasp_patch_radius: int,
    register_entity: Callable[..., RegisteredTargetPoses],
    manifest: dict[str, Any],
) -> ControlledObjectState:
    """Resolve the current object pose, grasp pose and rigid grasp attachment."""
    current_grounding_payload = scene_inputs.initial_adaptation.current_grounding
    controlled_pose_result = None
    inferred_grasp: Optional[FloatArray] = None
    if same_scene_identity_registration:
        # The current initial scene is intentionally identical to the demo initial
        # scene. Do not estimate a demo->current relative pose. Reuse the demo
        # initial controlled-object and grasp poses exactly, so:
        #
        #   T_current_from_demo = I
        #   T_W_object_current = T_W_object_demo(0)
        #   T_W_grasp_current  = T_W_grasp_demo(0)
        current_object = np.asarray(demo_object_trajectory[0], dtype=np.float64).copy()
        current_grasp = np.asarray(demo_grasp_trajectory[0], dtype=np.float64).copy()
        current_attachment = GraspAttachment(
            np.linalg.inv(current_object) @ current_grasp
        )
        manifest["controlled_object_pose_estimation"] = {
            "method": "same_scene_identity",
            "relative_transform_current_from_demo": np.eye(4, dtype=np.float64).tolist(),
            "registration_skipped": True,
        }
    else:
        if current_object_path is not None and current_object_path.exists():
            current_object = load_transform(current_object_path)
        else:
            demo_controlled = scene_inputs.demo_grounding_payload.get("controlled_entity")
            current_controlled = current_grounding_payload.get("controlled_entity")
            if not isinstance(demo_controlled, Mapping) or demo_controlled.get("bbox") is None:
                raise PipelineInputError(
                    "Controlled-object relative registration requires "
                    "demo_initial_grounding.controlled_entity.bbox."
                )
            if (
                not isinstance(current_controlled, Mapping)
                or current_controlled.get("bbox") is None
            ):
                raise PipelineInputError(
                    "Controlled-object relative registration requires "
                    "current_grounding.controlled_entity.bbox."
                )

            controlled_pose_result = register_entity(
                "controlled_object",
                demo_controlled["bbox"],
                current_controlled["bbox"],
                controlled_entity_rotation_constraint,
                demo_entity_pose_world=demo_object_trajectory[0],
            )

            current_object = controlled_pose_result.transform_world_target_current
            transform_world_current_from_demo_object = (
                current_object @ np.linalg.inv(demo_object_trajectory[0])
            )
            inferred_grasp = (
                transform_world_current_from_demo_object
                @ demo_grasp_trajectory[0]
            )
            current_attachment = GraspAttachment(
                np.linalg.inv(current_object) @ inferred_grasp
            )

        if current_grasp_path is not None and current_grasp_path.exists():
            current_grasp = load_transform(current_grasp_path)
            current_attachment = GraspAttachment(np.linalg.inv(current_object) @ current_grasp)
        elif controlled_pose_result is not None:
            assert inferred_grasp is not None
            current_grasp = inferred_grasp
        else:
            grasp_pixel = choose_grasp_pixel(current_grounding_payload, scene_inputs.current_depth.shape)
            grasp_point_world, _ = backproject_pixel_to_world(
                scene_inputs.current_depth,
                grasp_pixel,
                camera_model(current_camera),
                patch_radius=grasp_patch_radius,
            )
            current_attachment = build_grasp_attachment(current_object, grasp_point_world)
            current_grasp = current_object @ current_attachment.transform_object_grasp

    return ControlledObjectState(
        object_pose_world=current_object,
        grasp_pose_world=current_grasp,
        attachment=current_attachment,
        registration=controlled_pose_result,
    )


def prepare_target_geometry(
    scene_inputs: SceneInputs,
    *,
    demo_camera: CameraBundle,
    demo_config: Mapping[str, Any],
    base_dir: Path,
    reconstruction: Mapping[str, Any],
    world_up: Sequence[float],
    grasp_patch_radius: int,
    same_scene_identity_registration: bool,
    target_object_rotation_constraint: str,
    register_entity: Callable[..., RegisteredTargetPoses],
    paths: PipelinePaths,
    manifest: dict[str, Any],
) -> TargetGeometryInputs:
    """Prepare target groundings and optional object clouds for both scenes."""
    current_adaptation = scene_inputs.initial_adaptation
    current_grounding_payload = current_adaptation.current_grounding
    demo_target_grounding = parse_target_grounding(scene_inputs.demo_grounding_payload)
    current_target_grounding = parse_target_grounding(current_grounding_payload)
    demo_target_object_points_world: Optional[FloatArray] = None
    current_target_object_points_world: Optional[FloatArray] = None

    if current_adaptation.effective_spec.target_representation.value == "object_pose_6d":
        target_frame_method = reconstruction.get("target_object_frame_method", "pca")
        manifest["target_object_frame"] = {"method": target_frame_method}

        def target_source_frame(transform: FloatArray) -> FloatArray:
            if target_frame_method == "pca":
                return transform
            # Reuse the prompt axis convention; retain the exact existing target origin.
            result = np.asarray(transform, dtype=np.float64).copy()
            result[:3, :3], details = load_sam3d_controlled_rotation(
                required_path(demo_config, "sam3d_target_pose", base_dir),
                demo_camera, result[:3, 3], scene_inputs.demo_depth.shape, world_up,
            )
            details["origin_policy"] = "retain_target_origin"
            manifest["target_object_frame"] = details
            return result

        if same_scene_identity_registration:
            # Reconstruct the target once from the demo initial RGB-D geometry,
            # define its canonical demo frame once, and reuse both for current.
            # No demo->current image/depth registration is performed.
            target_depth_band_m = float(
                reconstruction.get("target_depth_band_m", 0.04)
            )
            target_bbox_stride = int(
                reconstruction.get("target_bbox_stride", 2)
            )
            target_minimum_points = int(
                reconstruction.get("target_minimum_points", 50)
            )

            demo_target_object_points_world = reconstruct_target_object_points_from_scene(
                scene_inputs.demo_depth,
                scene_inputs.demo_grounding_payload,
                camera_model(demo_camera),
                depth_band_m=target_depth_band_m,
                bbox_stride=target_bbox_stride,
                minimum_points=target_minimum_points,
                patch_radius=grasp_patch_radius,
            )
            current_target_object_points_world = np.asarray(
                demo_target_object_points_world, dtype=np.float64
            ).copy()

            if demo_target_grounding.transform_world_target is not None:
                transform_world_target_demo = np.asarray(
                    demo_target_grounding.transform_world_target,
                    dtype=np.float64,
                ).copy()
            else:
                transform_world_target_demo, _ = compute_pca_target_pose(
                    demo_target_object_points_world,
                    world_up=world_up,
                    camera_position_world=demo_camera.transform_world_camera[:3, 3],
                )

            transform_world_target_demo = target_source_frame(transform_world_target_demo)
            transform_world_target_current = np.asarray(
                transform_world_target_demo, dtype=np.float64
            ).copy()

            identity_dir = paths.registration / "target"
            identity_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                identity_dir / "same_scene_identity.json",
                {
                    "registration_skipped": True,
                    "relative_transform_current_from_demo": np.eye(
                        4, dtype=np.float64
                    ).tolist(),
                    "T_world_target_demo": transform_world_target_demo,
                    "T_world_target_current": transform_world_target_current,
                },
            )
            manifest["target_pose_estimation"] = {
                "method": "same_scene_identity",
                "registration_skipped": True,
                "relative_transform_current_from_demo": np.eye(
                    4, dtype=np.float64
                ).tolist(),
                "artifacts": str(identity_dir),
            }
        else:
            if demo_target_grounding.target_object_bbox is None:
                raise PipelineInputError(
                    "object_pose_6d requires demo target_object_bbox.",
                    missing=("demo_initial_grounding.target_geometry.target_object_bbox",),
                )
            if current_target_grounding.target_object_bbox is None:
                raise PipelineInputError(
                    "object_pose_6d requires current target_object_bbox.",
                    missing=("current_grounding.target_geometry.target_object_bbox",),
                )

            # Retain the segmented-cloud origin and optionally select SAM3D axes.
            target_pose_result = register_entity(
                "target",
                demo_target_grounding.target_object_bbox,
                current_target_grounding.target_object_bbox,
                target_object_rotation_constraint,
                demo_frame_transform=target_source_frame,
            )
            demo_target_object_points_world = target_pose_result.demo_points_world
            current_target_object_points_world = target_pose_result.current_points_world
            transform_world_target_demo = target_pose_result.transform_world_target_demo
            transform_world_target_current = target_pose_result.transform_world_target_current

        demo_target_grounding = replace(
            demo_target_grounding, transform_world_target=transform_world_target_demo,
        )
        current_target_grounding = replace(
            current_target_grounding, transform_world_target=transform_world_target_current,
        )

    return TargetGeometryInputs(
        demo_grounding=demo_target_grounding,
        current_grounding=current_target_grounding,
        demo_points_world=demo_target_object_points_world,
        current_points_world=current_target_object_points_world,
    )


def optimize_motion_reference(
    reference: MotionReference,
    q_start_candidates: FloatArray,
    *,
    using_multi_grasp_input: bool,
    optimizer_config: TrajectoryOptimizerConfig,
    robot_config: Mapping[str, Any],
    multi_grasp_mapping: Mapping[str, Any],
    initial_trajectory: Optional[FloatArray],
    sphere_json: Optional[Path],
    grasp_scores: Optional[FloatArray],
    expected_hand_poses_world: Optional[FloatArray],
    current_grasp: FloatArray,
    output_dir: Path,
    grasp_output_dir: Path,
) -> tuple[TrajectoryOptimizationResult, int]:
    """Optimize one or more starts, validate the chosen grasp, and save the plan."""
    best_candidate_index = 0

    if using_multi_grasp_input:
        result, best_candidate_index, _ = (
            run_multi_start_trajectory_optimization(
                reference=reference,
                q_start_candidates=q_start_candidates,
                output_dir=grasp_output_dir,
                result_output_dir=output_dir,
                urdf_description=str(
                    robot_config.get("urdf_description", "panda_description")
                ),
                target_link=str(
                    robot_config.get("target_link", "panda_hand_tcp")
                ),
                config=optimizer_config,
                sphere_json=sphere_json,
                graspgen_scores=grasp_scores,
                expected_hand_poses_world=expected_hand_poses_world,
                max_position_error_m=multi_grasp_mapping.get(
                    "max_position_error_m"
                ),
                max_axis_error_deg=multi_grasp_mapping.get(
                    "max_axis_error_deg"
                ),
                max_rotation_error_deg=multi_grasp_mapping.get(
                    "max_rotation_error_deg"
                ),
                max_grasp_pose_position_error_m=multi_grasp_mapping.get(
                    "max_grasp_pose_position_error_m"
                ),
                max_grasp_pose_rotation_error_deg=multi_grasp_mapping.get(
                    "max_grasp_pose_rotation_error_deg"
                ),
                clear_jax_caches_between_candidates=bool(
                    multi_grasp_mapping.get(
                        "clear_jax_caches_between_candidates",
                        True,
                    )
                ),
            )
        )

    else:
        result = run_trajectory_optimization(
            reference,
            q_start_candidates[0],
            urdf_description=str(
                robot_config.get("urdf_description", "panda_description")
            ),
            target_link=str(
                robot_config.get("target_link", "panda_hand_tcp")
            ),
            config=optimizer_config,
            initial_trajectory=initial_trajectory,
            sphere_json=sphere_json,
            desired_start_hand_pose_world=(
                None
                if expected_hand_poses_world is None
                else expected_hand_poses_world[0]
            ),
            transform_world_grasp_current=current_grasp,
        )
        if expected_hand_poses_world is not None:
            position_error_m, rotation_error_deg = grasp_pose_error(
                result.achieved_hand_poses_world[0],
                expected_hand_poses_world[0],
            )
            position_limit = multi_grasp_mapping.get(
                "max_grasp_pose_position_error_m"
            )
            rotation_limit = multi_grasp_mapping.get(
                "max_grasp_pose_rotation_error_deg"
            )
            if position_limit is not None and position_error_m > float(position_limit):
                raise PipelineInputError(
                    "Single-grasp IK/FK does not match the supplied GraspGen pose: "
                    f"position error {position_error_m:.6f} m exceeds "
                    f"{float(position_limit):.6f} m."
                )
            if rotation_limit is not None and rotation_error_deg > float(rotation_limit):
                raise PipelineInputError(
                    "Single-grasp IK/FK does not match the supplied GraspGen pose: "
                    f"rotation error {rotation_error_deg:.3f} deg exceeds "
                    f"{float(rotation_limit):.3f} deg."
                )

    # Multi-start already saved the selected plan in this directory.
    if not using_multi_grasp_input:
        save_optimization_result(output_dir, result, reference=reference)

    return result, best_candidate_index


def export_pipeline_results(
    result: TrajectoryOptimizationResult,
    *,
    paths: PipelinePaths,
    manifest: dict[str, Any],
    grasp_inputs: GraspCandidateInputs,
    using_multi_grasp_input: bool,
    expected_hand_poses_world: Optional[FloatArray],
    best_candidate_index: int,
    dt_s: float,
    sphere_json: Optional[Path],
    legacy_direct_grounding: bool,
) -> dict[str, Any]:
    """Save metrics, execution trajectories and the completed run manifest."""
    optimization_summary = result_summary(result)
    optimization_summary["grasp_candidates"] = {
        "mode": (
            "multi_grasp"
            if using_multi_grasp_input and expected_hand_poses_world is not None
            else "multi_start"
            if using_multi_grasp_input
            else "single_grasp"
        ),
        "candidate_count": int(len(grasp_inputs.q_start_candidates)),
        "best_candidate_index": int(best_candidate_index),
        "grasp_pose_validation_enabled": expected_hand_poses_world is not None,
        "q_seed_source": grasp_inputs.q_seed_source,
    }
    # Retain the old key for downstream readers.
    optimization_summary["multi_grasp"] = {
        "enabled": bool(using_multi_grasp_input),
        "candidate_count": int(len(grasp_inputs.q_start_candidates)),
        "best_candidate_index": int(best_candidate_index),
    }
    save_json(
        paths.trajectory / "optimization_metrics.json",
        optimization_summary,
    )

    pose7 = transform_trajectory_to_pose7(result.achieved_hand_poses_world)
    np.save(paths.trajectory / "ee_trajectory.npy", pose7)
    np.savetxt(
        paths.trajectory / "ee_trajectory.csv",
        pose7,
        delimiter=",",
        header="x,y,z,qx,qy,qz,qw",
        comments="",
    )
    np.save(paths.trajectory / "joint_trajectory.npy", result.planned_joints)
    save_json(
        paths.trajectory / "trajectory_summary.json",
        {
            "joint_trajectory_shape": result.planned_joints.shape,
            "ee_trajectory_shape": pose7.shape,
            "quaternion_order": "qx,qy,qz,qw",
            "dt_s": dt_s,
            "first_ee_pose": pose7[0],
            "last_ee_pose": pose7[-1],
        },
    )

    if sphere_json is None:
        manifest["warnings"].append(
            "Collision checking was disabled because robot.sphere_json was not supplied."
        )
    if grasp_inputs.q_seed_source == "panda_nominal_fallback":
        manifest["warnings"].append(
            "Neither current.q nor current.q_candidates was supplied. "
            "The standard Panda nominal configuration was repeated as the warm-start "
            "seed for every grasp pose; the supplied 6D grasp poses remain the active "
            "start constraints and q[0:T] is optimized jointly."
        )
    if using_multi_grasp_input and expected_hand_poses_world is None:
        manifest["warnings"].append(
            "Multiple q candidates were supplied without current.grasp_poses. "
            "They are treated as multiple IK starts for the same canonical "
            "object-attached grasp frame."
        )
    if legacy_direct_grounding:
        manifest["warnings"].append(
            "Legacy current scene-grounding JSON was used, so demonstration rotation "
            "policies were kept unchanged."
        )
    manifest["status"] = "complete"
    manifest["artifacts"].update(
        {
            "optimization_result": str(paths.trajectory / "optimization_result.npz"),
            "desired_effect_poses_world": str(paths.trajectory / "desired_effect_poses_world.npy"),
            "optimization_metrics": str(paths.trajectory / "optimization_metrics.json"),
            "trajectory_summary": str(paths.trajectory / "trajectory_summary.json"),
            "execution_joint_trajectory": str(paths.trajectory / "joint_trajectory.npy"),
            "execution_ee_trajectory_npy": str(paths.trajectory / "ee_trajectory.npy"),
            "execution_ee_trajectory_csv": str(paths.trajectory / "ee_trajectory.csv"),
        }
    )
    if using_multi_grasp_input:
        manifest["artifacts"]["multi_grasp_candidate_comparison"] = str(
            paths.grasps / "candidate_comparison.json"
        )
    save_json(paths.trajectory / "run_manifest.json", manifest)
    return manifest


def run_pipeline(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    check_only: bool = False,
    skip_optimization: bool = False,
) -> dict[str, Any]:
    base_dir = config_path.resolve().parent
    missing = validate_minimum_config(config, base_dir)

    # Validate inputs before --overwrite can remove a previous run.
    if missing:
        raise PipelineInputError(
            "Cannot run the pipeline because required inputs are missing.",
            missing=missing,
        )

    common_camera = config.get("camera", {})
    demo_camera_config = merge_mappings(common_camera, config.get("demo_camera"))
    current_camera_config = merge_mappings(common_camera, config.get("current_camera"))
    demo_camera = load_camera_bundle(demo_camera_config, base_dir)
    current_camera = load_camera_bundle(current_camera_config, base_dir)

    # Flow and grasp estimation can write inputs inside this run before planning.
    flow_path = resolve_path(config["demo"].get("object_flow"), base_dir)
    grasp_path = None
    if not config["current"].get("use_given_q", False):
        grasp_path, _ = resolve_first_path(
            config["current"], ("grasp_poses", "grasp_pose"), base_dir,
        )
    paths = create_output_tree(
        output_dir, overwrite=overwrite,
        preserve=[path.parent for path in (flow_path, grasp_path) if path is not None],
    )

    demo_config = config["demo"]
    current_config = config["current"]
    robot_config = config.get("robot", {})
    reconstruction = config.get("reconstruction", {})
    scene_parameters = config.get("scene_geometry", {})
    optimizer_mapping = config.get("optimizer", {})
    multi_grasp_mapping = config.get("multi_grasp", {})
    target_pose_mapping = config.get("target_pose_registration", {})
    relative_pose_options = loftr_relative_registration_options(
        target_pose_mapping if isinstance(target_pose_mapping, Mapping) else {}
    )
    same_scene_identity_registration = bool(
        reconstruction.get("same_scene_identity_registration", False)
    )

    manifest: dict[str, Any] = {
        "status": "checked" if check_only else "running",
        "config": str(config_path.resolve()),
        "output_dir": str(paths.root),
        "camera_assumptions": {
            "demo": demo_camera.assumptions,
            "current": current_camera.assumptions,
        },
        "warnings": [],
        "artifacts": {},
    }
    if demo_camera.assumptions or current_camera.assumptions:
        manifest["warnings"].append(
            "Verify the camera-extrinsic direction assumptions in run_manifest.json."
        )
    if check_only:
        save_json(paths.trajectory / "run_manifest.json", manifest)
        return manifest

    demo_depth_path = required_path(demo_config, "initial_depth", base_dir)
    demo_analysis_path = required_first_path(
        demo_config,
        ("analysis", "video_grounding"),
        base_dir,
        label="demo.analysis",
    )
    demo_grounding_path = resolve_path(demo_config.get("scene_grounding"), base_dir)
    current_depth_path = required_path(current_config, "initial_depth", base_dir)
    current_adaptation_path = required_first_path(
        current_config,
        ("adaptation", "scene_grounding"),
        base_dir,
        label="current.adaptation",
    )

    q_candidates_path = resolve_path(
        current_config.get(
            "q_candidates",
            current_config.get("q_start_candidates"),
        ),
        base_dir,
    )
    q_single_path = resolve_path(current_config.get("q"), base_dir)
    grasp_scores_path = resolve_path(
        current_config.get(
            "grasp_scores",
            current_config.get("graspgen_scores"),
        ),
        base_dir,
    )
    grasp_poses_path, grasp_poses_key = resolve_first_path(
        current_config,
        ("grasp_poses", "grasp_pose"),
        base_dir,
    )

    use_given_q = bool(current_config.get("use_given_q", False))

    if use_given_q:
        if q_single_path is None or not q_single_path.exists():
            raise PipelineInputError(
                "current.use_given_q=true requires a valid current.q file.",
                missing=("current.q",),
            )

        # Force legacy joint-driven single-start mode.
        q_candidates_path = None
        grasp_scores_path = None
        grasp_poses_path = None
        grasp_poses_key = None

        manifest["warnings"].append(
            "current.use_given_q=true: using current.q as the fixed initial "
            "robot configuration; q_candidates, grasp_poses, and grasp_scores "
            "are ignored."
        )

    scene_inputs = load_scene_inputs(
        demo_depth_path=demo_depth_path,
        current_depth_path=current_depth_path,
        demo_analysis_path=demo_analysis_path,
        demo_grounding_path=demo_grounding_path,
        current_adaptation_path=current_adaptation_path,
    )
    current_adaptation = scene_inputs.initial_adaptation
    current_grounding_payload = current_adaptation.current_grounding

    grasp_inputs = prepare_grasp_candidates(
        q_single_path=q_single_path,
        q_candidates_path=q_candidates_path,
        grasp_poses_path=grasp_poses_path,
        grasp_poses_key=grasp_poses_key,
        grasp_scores_path=grasp_scores_path,
    )
    candidate_count = int(grasp_inputs.q_start_candidates.shape[0])
    using_multi_grasp_input = candidate_count > 1

    depth_band_m = float(reconstruction.get("object_depth_band_m", 0.04))
    bbox_stride = int(reconstruction.get("object_bbox_stride", 2))
    minimum_points = int(reconstruction.get("object_minimum_points", 50))
    grasp_patch_radius = int(reconstruction.get("grasp_patch_radius", 2))
    world_up = tuple(reconstruction.get("world_up", (0.0, 0.0, 1.0)))
    controlled_frame_method = str(reconstruction.get("controlled_object_frame_method", "pca"))
    controlled_entity_rotation_constraint = str(
        target_pose_mapping.get("controlled_entity_rotation_constraint", "yaw_only")
    )
    target_object_rotation_constraint = str(
        target_pose_mapping.get("target_object_rotation_constraint", "full_6d")
    )
    sam2_model_path = str(
        target_pose_mapping.get("sam2_model_path", "facebook/sam2-hiera-large")
    )
    registration_device = str(target_pose_mapping.get("device", "cuda"))

    demo_object_trajectory, demo_grasp_trajectory, frame_details = prepare_demo_motion(
        demo_config,
        reconstruction,
        base_dir=base_dir,
        demo_camera=demo_camera,
        demo_depth=scene_inputs.demo_depth,
        demo_grounding_payload=scene_inputs.demo_grounding_payload,
        world_up=world_up,
        grasp_patch_radius=grasp_patch_radius,
        controlled_frame_method=controlled_frame_method,
    )
    manifest["controlled_object_frame"] = frame_details

    # Build one demo-initial controlled-object cloud for optional mesh-free
    # current-scene registration. It is expressed in the demonstrated canonical
    # object frame so it remains independent of the demo world placement.
    demo_points_world, _, _ = reconstruct_controlled_object_points_from_scene(
        scene_inputs.demo_depth,
        scene_inputs.demo_grounding_payload,
        camera_model(demo_camera),
        depth_band_m=depth_band_m,
        bbox_stride=bbox_stride,
        minimum_points=minimum_points,
        grasp_patch_radius=grasp_patch_radius,
    )
    demo_points_object = points_world_to_object(
        demo_object_trajectory[0],
        demo_points_world,
    )

    # Register the controlled object and target with their own rotation constraints.
    current_object_path = resolve_path(current_config.get("T_world_object"), base_dir)
    current_grasp_path = resolve_path(current_config.get("T_world_grasp"), base_dir)
    register_entity = create_entity_registrar(
        scene_inputs=scene_inputs,
        demo_config=demo_config,
        current_config=current_config,
        base_dir=base_dir,
        demo_camera=demo_camera,
        current_camera=current_camera,
        sam2_model_path=sam2_model_path,
        registration_device=registration_device,
        world_up=world_up,
        relative_pose_options=relative_pose_options,
        output_dir=paths.registration,
        manifest=manifest,
    )

    controlled_state = prepare_controlled_object(
        scene_inputs,
        demo_object_trajectory,
        demo_grasp_trajectory,
        current_camera=current_camera,
        current_object_path=current_object_path,
        current_grasp_path=current_grasp_path,
        same_scene_identity_registration=same_scene_identity_registration,
        controlled_entity_rotation_constraint=controlled_entity_rotation_constraint,
        grasp_patch_radius=grasp_patch_radius,
        register_entity=register_entity,
        manifest=manifest,
    )

    expected_hand_poses_world: Optional[FloatArray]
    if grasp_inputs.raw_grasp_poses is None:
        expected_hand_poses_world = None
    else:
        expected_hand_poses_world = grasp_poses_to_world(
            grasp_inputs.raw_grasp_poses,
            frame=str(current_config.get("grasp_pose_frame", "world")),
            transform_world_camera=current_camera.transform_world_camera,
            transform_world_object=controlled_state.object_pose_world,
        )
        if len(expected_hand_poses_world) != candidate_count:
            raise ValueError(
                "The number of current grasp poses must match the number of "
                f"optimization candidates: {len(expected_hand_poses_world)} versus "
                f"{candidate_count}."
            )

    if same_scene_identity_registration:
        current_points_world = np.asarray(demo_points_world, dtype=np.float64).copy()
    elif controlled_state.registration is not None:
        current_points_world = controlled_state.registration.current_points_world
    else:
        current_points_world, _, _ = reconstruct_controlled_object_points_from_scene(
            scene_inputs.current_depth,
            current_grounding_payload,
            camera_model(current_camera),
            depth_band_m=depth_band_m,
            bbox_stride=bbox_stride,
            minimum_points=minimum_points,
            grasp_patch_radius=grasp_patch_radius,
        )
    # Build geometry for the demo and current initial scenes.
    scene_kwargs = {
        "point_patch_radius": int(scene_parameters.get("point_patch_radius", 2)),
        "plane_patch_radius": int(scene_parameters.get("plane_patch_radius", 8)),
        "plane_patch_stride": int(scene_parameters.get("plane_patch_stride", 2)),
        "preferred_normal_world": tuple(
            scene_parameters.get("preferred_normal_world", (0.0, 0.0, 1.0))
        ),
    }

    target_geometry = prepare_target_geometry(
        scene_inputs,
        demo_camera=demo_camera,
        demo_config=demo_config,
        base_dir=base_dir,
        reconstruction=reconstruction,
        world_up=world_up,
        grasp_patch_radius=grasp_patch_radius,
        same_scene_identity_registration=same_scene_identity_registration,
        target_object_rotation_constraint=target_object_rotation_constraint,
        register_entity=register_entity,
        paths=paths,
        manifest=manifest,
    )

    poses = {
        "frame_convention": "Object-to-world transforms; translation in meters.",
        "controlled_object": {
            "demo": demo_object_trajectory[0],
            "current": controlled_state.object_pose_world,
        },
    }
    if (
        target_geometry.demo_grounding.transform_world_target is not None
        or target_geometry.current_grounding.transform_world_target is not None
    ):
        poses["target"] = {
            "demo": target_geometry.demo_grounding.transform_world_target,
            "current": target_geometry.current_grounding.transform_world_target,
        }
    poses_path = paths.registration / "poses.json"
    save_json(poses_path, poses)
    manifest["artifacts"]["object_poses"] = str(poses_path)

    demo_scene_geometry = build_scene_geometry(
        depth=scene_inputs.demo_depth,
        camera=camera_model(demo_camera),
        grounding=target_geometry.demo_grounding,
        transform_world_object_current=demo_object_trajectory[0],
        transform_world_grasp_current=demo_grasp_trajectory[0],
        target_object_points_world=target_geometry.demo_points_world,
        **scene_kwargs,
    )
    current_scene_geometry = build_scene_geometry(
        depth=scene_inputs.current_depth,
        camera=camera_model(current_camera),
        grounding=target_geometry.current_grounding,
        transform_world_object_current=controlled_state.object_pose_world,
        transform_world_grasp_current=controlled_state.grasp_pose_world,
        target_object_points_world=target_geometry.current_points_world,
        **scene_kwargs,
    )

    # Build the task-space motion reference for the current scene.
    try:
        current_adaptation = parse_current_adaptation(
            scene_inputs.current_adaptation_payload,
            scene_inputs.video_structure_payload,
            total_frames=demo_grasp_trajectory.shape[0],
            allow_legacy_direct_grounding=True,
        )
    except CurrentAdaptationValidationError as error:
        raise PipelineInputError(str(error)) from error

    effective_spec = current_adaptation.effective_spec

    reference = build_motion_reference(
        demo_grasp_trajectory,
        effective_spec,
        demo_scene_geometry,
        current_scene_geometry,
        transform_world_grasp_current=controlled_state.grasp_pose_world,
        plane_press_down_m=float(
            config.get("motion_reference", {}).get("plane_press_down_m", 0.0)
        ),
    )
    if skip_optimization:
        reference_path = paths.trajectory / "motion_reference.npz"
        save_motion_reference(reference_path, reference)
        manifest["artifacts"]["motion_reference"] = str(reference_path)
        manifest["status"] = "reference_built_optimization_skipped"
        manifest["warnings"].append(
            "Optimization was skipped; no joint or Cartesian execution trajectory was produced."
        )
        save_json(paths.trajectory / "run_manifest.json", manifest)
        return manifest

    save_json(paths.trajectory / "config_used.json", config)
    manifest["artifacts"]["config_used"] = str(paths.trajectory / "config_used.json")

    # Keep camera values, but reference task RGB-D instead of copying large inputs.
    replay_scene = config.get("replay_scene", {})
    replay_rgb = resolve_path(
        replay_scene.get("rgb") or current_config.get("rgb") or current_config.get("initial_rgb"),
        base_dir,
    )
    if replay_rgb is not None:
        replay_depth = resolve_path(
            replay_scene.get("depth") or current_config["initial_depth"], base_dir,
        )
        scene_path = paths.trajectory / "replay_scene.json"
        save_json(scene_path, {
            "rgb": relpath(replay_rgb, paths.trajectory),
            "depth": relpath(replay_depth, paths.trajectory),
            "intrinsic_matrix": current_camera.intrinsic_matrix,
            "T_world_camera": current_camera.transform_world_camera,
            "depth_scale_to_m": current_camera.depth_scale_to_m,
        })
        manifest["artifacts"]["replay_scene"] = str(scene_path)

    # Optimize joint motion and export the TCP trajectory.
    optimizer_config = optimizer_config_from_mapping(optimizer_mapping)
    sphere_json = resolve_path(robot_config.get("sphere_json"), base_dir)
    initial_trajectory_path = resolve_path(
        optimizer_mapping.get("initial_trajectory"), base_dir
    )
    initial_trajectory = (
        None
        if initial_trajectory_path is None
        else np.asarray(np.load(initial_trajectory_path, allow_pickle=False), dtype=np.float64)
    )

    if using_multi_grasp_input and initial_trajectory is not None:
        manifest["warnings"].append(
            "optimizer.initial_trajectory is ignored for multi-grasp optimization "
            "because each grasp candidate has a different start configuration."
        )

    result, best_candidate_index = optimize_motion_reference(
        reference,
        grasp_inputs.q_start_candidates,
        using_multi_grasp_input=using_multi_grasp_input,
        optimizer_config=optimizer_config,
        robot_config=robot_config,
        multi_grasp_mapping=multi_grasp_mapping,
        initial_trajectory=initial_trajectory,
        sphere_json=sphere_json,
        grasp_scores=grasp_inputs.grasp_scores,
        expected_hand_poses_world=expected_hand_poses_world,
        current_grasp=controlled_state.grasp_pose_world,
        output_dir=paths.trajectory,
        grasp_output_dir=paths.grasps,
    )
    return export_pipeline_results(
        result,
        paths=paths,
        manifest=manifest,
        grasp_inputs=grasp_inputs,
        using_multi_grasp_input=using_multi_grasp_input,
        expected_hand_poses_world=expected_hand_poses_world,
        best_candidate_index=best_candidate_index,
        dt_s=optimizer_config.dt,
        sphere_json=sphere_json,
        legacy_direct_grounding=current_adaptation.legacy_direct_grounding,
    )
