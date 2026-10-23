from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple
import json

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .object_flow_pose import compute_pca_object_pose
from .schemas import TargetRepresentation
from .se3 import (
    as_transform,
    invert_transform,
    make_transform,
    relative_transform,
    transform_point,
)


FloatArray = NDArray[np.float64]
Pixel = Tuple[int, int]


@dataclass(frozen=True)
class CameraModel:
    """Rectified pinhole camera used with an aligned depth image.

    Convention:
        intrinsic_matrix is K for pixels [u, v].
        transform_world_camera is T_world_camera.
        depth_scale_to_m converts stored depth values to metres.
    """

    intrinsic_matrix: FloatArray
    transform_world_camera: FloatArray
    depth_scale_to_m: float = 1.0
    minimum_depth_m: float = 0.05
    maximum_depth_m: float = 5.0

    def __post_init__(self) -> None:
        intrinsic = np.asarray(self.intrinsic_matrix, dtype=np.float64)
        transform = as_transform(self.transform_world_camera)

        if intrinsic.shape != (3, 3):
            raise ValueError(
                "intrinsic_matrix must have shape (3, 3), "
                f"got {intrinsic.shape}."
            )
        if not np.all(np.isfinite(intrinsic)):
            raise ValueError("intrinsic_matrix contains non-finite values.")
        if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            raise ValueError("Camera focal lengths must be positive.")
        if self.depth_scale_to_m <= 0.0:
            raise ValueError("depth_scale_to_m must be positive.")
        if self.minimum_depth_m <= 0.0:
            raise ValueError("minimum_depth_m must be positive.")
        if self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError(
                "maximum_depth_m must be greater than minimum_depth_m."
            )

        object.__setattr__(self, "intrinsic_matrix", intrinsic.copy())
        object.__setattr__(self, "transform_world_camera", transform)


@dataclass(frozen=True)
class TargetGrounding:
    """Pixel evidence needed to instantiate one target representation."""

    representation: TargetRepresentation
    target_point_pixel: Optional[Pixel] = None
    surface_pixels: Tuple[Pixel, ...] = ()
    axis_origin_pixel: Optional[Pixel] = None
    axis_pixels: Tuple[Pixel, ...] = ()
    axis_positive_endpoint_pixel: Optional[Pixel] = None
    plane_reference_pixel: Optional[Pixel] = None
    target_object_bbox: Optional[Tuple[int, int, int, int]] = None
    target_object_positive_pixels: Tuple[Pixel, ...] = ()
    transform_world_target: Optional[FloatArray] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if self.transform_world_target is not None:
            object.__setattr__(
                self,
                "transform_world_target",
                as_transform(self.transform_world_target),
            )


@dataclass(frozen=True)
class GeometricFeature:
    """Metric feature instantiated in the robot/world frame.

    `transform_world_gauge` is a complete deterministic frame used by later
    computation. It does not imply that all three axes are semantically
    constrained. For point-normal, axis, and plane features, only +z is
    semantic unless a later phase specification says otherwise.
    """

    representation: TargetRepresentation
    point_world: Optional[FloatArray]
    normal_world: Optional[FloatArray] = None
    axis_world: Optional[FloatArray] = None
    transform_world_gauge: Optional[FloatArray] = None
    transform_grasp_feature: Optional[FloatArray] = None
    description: Optional[str] = None
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("point_world", "normal_world", "axis_world"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values.")
            object.__setattr__(self, name, array.copy())

        if self.transform_world_gauge is not None:
            object.__setattr__(
                self,
                "transform_world_gauge",
                as_transform(self.transform_world_gauge),
            )
        if self.transform_grasp_feature is not None:
            object.__setattr__(
                self,
                "transform_grasp_feature",
                as_transform(self.transform_grasp_feature),
            )


@dataclass(frozen=True)
class SceneGeometryDiagnostics:
    target_depth_m: Optional[float]
    target_surface_point_count: int
    target_fit_rms_m: Optional[float]
    target_fit_inlier_count: int


@dataclass(frozen=True)
class SceneGeometry:
    effect: GeometricFeature
    target: GeometricFeature
    diagnostics: SceneGeometryDiagnostics


@dataclass(frozen=True)
class PlaneFit:
    point_world: FloatArray
    normal_world: FloatArray
    inlier_mask: NDArray[np.bool_]
    residuals_m: FloatArray
    rms_m: float


def _normalize(vector: ArrayLike, *, name: str = "vector") -> FloatArray:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    norm = float(np.linalg.norm(array))
    if norm < 1e-12:
        raise ValueError(f"{name} has near-zero norm.")
    return array / norm


def _pixel(value: Any, *, name: str) -> Optional[Pixel]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape != (2,):
        raise ValueError(f"{name} must contain [u, v], got shape {array.shape}.")
    return int(array[0]), int(array[1])


def _pixels(value: Any, *, name: str) -> Tuple[Pixel, ...]:
    if value is None:
        return ()
    array = np.asarray(value)
    if array.size == 0:
        return ()
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {array.shape}.")
    return tuple((int(row[0]), int(row[1])) for row in array)


def _bbox(value: Any, *, name: str) -> Optional[Tuple[int, int, int, int]]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape != (4,):
        raise ValueError(f"{name} must contain [x1,y1,x2,y2], got {array.shape}.")
    return tuple(int(item) for item in array)


def parse_target_grounding(payload: Mapping[str, Any]) -> TargetGrounding:
    """Parse target geometry from the general grounding JSON.

    Supported layouts:
      1. payload["target_geometry"] from scene_grounding_prompt_v2_general.
      2. The provisional screw/plate example JSON used in this project.
      3. A direct target-geometry mapping.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("Grounding payload must be a mapping.")

    direct_transform = payload.get("T_world_target")

    if "target_geometry" in payload:
        geometry = payload["target_geometry"]
        target_entity = payload.get("target_entity", {})
        description = (
            target_entity.get("description")
            if isinstance(target_entity, Mapping)
            else None
        )
    elif "vlm_grounding_provisional" in payload:
        provisional = payload["vlm_grounding_provisional"]
        target = provisional.get("target", {})
        target_section = payload.get("target", {})
        geometry = {
            "representation": target_section.get("geometry_level", "point"),
            "target_point_pixel": target.get("seed_pixel_uv"),
        }
        description = target.get("description")
        direct_transform = target_section.get("T_W_T_gauge", direct_transform)
    else:
        geometry = payload
        description = payload.get("description")

    if not isinstance(geometry, Mapping):
        raise TypeError("target_geometry must be a mapping.")

    representation_raw = geometry.get("representation")
    if representation_raw is None:
        raise ValueError("Target grounding is missing 'representation'.")

    try:
        representation = TargetRepresentation(str(representation_raw))
    except ValueError as error:
        raise ValueError(
            f"Unsupported target representation: {representation_raw!r}."
        ) from error

    transform_value = geometry.get(
        "T_world_target",
        geometry.get("transform_world_target", direct_transform),
    )

    return TargetGrounding(
        representation=representation,
        target_point_pixel=_pixel(
            geometry.get("target_point_pixel"),
            name="target_point_pixel",
        ),
        surface_pixels=_pixels(
            geometry.get("surface_pixels"),
            name="surface_pixels",
        ),
        axis_origin_pixel=_pixel(
            geometry.get("axis_origin_pixel"),
            name="axis_origin_pixel",
        ),
        axis_pixels=_pixels(
            geometry.get("axis_pixels"),
            name="axis_pixels",
        ),
        axis_positive_endpoint_pixel=_pixel(
            geometry.get("axis_positive_endpoint_pixel"),
            name="axis_positive_endpoint_pixel",
        ),
        plane_reference_pixel=_pixel(
            geometry.get("plane_reference_pixel"),
            name="plane_reference_pixel",
        ),
        target_object_bbox=_bbox(
            geometry.get("target_object_bbox"),
            name="target_object_bbox",
        ),
        target_object_positive_pixels=_pixels(
            geometry.get("target_object_positive_pixels"),
            name="target_object_positive_pixels",
        ),
        transform_world_target=(
            None
            if transform_value is None
            else np.asarray(transform_value, dtype=np.float64)
        ),
        description=description,
    )


def _validate_depth_image(depth: ArrayLike) -> FloatArray:
    array = np.asarray(depth, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {array.shape}.")
    return array


def _validate_pixel(pixel: Pixel, image_shape: Tuple[int, int]) -> None:
    u, v = pixel
    height, width = image_shape
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError(
            f"Pixel {(u, v)} lies outside image width={width}, height={height}."
        )


def _valid_depth_mask(depth_m: FloatArray, camera: CameraModel) -> NDArray[np.bool_]:
    return (
        np.isfinite(depth_m)
        & (depth_m >= camera.minimum_depth_m)
        & (depth_m <= camera.maximum_depth_m)
    )


def sample_depth_median(
    depth: ArrayLike,
    pixel: Pixel,
    camera: CameraModel,
    *,
    patch_radius: int = 2,
    fallback_to_nearest: bool = True,
    nearest_search_radius: Optional[int] = None,
) -> float:
    """Return robust depth in metres around one pixel.

    The requested patch is used first. If that patch contains no valid depth and
    ``fallback_to_nearest`` is true, the closest valid depth pixel in image
    space is found. The returned value is the median of valid depth samples in
    a patch of the same radius around that nearest pixel.

    ``nearest_search_radius=None`` searches the complete depth image. Set it to
    a non-negative pixel radius to reject fallbacks that are too far away.
    """
    depth_array = _validate_depth_image(depth)
    _validate_pixel(pixel, depth_array.shape)
    if patch_radius < 0:
        raise ValueError("patch_radius must be non-negative.")
    if nearest_search_radius is not None and nearest_search_radius < 0:
        raise ValueError("nearest_search_radius must be non-negative or None.")

    u, v = pixel
    height, width = depth_array.shape
    depth_m = depth_array * camera.depth_scale_to_m
    valid_mask = _valid_depth_mask(depth_m, camera)

    y1 = max(0, v - patch_radius)
    y2 = min(height, v + patch_radius + 1)
    x1 = max(0, u - patch_radius)
    x2 = min(width, u + patch_radius + 1)
    initial_values = depth_m[y1:y2, x1:x2]
    initial_valid = valid_mask[y1:y2, x1:x2]
    if np.any(initial_valid):
        return float(np.median(initial_values[initial_valid]))

    if not fallback_to_nearest:
        raise ValueError(f"No valid depth around pixel {(u, v)}.")

    if nearest_search_radius is None:
        search_x1, search_y1 = 0, 0
        search_x2, search_y2 = width, height
    else:
        search_x1 = max(0, u - nearest_search_radius)
        search_x2 = min(width, u + nearest_search_radius + 1)
        search_y1 = max(0, v - nearest_search_radius)
        search_y2 = min(height, v + nearest_search_radius + 1)

    search_valid = valid_mask[search_y1:search_y2, search_x1:search_x2]
    valid_v_local, valid_u_local = np.nonzero(search_valid)
    if valid_u_local.size == 0:
        radius_text = (
            "the complete image"
            if nearest_search_radius is None
            else f"a radius of {nearest_search_radius} pixels"
        )
        raise ValueError(
            f"No valid depth around pixel {(u, v)} or within {radius_text}."
        )

    valid_u = valid_u_local + search_x1
    valid_v = valid_v_local + search_y1
    distance_squared = (valid_u - u) ** 2 + (valid_v - v) ** 2
    nearest_index = int(np.argmin(distance_squared))
    nearest_u = int(valid_u[nearest_index])
    nearest_v = int(valid_v[nearest_index])

    nearest_y1 = max(0, nearest_v - patch_radius)
    nearest_y2 = min(height, nearest_v + patch_radius + 1)
    nearest_x1 = max(0, nearest_u - patch_radius)
    nearest_x2 = min(width, nearest_u + patch_radius + 1)
    nearest_values = depth_m[nearest_y1:nearest_y2, nearest_x1:nearest_x2]
    nearest_valid = valid_mask[nearest_y1:nearest_y2, nearest_x1:nearest_x2]

    # The nearest center pixel itself is valid, so this is guaranteed non-empty.
    return float(np.median(nearest_values[nearest_valid]))


def backproject_pixel_to_camera(
    pixel: Pixel,
    depth_m: float,
    intrinsic_matrix: ArrayLike,
) -> FloatArray:
    """Back-project [u, v] and metric depth into camera coordinates."""
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic_matrix must have shape (3, 3).")
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError("depth_m must be finite and positive.")

    u, v = pixel
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    return np.array(
        [
            (u - cx) * depth_m / fx,
            (v - cy) * depth_m / fy,
            depth_m,
        ],
        dtype=np.float64,
    )


def backproject_pixel_to_world(
    depth: ArrayLike,
    pixel: Pixel,
    camera: CameraModel,
    *,
    patch_radius: int = 2,
    fallback_to_nearest: bool = True,
    nearest_search_radius: Optional[int] = None,
) -> Tuple[FloatArray, float]:
    depth_m = sample_depth_median(
        depth,
        pixel,
        camera,
        patch_radius=patch_radius,
        fallback_to_nearest=fallback_to_nearest,
        nearest_search_radius=nearest_search_radius,
    )
    point_camera = backproject_pixel_to_camera(
        pixel,
        depth_m,
        camera.intrinsic_matrix,
    )
    point_world = transform_point(
        camera.transform_world_camera,
        point_camera,
    )
    return point_world, depth_m


def collect_patch_pixels(
    center: Pixel,
    image_shape: Tuple[int, int],
    *,
    radius: int,
    stride: int = 1,
) -> Tuple[Pixel, ...]:
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    u_center, v_center = center
    height, width = image_shape
    _validate_pixel(center, image_shape)

    result = []
    for v in range(
        max(0, v_center - radius),
        min(height, v_center + radius + 1),
        stride,
    ):
        for u in range(
            max(0, u_center - radius),
            min(width, u_center + radius + 1),
            stride,
        ):
            result.append((u, v))
    return tuple(result)


def backproject_pixels_to_world(
    depth: ArrayLike,
    pixels: Sequence[Pixel],
    camera: CameraModel,
    *,
    patch_radius: int = 0,
    ignore_invalid: bool = True,
    fallback_to_nearest: bool = False,
    nearest_search_radius: Optional[int] = None,
) -> Tuple[FloatArray, Tuple[Pixel, ...]]:
    """Back-project many pixels, optionally discarding invalid samples."""
    points = []
    used_pixels = []
    for pixel in pixels:
        try:
            point, _ = backproject_pixel_to_world(
                depth,
                pixel,
                camera,
                patch_radius=patch_radius,
                fallback_to_nearest=fallback_to_nearest,
                nearest_search_radius=nearest_search_radius,
            )
        except ValueError:
            if ignore_invalid:
                continue
            raise
        points.append(point)
        used_pixels.append(pixel)

    if not points:
        raise ValueError("No valid 3D points could be back-projected.")
    return np.stack(points, axis=0), tuple(used_pixels)


def orient_direction(
    direction: ArrayLike,
    preferred_positive_direction: Optional[ArrayLike],
) -> FloatArray:
    """Resolve a direction sign using a semantic or deterministic reference."""
    result = _normalize(direction, name="direction")
    if preferred_positive_direction is not None:
        reference = _normalize(
            preferred_positive_direction,
            name="preferred_positive_direction",
        )
        if float(np.dot(result, reference)) < 0.0:
            result = -result
        return result

    # Deterministic fallback: the largest-magnitude world component is positive.
    dominant = int(np.argmax(np.abs(result)))
    if result[dominant] < 0.0:
        result = -result
    return result


def fit_plane_robust(
    points_world: ArrayLike,
    *,
    preferred_normal_world: Optional[ArrayLike] = None,
    maximum_iterations: int = 5,
    minimum_inlier_threshold_m: float = 0.002,
    mad_scale: float = 3.5,
) -> PlaneFit:
    """Fit a plane with SVD and iterative median-residual trimming."""
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points_world must have shape (N, 3), got {points.shape}."
        )
    if points.shape[0] < 3:
        raise ValueError("At least three points are required for plane fitting.")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_world contains non-finite values.")

    inliers = np.ones(points.shape[0], dtype=bool)

    for _ in range(maximum_iterations):
        selected = points[inliers]
        if selected.shape[0] < 3:
            raise ValueError("Plane fitting retained fewer than three inliers.")
        centroid = selected.mean(axis=0)
        centered = selected - centroid
        covariance = centered.T @ centered / selected.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, int(np.argmin(eigenvalues))]
        residuals = np.abs((points - centroid) @ normal)

        median = float(np.median(residuals[inliers]))
        mad = float(np.median(np.abs(residuals[inliers] - median)))
        robust_sigma = 1.4826 * mad
        threshold = max(
            minimum_inlier_threshold_m,
            median + mad_scale * robust_sigma,
        )
        new_inliers = residuals <= threshold
        if new_inliers.sum() < 3 or np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers

    selected = points[inliers]
    centroid = selected.mean(axis=0)
    centered = selected - centroid
    covariance = centered.T @ centered / selected.shape[0]
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = orient_direction(
        eigenvectors[:, 0],
        preferred_normal_world,
    )
    residuals = np.abs((points - centroid) @ normal)
    rms = float(np.sqrt(np.mean(np.square(residuals[inliers]))))

    return PlaneFit(
        point_world=centroid,
        normal_world=normal,
        inlier_mask=inliers,
        residuals_m=residuals,
        rms_m=rms,
    )


def frame_from_z_axis(
    origin_world: ArrayLike,
    z_axis_world: ArrayLike,
    *,
    x_hint_world: Optional[ArrayLike] = None,
) -> FloatArray:
    """Build a deterministic right-handed gauge frame around a semantic +z."""
    origin = np.asarray(origin_world, dtype=np.float64)
    if origin.shape != (3,):
        raise ValueError("origin_world must have shape (3,).")
    z_axis = _normalize(z_axis_world, name="z_axis_world")

    candidates = []
    if x_hint_world is not None:
        candidates.append(np.asarray(x_hint_world, dtype=np.float64))
    candidates.extend(
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
    )

    x_axis = None
    for candidate in candidates:
        if candidate.shape != (3,) or not np.all(np.isfinite(candidate)):
            continue
        projected = candidate - z_axis * float(np.dot(candidate, z_axis))
        if np.linalg.norm(projected) > 1e-8:
            x_axis = _normalize(projected, name="projected x hint")
            break
    if x_axis is None:
        raise ValueError("Could not construct a gauge x-axis.")

    y_axis = _normalize(np.cross(z_axis, x_axis), name="gauge y-axis")
    x_axis = _normalize(np.cross(y_axis, z_axis), name="gauge x-axis")
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return make_transform(rotation, origin)


def plane_x_hint_from_grasp(
    transform_world_grasp: ArrayLike,
    plane_normal_world: ArrayLike,
) -> FloatArray:
    """Return an in-plane gauge direction tied to the initial grasp pose.

    A plane target semantically defines only its origin and +z normal.  Its x/y
    axes are computational gauges used for target-relative demonstration
    transfer.  Anchor that gauge to the initial grasp orientation so the
    demonstrated in-plane motion is transferred consistently without assigning
    semantic meaning to the plane's x/y axes.
    """
    transform = as_transform(transform_world_grasp)
    normal = _normalize(plane_normal_world, name="plane_normal_world")

    # Prefer the same grasp-local axis in both demo/current scenes.  Fall back
    # only when that axis is nearly parallel to the plane normal.
    for axis_index in (0, 1, 2):
        candidate = transform[:3, axis_index]
        projected = candidate - normal * float(np.dot(candidate, normal))
        if np.linalg.norm(projected) > 1e-6:
            return _normalize(projected, name="plane x hint from grasp")

    raise ValueError("Could not construct a plane x gauge from the grasp pose.")


def build_object_origin_effect(
    transform_world_object_current: ArrayLike,
    transform_world_grasp_current: ArrayLike,
    *,
    description: str = "controlled object-frame origin",
) -> GeometricFeature:
    """Use the current object-frame origin as the minimal effect feature."""
    transform_world_object = as_transform(transform_world_object_current)
    transform_world_grasp = as_transform(transform_world_grasp_current)
    transform_grasp_effect = relative_transform(
        transform_world_grasp,
        transform_world_object,
    )
    return GeometricFeature(
        representation=TargetRepresentation.POINT,
        point_world=transform_world_object[:3, 3],
        transform_world_gauge=transform_world_object,
        transform_grasp_feature=transform_grasp_effect,
        description=description,
        source="current object pose",
        metadata={"orientation_is_gauge": True},
    )


def build_object_point_effect(
    transform_world_object_current: ArrayLike,
    transform_world_grasp_current: ArrayLike,
    point_world: ArrayLike,
    *,
    description: str = "task-relevant controlled point",
) -> GeometricFeature:
    """Build an object-attached effect point with the object's gauge orientation."""
    transform_world_object = as_transform(transform_world_object_current)
    transform_world_grasp = as_transform(transform_world_grasp_current)
    point = np.asarray(point_world, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point_world must be a finite vector with shape (3,).")

    transform_world_effect = transform_world_object.copy()
    transform_world_effect[:3, 3] = point
    transform_grasp_effect = relative_transform(
        transform_world_grasp,
        transform_world_effect,
    )
    return GeometricFeature(
        representation=TargetRepresentation.POINT,
        point_world=point,
        transform_world_gauge=transform_world_effect,
        transform_grasp_feature=transform_grasp_effect,
        description=description,
        source="controlled-point grounding",
        metadata={"orientation_is_gauge": True},
    )


def _target_point(
    depth: ArrayLike,
    grounding: TargetGrounding,
    camera: CameraModel,
    *,
    point_patch_radius: int,
) -> Tuple[FloatArray, float]:
    if grounding.target_point_pixel is None:
        raise ValueError("Point target requires target_point_pixel.")
    return backproject_pixel_to_world(
        depth,
        grounding.target_point_pixel,
        camera,
        patch_radius=point_patch_radius,
    )


def _instantiate_surface_target(
    depth_array: FloatArray,
    grounding: TargetGrounding,
    camera: CameraModel,
    *,
    effect_point_world: Optional[ArrayLike],
    transform_world_grasp_reference: Optional[ArrayLike],
    point_patch_radius: int,
    plane_patch_radius: int,
    plane_patch_stride: int,
    preferred_normal_world: Optional[ArrayLike],
) -> Tuple[GeometricFeature, SceneGeometryDiagnostics]:
    """Fit a local surface and construct a point-normal or plane target."""
    representation = grounding.representation
    reference_pixel = (
        grounding.target_point_pixel
        if representation is TargetRepresentation.POINT_NORMAL
        else grounding.plane_reference_pixel
    )
    if reference_pixel is None:
        reference_pixel = grounding.target_point_pixel
    if reference_pixel is None:
        raise ValueError(
            f"{representation.value} requires a target/reference pixel."
        )

    reference_point_world, depth_m = backproject_pixel_to_world(
        depth_array,
        reference_pixel,
        camera,
        patch_radius=point_patch_radius,
    )

    surface_pixels = grounding.surface_pixels
    if len(surface_pixels) < 3:
        surface_pixels = collect_patch_pixels(
            reference_pixel,
            depth_array.shape,
            radius=plane_patch_radius,
            stride=plane_patch_stride,
        )

    surface_points_world, _ = backproject_pixels_to_world(
        depth_array,
        surface_pixels,
        camera,
        patch_radius=0,
        ignore_invalid=True,
    )
    fit = fit_plane_robust(
        surface_points_world,
        preferred_normal_world=preferred_normal_world,
    )

    if representation is TargetRepresentation.POINT_NORMAL:
        origin_world = reference_point_world
    else:
        # A grounded plane reference point should remain on the fitted plane.
        displacement = reference_point_world - fit.point_world
        origin_world = (
            reference_point_world
            - fit.normal_world * float(np.dot(displacement, fit.normal_world))
        )

    x_hint = None
    gauge_source = "deterministic_fallback"
    if (
        representation is TargetRepresentation.PLANE
        and transform_world_grasp_reference is not None
    ):
        # Plane x/y are non-semantic.  Use the initial grasp orientation as
        # their gauge so target-relative transfer is anchored jointly by
        # the important surface point (origin), plane normal (+z), and the
        # initial grasp orientation (in-plane gauge).
        x_hint = plane_x_hint_from_grasp(
            transform_world_grasp_reference,
            fit.normal_world,
        )
        gauge_source = "initial_grasp_orientation"
    elif effect_point_world is not None:
        # Preserve the previous point-normal behavior.
        x_hint = np.asarray(effect_point_world, dtype=np.float64) - origin_world
        gauge_source = "effect_point_direction"

    transform_world_gauge = frame_from_z_axis(
        origin_world,
        fit.normal_world,
        x_hint_world=x_hint,
    )
    feature = GeometricFeature(
        representation=representation,
        point_world=origin_world,
        normal_world=fit.normal_world,
        transform_world_gauge=transform_world_gauge,
        description=grounding.description,
        source="local RGB-D plane fit",
        metadata={
            "semantic_axes": ["z"],
            "rotation_about_z": "gauge_only",
            "xy_gauge_source": gauge_source,
        },
    )
    diagnostics = SceneGeometryDiagnostics(
        target_depth_m=depth_m,
        target_surface_point_count=int(surface_points_world.shape[0]),
        target_fit_rms_m=fit.rms_m,
        target_fit_inlier_count=int(fit.inlier_mask.sum()),
    )
    return feature, diagnostics


def _instantiate_axis_target(
    depth_array: FloatArray,
    grounding: TargetGrounding,
    camera: CameraModel,
    *,
    effect_point_world: Optional[ArrayLike],
    point_patch_radius: int,
    preferred_axis_world: Optional[ArrayLike],
) -> Tuple[GeometricFeature, SceneGeometryDiagnostics]:
    """Fit an axis and resolve its origin and positive direction."""
    representation = grounding.representation
    axis_pixels = grounding.axis_pixels
    if len(axis_pixels) < 2:
        raise ValueError("Axis target requires at least two axis_pixels.")
    axis_points_world, _ = backproject_pixels_to_world(
        depth_array,
        axis_pixels,
        camera,
        patch_radius=point_patch_radius,
        ignore_invalid=False,
        fallback_to_nearest=True,
    )
    centroid = axis_points_world.mean(axis=0)
    centered = axis_points_world - centroid
    covariance = centered.T @ centered / axis_points_world.shape[0]
    _, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, 2]

    if grounding.axis_origin_pixel is not None:
        origin_world, depth_m = backproject_pixel_to_world(
            depth_array,
            grounding.axis_origin_pixel,
            camera,
            patch_radius=point_patch_radius,
        )
    else:
        origin_world = centroid
        point_camera = transform_point(
            invert_transform(camera.transform_world_camera),
            origin_world,
        )
        depth_m = float(point_camera[2])

    direction_reference = preferred_axis_world
    if grounding.axis_positive_endpoint_pixel is not None:
        positive_endpoint_world, _ = backproject_pixel_to_world(
            depth_array,
            grounding.axis_positive_endpoint_pixel,
            camera,
            patch_radius=point_patch_radius,
        )
        direction_reference = positive_endpoint_world - origin_world

    axis = orient_direction(axis, direction_reference)
    x_hint = None
    if effect_point_world is not None:
        x_hint = np.asarray(effect_point_world, dtype=np.float64) - origin_world
    transform_world_gauge = frame_from_z_axis(
        origin_world,
        axis,
        x_hint_world=x_hint,
    )
    residual_vectors = np.cross(
        axis_points_world - origin_world,
        axis,
    )
    residuals = np.linalg.norm(residual_vectors, axis=1)
    feature = GeometricFeature(
        representation=representation,
        point_world=origin_world,
        axis_world=axis,
        transform_world_gauge=transform_world_gauge,
        description=grounding.description,
        source="RGB-D axis fit",
        metadata={
            "semantic_axes": ["z"],
            "rotation_about_z": "gauge_only",
        },
    )
    diagnostics = SceneGeometryDiagnostics(
        target_depth_m=depth_m,
        target_surface_point_count=int(axis_points_world.shape[0]),
        target_fit_rms_m=float(np.sqrt(np.mean(np.square(residuals)))),
        target_fit_inlier_count=int(axis_points_world.shape[0]),
    )
    return feature, diagnostics


def instantiate_target_geometry(
    depth: ArrayLike,
    grounding: TargetGrounding,
    camera: CameraModel,
    *,
    effect_point_world: Optional[ArrayLike] = None,
    transform_world_grasp_reference: Optional[ArrayLike] = None,
    point_patch_radius: int = 2,
    plane_patch_radius: int = 8,
    plane_patch_stride: int = 2,
    preferred_normal_world: Optional[ArrayLike] = (0.0, 0.0, 1.0),
    preferred_axis_world: Optional[ArrayLike] = None,
    target_object_points_world: Optional[ArrayLike] = None,
) -> Tuple[GeometricFeature, SceneGeometryDiagnostics]:
    """Instantiate target metric geometry from aligned RGB-D evidence."""
    depth_array = _validate_depth_image(depth)
    representation = grounding.representation

    if representation is TargetRepresentation.POINT:
        point_world, depth_m = _target_point(
            depth_array,
            grounding,
            camera,
            point_patch_radius=point_patch_radius,
        )
        feature = GeometricFeature(
            representation=representation,
            point_world=point_world,
            description=grounding.description,
            source="target pixel + aligned depth",
        )
        diagnostics = SceneGeometryDiagnostics(
            target_depth_m=depth_m,
            target_surface_point_count=1,
            target_fit_rms_m=None,
            target_fit_inlier_count=1,
        )
        return feature, diagnostics

    if representation in (
        TargetRepresentation.POINT_NORMAL,
        TargetRepresentation.PLANE,
    ):
        return _instantiate_surface_target(
            depth_array,
            grounding,
            camera,
            effect_point_world=effect_point_world,
            transform_world_grasp_reference=transform_world_grasp_reference,
            point_patch_radius=point_patch_radius,
            plane_patch_radius=plane_patch_radius,
            plane_patch_stride=plane_patch_stride,
            preferred_normal_world=preferred_normal_world,
        )

    if representation is TargetRepresentation.AXIS:
        return _instantiate_axis_target(
            depth_array,
            grounding,
            camera,
            effect_point_world=effect_point_world,
            point_patch_radius=point_patch_radius,
            preferred_axis_world=preferred_axis_world,
        )

    if representation is TargetRepresentation.OBJECT_POSE_6D:
        if grounding.transform_world_target is not None:
            transform = as_transform(grounding.transform_world_target)
            feature = GeometricFeature(
                representation=representation,
                point_world=transform[:3, 3],
                transform_world_gauge=transform,
                description=grounding.description,
                source="provided 6D target transform",
                metadata={"semantic_axes": ["x", "y", "z"]},
            )
            diagnostics = SceneGeometryDiagnostics(
                target_depth_m=None,
                target_surface_point_count=0,
                target_fit_rms_m=None,
                target_fit_inlier_count=0,
            )
            return feature, diagnostics

        if target_object_points_world is None:
            raise ValueError(
                "object_pose_6d requires either transform_world_target or "
                "a segmented target_object_points_world point cloud. A bbox "
                "or sparse positive pixels alone is not sufficient for a "
                "reliable full 6D frame."
            )
        transform, eigenvalues = compute_pca_object_pose(
            target_object_points_world,
            world_up=(0.0, 0.0, 1.0),
            camera_position_world=camera.transform_world_camera[:3, 3],
        )
        feature = GeometricFeature(
            representation=representation,
            point_world=transform[:3, 3],
            transform_world_gauge=transform,
            description=grounding.description,
            source="segmented target point-cloud PCA up/camera frame",
            metadata={
                "semantic_axes": ["x", "y", "z"],
                "pca_eigenvalues_ascending": eigenvalues.tolist(),
            },
        )
        diagnostics = SceneGeometryDiagnostics(
            target_depth_m=None,
            target_surface_point_count=int(
                np.asarray(target_object_points_world).shape[0]
            ),
            target_fit_rms_m=None,
            target_fit_inlier_count=int(
                np.asarray(target_object_points_world).shape[0]
            ),
        )
        return feature, diagnostics

    raise ValueError(f"Unsupported representation: {representation}.")


def build_scene_geometry(
    depth: ArrayLike,
    camera: CameraModel,
    grounding: TargetGrounding,
    transform_world_object_current: ArrayLike,
    transform_world_grasp_current: ArrayLike,
    *,
    point_patch_radius: int = 2,
    plane_patch_radius: int = 8,
    plane_patch_stride: int = 2,
    preferred_normal_world: Optional[ArrayLike] = (0.0, 0.0, 1.0),
    preferred_axis_world: Optional[ArrayLike] = None,
    target_object_points_world: Optional[ArrayLike] = None,
    effect_point_world: Optional[ArrayLike] = None,
    effect_description: Optional[str] = None,
) -> SceneGeometry:
    """Build the minimal current-scene geometry for later reference transfer.

    Effect-frame convention:
        effect = controlled object's canonical origin.

    The effect is stored relative to the canonical grasp frame so later robot
    FK can evaluate it as:

        T_world_effect(q) = T_world_grasp(q) @ T_grasp_effect
    """
    effect = (
        build_object_origin_effect(
            transform_world_object_current,
            transform_world_grasp_current,
        )
        if effect_point_world is None
        else build_object_point_effect(
            transform_world_object_current,
            transform_world_grasp_current,
            effect_point_world,
            description=(effect_description or "task-relevant controlled point"),
        )
    )
    target, diagnostics = instantiate_target_geometry(
        depth,
        grounding,
        camera,
        effect_point_world=effect.point_world,
        transform_world_grasp_reference=transform_world_grasp_current,
        point_patch_radius=point_patch_radius,
        plane_patch_radius=plane_patch_radius,
        plane_patch_stride=plane_patch_stride,
        preferred_normal_world=preferred_normal_world,
        preferred_axis_world=preferred_axis_world,
        target_object_points_world=target_object_points_world,
    )
    return SceneGeometry(
        effect=effect,
        target=target,
        diagnostics=diagnostics,
    )


def _optional_array(value: Optional[FloatArray], shape: Tuple[int, ...]) -> FloatArray:
    if value is None:
        return np.empty((0,), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"Expected optional array shape {shape}, got {array.shape}.")
    return array


def save_scene_geometry(path: str | Path, scene: SceneGeometry) -> None:
    """Save scene geometry as a compact NPZ bundle for reference construction."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        effect_representation=np.asarray(scene.effect.representation.value),
        effect_point_world=_optional_array(scene.effect.point_world, (3,)),
        effect_normal_world=_optional_array(scene.effect.normal_world, (3,)),
        effect_axis_world=_optional_array(scene.effect.axis_world, (3,)),
        effect_transform_world_gauge=_optional_array(
            scene.effect.transform_world_gauge, (4, 4)
        ),
        effect_transform_grasp_feature=_optional_array(
            scene.effect.transform_grasp_feature, (4, 4)
        ),
        effect_description=np.asarray(scene.effect.description or ""),
        effect_source=np.asarray(scene.effect.source),
        effect_metadata_json=np.asarray(json.dumps(dict(scene.effect.metadata))),
        target_representation=np.asarray(scene.target.representation.value),
        target_point_world=_optional_array(scene.target.point_world, (3,)),
        target_normal_world=_optional_array(scene.target.normal_world, (3,)),
        target_axis_world=_optional_array(scene.target.axis_world, (3,)),
        target_transform_world_gauge=_optional_array(
            scene.target.transform_world_gauge, (4, 4)
        ),
        target_transform_grasp_feature=_optional_array(
            scene.target.transform_grasp_feature, (4, 4)
        ),
        target_description=np.asarray(scene.target.description or ""),
        target_source=np.asarray(scene.target.source),
        target_metadata_json=np.asarray(json.dumps(dict(scene.target.metadata))),
        diagnostic_target_depth_m=np.asarray(
            np.nan
            if scene.diagnostics.target_depth_m is None
            else scene.diagnostics.target_depth_m,
            dtype=np.float64,
        ),
        diagnostic_target_surface_point_count=np.asarray(
            scene.diagnostics.target_surface_point_count,
            dtype=np.int64,
        ),
        diagnostic_target_fit_rms_m=np.asarray(
            np.nan
            if scene.diagnostics.target_fit_rms_m is None
            else scene.diagnostics.target_fit_rms_m,
            dtype=np.float64,
        ),
        diagnostic_target_fit_inlier_count=np.asarray(
            scene.diagnostics.target_fit_inlier_count,
            dtype=np.int64,
        ),
    )
