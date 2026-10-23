#!/usr/bin/env python3
"""Simplified VLM-guided Franka grasp-pose reconstruction.

The VLM/API is NOT called by this script.

Input:
    --rgb    camera_rgb.png
    --depth  initial_depth.npy      aligned to RGB, metres
    --json        grasp_spec.json                 manually saved VLM output
    --camera-json camera_calibration_info.json       runtime camera calibration

VLM decisions per grasp:
    grasp_mode      = "top" | "surface_normal"
    anchor_pixel
    open_axis_mode  = "surface_normal" | "explicit_axis"
    axis_pixel_start / axis_pixel_end (required only for explicit_axis)

Deterministic geometry:
    +surface_normal = local visible-surface normal, signed toward camera

    pointing:
        top            -> world -Z
        surface_normal -> -surface_normal

    opening axis:
        surface_normal -> project local surface normal perpendicular to pointing
        explicit_axis  -> lift VLM start/end pixels to 3D and project the
                          resulting axis perpendicular to pointing

    TCP:
        default TCP = anchor
        optional manual tcp_depth_cm moves TCP along -surface_normal

The VLM never chooses pointing sign, TCP translation direction/sign, or metric depth.

Two poses are emitted for every accepted grasp:
    Pose A = selected orientation
    Pose B = same TCP + pointing, opening axis reversed (180 deg about pointing)

Franka TCP convention used here (confirmed for the user's setup):
    panda_hand_tcp +Z = physical gripper pointing direction
    panda_hand_tcp +Y = finger opening/closing direction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# =============================================================================
# CAMERA CALIBRATION
# =============================================================================

def load_camera_calibration(
    camera_json: str | Path,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load K and T_world_camera from the runtime calibration JSON.

    Expected structure:
        {
          "external_sensor0": {
            "intrinsics": [[...], [...], [...]],
            "extrinsics": [[...], [...], [...], [...]]
          }
        }

    ``extrinsics`` is interpreted as camera optical frame -> panda_link0/world
    frame, i.e. T_world_camera, matching the rest of this pipeline.
    """
    path = Path(camera_json).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            f"Camera calibration JSON must contain at least one camera: {path}"
        )

    if camera_name not in payload:
        raise KeyError(
            f"Camera {camera_name!r} not found in {path}. "
            f"Available cameras: {list(payload.keys())}"
        )

    camera = payload[camera_name]
    if not isinstance(camera, dict):
        raise ValueError(
            f"Calibration entry for {camera_name!r} must be a JSON object."
        )

    K = np.asarray(camera["intrinsics"], dtype=np.float64)
    T_world_camera = np.asarray(camera["extrinsics"], dtype=np.float64)

    if K.shape != (3, 3):
        raise ValueError(
            f"Expected {camera_name}.intrinsics shape (3,3), got {K.shape}"
        )
    if T_world_camera.shape != (4, 4):
        raise ValueError(
            f"Expected {camera_name}.extrinsics shape (4,4), "
            f"got {T_world_camera.shape}"
        )
    if not np.all(np.isfinite(K)):
        raise ValueError("Camera intrinsics contain non-finite values.")
    if not np.all(np.isfinite(T_world_camera)):
        raise ValueError("Camera extrinsics contain non-finite values.")
    if not np.allclose(T_world_camera[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(
            "Camera extrinsic last row must be [0, 0, 0, 1] for T_world_camera."
        )

    return K, T_world_camera


WORLD_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)


# =============================================================================
# PANDA TCP AXES
# =============================================================================

TCP_POINTING_LOCAL_AXIS = "+z"
TCP_OPEN_LOCAL_AXIS = "+y"

GRASP_MODES = ("top", "surface_normal")
OPEN_AXIS_MODES = ("surface_normal", "explicit_axis")
DEFAULT_TCP_DEPTH_CM = 0
MAX_TCP_DEPTH_CM = 4.0


# =============================================================================
# VALIDATION / GEOMETRY DEFAULTS
# =============================================================================

DEFAULT_NORMAL_RADIUS_PX = 10
DEFAULT_NORMAL_DEPTH_GATE_M = 0.02
DEFAULT_NORMAL_PLANE_RESIDUAL_M = 0.004

DEFAULT_MIN_AXIS_PIXEL_DISTANCE_PX = 8.0
DEFAULT_MAX_ANCHOR_TO_AXIS_LINE_PX = 50.0
DEFAULT_MAX_AXIS_POINT_PLANE_DISTANCE_M = 0.03
DEFAULT_MIN_AXIS_LINE_SAME_SURFACE_FRACTION = 0.60

# After projecting an opening candidate perpendicular to pointing, reject if
# very little of the original direction remains. 0.25 means at least 25%.
DEFAULT_MIN_OPEN_PROJECTION_NORM = 0.25


# =============================================================================
# BASIC MATH
# =============================================================================


def normalize(vector: np.ndarray, name: str = "vector", eps: float = 1e-9) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < eps:
        raise ValueError(f"Cannot normalize {name}; norm={norm}")
    return vector / norm


def signed_local_axis(name: str) -> np.ndarray:
    table = {
        "+x": [1.0, 0.0, 0.0],
        "-x": [-1.0, 0.0, 0.0],
        "+y": [0.0, 1.0, 0.0],
        "-y": [0.0, -1.0, 0.0],
        "+z": [0.0, 0.0, 1.0],
        "-z": [0.0, 0.0, -1.0],
    }
    if name not in table:
        raise ValueError(f"Unsupported local TCP axis: {name!r}")
    return np.asarray(table[name], dtype=np.float64)


def project_perpendicular(vector: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, float]:
    """Project vector onto the plane perpendicular to unit axis.

    Returns:
        projected_unit_direction, retained_fraction

    retained_fraction = ||projection|| / ||original||.
    """
    vector = np.asarray(vector, dtype=np.float64)
    axis = normalize(axis, "projection_axis")

    original_norm = float(np.linalg.norm(vector))
    if original_norm < 1e-9:
        raise ValueError("Cannot project a zero-length vector.")

    projected = vector - float(np.dot(vector, axis)) * axis
    projected_norm = float(np.linalg.norm(projected))
    retained_fraction = projected_norm / original_norm

    if projected_norm < 1e-9:
        return projected, retained_fraction

    return projected / projected_norm, retained_fraction


def build_tcp_rotation(pointing_world: np.ndarray, open_world: np.ndarray) -> np.ndarray:
    """Map desired physical pointing/open directions onto panda_hand_tcp axes."""
    pointing_world = normalize(pointing_world, "pointing_world")

    open_world, retained = project_perpendicular(open_world, pointing_world)
    if retained < 1e-9:
        raise ValueError("Opening direction is parallel to pointing direction.")
    open_world = normalize(open_world, "open_world")

    third_world = normalize(
        np.cross(pointing_world, open_world),
        "third_world",
    )

    pointing_local = normalize(
        signed_local_axis(TCP_POINTING_LOCAL_AXIS),
        "pointing_local",
    )
    open_local = normalize(
        signed_local_axis(TCP_OPEN_LOCAL_AXIS),
        "open_local",
    )

    if abs(float(np.dot(pointing_local, open_local))) > 1e-8:
        raise ValueError("TCP pointing/open local axes must be orthogonal.")

    third_local = normalize(
        np.cross(pointing_local, open_local),
        "third_local",
    )

    world_basis = np.column_stack(
        (pointing_world, open_world, third_world)
    )
    local_basis = np.column_stack(
        (pointing_local, open_local, third_local)
    )

    rotation_world_tcp = world_basis @ local_basis.T

    # Numerical cleanup to the nearest proper rotation.
    u, _, vt = np.linalg.svd(rotation_world_tcp)
    rotation_world_tcp = u @ vt
    if np.linalg.det(rotation_world_tcp) < 0.0:
        u[:, -1] *= -1.0
        rotation_world_tcp = u @ vt

    return rotation_world_tcp


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64)
    return transform[:3, :3] @ point + transform[:3, 3]


def transform_vector(transform: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return transform[:3, :3] @ vector


# =============================================================================
# RGB-D GEOMETRY
# =============================================================================


def validate_pixel(pixel: Any, width: int, height: int, name: str) -> tuple[int, int]:
    if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
        raise ValueError(f"{name} must be [u, v].")

    u = int(round(float(pixel[0])))
    v = int(round(float(pixel[1])))

    if not (0 <= u < width and 0 <= v < height):
        raise ValueError(
            f"{name}={(u, v)} outside image width={width}, height={height}."
        )

    return u, v


def backproject_pixel(u: int, v: int, depth_m: float, K: np.ndarray) -> np.ndarray:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    z = float(depth_m)
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy

    return np.array([x, y, z], dtype=np.float64)


def robust_depth_at_pixel(
    depth: np.ndarray,
    u: int,
    v: int,
    search_radius_px: int = 3,
) -> float:
    """Use exact pixel depth whenever valid; tiny fallback only if invalid."""
    center = float(depth[v, u])
    if np.isfinite(center) and center > 0.0:
        return center

    height, width = depth.shape
    u0 = max(0, u - search_radius_px)
    u1 = min(width, u + search_radius_px + 1)
    v0 = max(0, v - search_radius_px)
    v1 = min(height, v + search_radius_px + 1)

    patch = np.asarray(depth[v0:v1, u0:u1], dtype=np.float64)
    valid = np.isfinite(patch) & (patch > 0.0)

    if not np.any(valid):
        raise ValueError(
            f"No valid depth within {search_radius_px}px of pixel {(u, v)}."
        )

    return float(np.median(patch[valid]))


def collect_local_surface_points(
    depth: np.ndarray,
    anchor_u: int,
    anchor_v: int,
    anchor_depth: float,
    K: np.ndarray,
    radius_px: int,
    depth_gate_m: float,
) -> np.ndarray:
    """Collect nearby points while suppressing neighboring depth surfaces."""
    height, width = depth.shape

    u0 = max(0, anchor_u - radius_px)
    u1 = min(width, anchor_u + radius_px + 1)
    v0 = max(0, anchor_v - radius_px)
    v1 = min(height, anchor_v + radius_px + 1)

    points = []
    for v in range(v0, v1):
        for u in range(u0, u1):
            z = float(depth[v, u])
            if not np.isfinite(z) or z <= 0.0:
                continue
            if abs(z - anchor_depth) > depth_gate_m:
                continue
            points.append(backproject_pixel(u, v, z, K))

    if len(points) < 12:
        raise ValueError(
            f"Only {len(points)} local points survived around anchor "
            f"{(anchor_u, anchor_v)}. Choose a more interior pixel."
        )

    return np.asarray(points, dtype=np.float64)


def fit_surface_normal(
    points_camera: np.ndarray,
    anchor_camera: np.ndarray,
    max_iterations: int = 3,
    residual_threshold_m: float = DEFAULT_NORMAL_PLANE_RESIDUAL_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust local PCA plane fit; positive normal is signed toward camera."""
    points = np.asarray(points_camera, dtype=np.float64)
    if points.shape[0] < 3:
        raise ValueError("Need at least three points for surface-normal fitting.")

    active = points.copy()
    normal = None

    for _ in range(max_iterations):
        centroid = active.mean(axis=0)
        centered = active - centroid
        covariance = centered.T @ centered / max(len(active), 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        normal = normalize(
            eigenvectors[:, int(np.argmin(eigenvalues))],
            "surface_normal_camera",
        )

        residuals = np.abs((points - centroid) @ normal)
        keep = residuals <= residual_threshold_m

        if np.count_nonzero(keep) < 10:
            break

        new_active = points[keep]
        if len(new_active) == len(active):
            active = new_active
            break
        active = new_active

    if normal is None:
        raise RuntimeError("Surface-normal estimation failed.")

    # Camera optical center is [0, 0, 0].
    toward_camera = normalize(-anchor_camera, "anchor_to_camera")
    if float(np.dot(normal, toward_camera)) < 0.0:
        normal = -normal

    return normal, active


def point_to_segment_distance_2d(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)

    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))

    t = float(np.dot(point - start, segment) / denom)
    t = min(1.0, max(0.0, t))
    closest = start + t * segment
    return float(np.linalg.norm(point - closest))


def axis_segment_same_surface_fraction(
    depth: np.ndarray,
    K: np.ndarray,
    start_uv: np.ndarray,
    end_uv: np.ndarray,
    anchor_camera: np.ndarray,
    surface_normal_camera: np.ndarray,
    max_plane_distance_m: float,
    sample_count: int = 21,
) -> tuple[float, int, int]:
    """Approximate continuity check for an explicit image-space axis."""
    height, width = depth.shape
    same_surface = 0
    valid_depth = 0

    for t in np.linspace(0.0, 1.0, sample_count):
        uv = (1.0 - t) * start_uv + t * end_uv
        u = int(round(float(uv[0])))
        v = int(round(float(uv[1])))

        if not (0 <= u < width and 0 <= v < height):
            continue

        z = float(depth[v, u])
        if not np.isfinite(z) or z <= 0.0:
            continue

        valid_depth += 1
        point = backproject_pixel(u, v, z, K)
        plane_distance = abs(
            float(np.dot(point - anchor_camera, surface_normal_camera))
        )

        if plane_distance <= max_plane_distance_m:
            same_surface += 1

    fraction = float(same_surface) / float(sample_count)
    return fraction, same_surface, valid_depth


def reconstruct_anchor_and_normal(
    depth: np.ndarray,
    K: np.ndarray,
    anchor_pixel: Any,
    normal_radius_px: int,
    normal_depth_gate_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = depth.shape
    anchor_u, anchor_v = validate_pixel(
        anchor_pixel,
        width,
        height,
        "anchor_pixel",
    )

    anchor_depth = robust_depth_at_pixel(depth, anchor_u, anchor_v)
    anchor_camera = backproject_pixel(anchor_u, anchor_v, anchor_depth, K)

    local_points = collect_local_surface_points(
        depth=depth,
        anchor_u=anchor_u,
        anchor_v=anchor_v,
        anchor_depth=anchor_depth,
        K=K,
        radius_px=normal_radius_px,
        depth_gate_m=normal_depth_gate_m,
    )

    surface_normal_camera, inliers = fit_surface_normal(
        local_points,
        anchor_camera,
    )

    debug = {
        "anchor_uv": [anchor_u, anchor_v],
        "anchor_depth_m": float(anchor_depth),
        "anchor_camera": anchor_camera,
        "normal_fit_point_count": int(len(inliers)),
        "surface_normal_camera": surface_normal_camera,
    }

    return anchor_camera, surface_normal_camera, debug


def reconstruct_explicit_axis_camera(
    depth: np.ndarray,
    K: np.ndarray,
    grasp: dict[str, Any],
    anchor_camera: np.ndarray,
    surface_normal_camera: np.ndarray,
    min_axis_pixel_distance_px: float,
    max_anchor_to_axis_line_px: float,
    max_axis_point_plane_distance_m: float,
    min_axis_line_same_surface_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = depth.shape

    start_u, start_v = validate_pixel(
        grasp["axis_pixel_start"], width, height, "axis_pixel_start"
    )
    end_u, end_v = validate_pixel(
        grasp["axis_pixel_end"], width, height, "axis_pixel_end"
    )
    anchor_u, anchor_v = validate_pixel(
        grasp["anchor_pixel"], width, height, "anchor_pixel"
    )

    anchor_uv = np.array([anchor_u, anchor_v], dtype=np.float64)
    start_uv = np.array([start_u, start_v], dtype=np.float64)
    end_uv = np.array([end_u, end_v], dtype=np.float64)

    axis_pixel_distance = float(np.linalg.norm(end_uv - start_uv))
    if axis_pixel_distance < min_axis_pixel_distance_px:
        raise ValueError(
            "Explicit-axis pixels are too close: "
            f"{axis_pixel_distance:.1f}px < {min_axis_pixel_distance_px:.1f}px."
        )

    anchor_to_axis_line_px = point_to_segment_distance_2d(
        anchor_uv, start_uv, end_uv
    )
    if anchor_to_axis_line_px > max_anchor_to_axis_line_px:
        raise ValueError(
            "Anchor is too far from the explicit-axis segment: "
            f"{anchor_to_axis_line_px:.1f}px > {max_anchor_to_axis_line_px:.1f}px."
        )

    start_depth = robust_depth_at_pixel(depth, start_u, start_v)
    end_depth = robust_depth_at_pixel(depth, end_u, end_v)

    start_camera = backproject_pixel(start_u, start_v, start_depth, K)
    end_camera = backproject_pixel(end_u, end_v, end_depth, K)

    start_plane_distance = abs(
        float(np.dot(start_camera - anchor_camera, surface_normal_camera))
    )
    end_plane_distance = abs(
        float(np.dot(end_camera - anchor_camera, surface_normal_camera))
    )

    for name, distance in (
        ("axis_pixel_start", start_plane_distance),
        ("axis_pixel_end", end_plane_distance),
    ):
        if distance > max_axis_point_plane_distance_m:
            raise ValueError(
                f"{name} is too far from the anchor local surface: "
                f"{distance * 1000.0:.1f} mm > "
                f"{max_axis_point_plane_distance_m * 1000.0:.1f} mm."
            )

    fraction, same_count, valid_count = axis_segment_same_surface_fraction(
        depth=depth,
        K=K,
        start_uv=start_uv,
        end_uv=end_uv,
        anchor_camera=anchor_camera,
        surface_normal_camera=surface_normal_camera,
        max_plane_distance_m=max_axis_point_plane_distance_m,
        sample_count=21,
    )

    if fraction < min_axis_line_same_surface_fraction:
        raise ValueError(
            "Explicit-axis image segment is not sufficiently consistent with "
            f"the anchor local surface: {fraction:.2f} < "
            f"{min_axis_line_same_surface_fraction:.2f}."
        )

    explicit_raw_camera = end_camera - start_camera
    explicit_axis_camera = normalize(
        explicit_raw_camera,
        "explicit_raw_camera",
    )

    debug = {
        "axis_start_uv": [start_u, start_v],
        "axis_end_uv": [end_u, end_v],
        "axis_start_depth_m": float(start_depth),
        "axis_end_depth_m": float(end_depth),
        "axis_pixel_distance_px": axis_pixel_distance,
        "anchor_to_axis_line_px": anchor_to_axis_line_px,
        "axis_start_plane_distance_m": start_plane_distance,
        "axis_end_plane_distance_m": end_plane_distance,
        "axis_line_same_surface_fraction": fraction,
        "axis_line_same_surface_count": same_count,
        "axis_line_valid_depth_count": valid_count,
        "explicit_raw_camera": explicit_raw_camera,
    }

    return explicit_axis_camera, debug


# =============================================================================
# JSON VALIDATION
# =============================================================================


def validate_grasp_spec(grasp: dict[str, Any], width: int, height: int) -> None:
    required = (
        "grasp_mode",
        "anchor_pixel",
        "open_axis_mode",
        "reason",
    )

    missing = [key for key in required if key not in grasp]
    if missing:
        raise ValueError(f"Missing JSON keys: {missing}")

    for key, modes in (("grasp_mode", GRASP_MODES), ("open_axis_mode", OPEN_AXIS_MODES)):
        if grasp[key] not in modes:
            raise ValueError(f"Invalid {key}={grasp[key]!r}; expected one of {modes}.")
    grasp_mode = grasp["grasp_mode"]
    open_axis_mode = grasp["open_axis_mode"]

    validate_pixel(grasp["anchor_pixel"], width, height, "anchor_pixel")

    if grasp_mode == "surface_normal" and open_axis_mode != "explicit_axis":
        raise ValueError(
            "grasp_mode='surface_normal' requires "
            "open_axis_mode='explicit_axis' because surface normal is parallel "
            "to the pointing direction."
        )

    if open_axis_mode == "surface_normal":
        if grasp.get("axis_pixel_start") is not None or grasp.get("axis_pixel_end") is not None:
            raise ValueError(
                "open_axis_mode='surface_normal' requires both axis pixels to be null or omitted."
            )
        grasp["axis_pixel_start"] = None
        grasp["axis_pixel_end"] = None
    else:
        if grasp.get("axis_pixel_start") is None or grasp.get("axis_pixel_end") is None:
            raise ValueError(
                "open_axis_mode='explicit_axis' requires both axis pixels."
            )
        validate_pixel(
            grasp["axis_pixel_start"], width, height, "axis_pixel_start"
        )
        validate_pixel(
            grasp["axis_pixel_end"], width, height, "axis_pixel_end"
        )

    # Optional signed manual offset: positive moves inward, negative outward.
    depth_cm = grasp.get("tcp_depth_cm", DEFAULT_TCP_DEPTH_CM)
    if isinstance(depth_cm, bool):
        raise ValueError("tcp_depth_cm must be numeric, not boolean.")

    try:
        depth_cm_float = float(depth_cm)
    except Exception as exc:
        raise ValueError("tcp_depth_cm must be numeric.") from exc

    if not np.isfinite(depth_cm_float):
        raise ValueError("tcp_depth_cm must be finite.")
    if not (-MAX_TCP_DEPTH_CM <= depth_cm_float <= MAX_TCP_DEPTH_CM):
        raise ValueError(
            f"tcp_depth_cm={depth_cm_float!r} invalid; expected {-MAX_TCP_DEPTH_CM} <= value <= "
            f"{MAX_TCP_DEPTH_CM} cm."
        )

    grasp["tcp_depth_cm"] = depth_cm_float


# =============================================================================
# GRASP CONSTRUCTION
# =============================================================================


def construct_grasp_pair(
    grasp: dict[str, Any],
    depth: np.ndarray,
    K: np.ndarray,
    T_world_camera: np.ndarray,
    normal_radius_px: int,
    normal_depth_gate_m: float,
    min_axis_pixel_distance_px: float,
    max_anchor_to_axis_line_px: float,
    max_axis_point_plane_distance_m: float,
    min_axis_line_same_surface_fraction: float,
    min_open_projection_norm: float,
) -> dict[str, Any]:
    # -------------------------------------------------------------------------
    # Anchor + outward local surface normal (positive direction -> camera)
    # -------------------------------------------------------------------------
    anchor_camera, surface_normal_camera, debug = reconstruct_anchor_and_normal(
        depth=depth,
        K=K,
        anchor_pixel=grasp["anchor_pixel"],
        normal_radius_px=normal_radius_px,
        normal_depth_gate_m=normal_depth_gate_m,
    )

    anchor_world = transform_point(T_world_camera, anchor_camera)
    surface_normal_world = normalize(
        transform_vector(T_world_camera, surface_normal_camera),
        "surface_normal_world",
    )

    # -------------------------------------------------------------------------
    # Deterministic pointing direction
    # -------------------------------------------------------------------------
    grasp_mode = grasp["grasp_mode"]

    if grasp_mode == "top":
        pointing_world = WORLD_DOWN.copy()
    elif grasp_mode == "surface_normal":
        pointing_world = -surface_normal_world
    else:
        raise ValueError(f"Unsupported grasp_mode={grasp_mode!r}")

    pointing_world = normalize(pointing_world, "pointing_world")

    # -------------------------------------------------------------------------
    # Opening/closing axis
    # -------------------------------------------------------------------------
    open_axis_mode = grasp["open_axis_mode"]
    explicit_debug = None

    if open_axis_mode == "surface_normal":
        open_candidate_world = surface_normal_world.copy()
    elif open_axis_mode == "explicit_axis":
        explicit_axis_camera, explicit_debug = reconstruct_explicit_axis_camera(
            depth=depth,
            K=K,
            grasp=grasp,
            anchor_camera=anchor_camera,
            surface_normal_camera=surface_normal_camera,
            min_axis_pixel_distance_px=min_axis_pixel_distance_px,
            max_anchor_to_axis_line_px=max_anchor_to_axis_line_px,
            max_axis_point_plane_distance_m=max_axis_point_plane_distance_m,
            min_axis_line_same_surface_fraction=min_axis_line_same_surface_fraction,
        )
        open_candidate_world = normalize(
            transform_vector(T_world_camera, explicit_axis_camera),
            "explicit_axis_world",
        )
    else:
        raise ValueError(f"Unsupported open_axis_mode={open_axis_mode!r}")

    open_source = open_axis_mode
    open_world_A, retained_fraction = project_perpendicular(
        open_candidate_world, pointing_world,
    )
    if retained_fraction < min_open_projection_norm:
        raise ValueError(
            f"{open_source} opening axis is too parallel to pointing: "
            f"retained_fraction={retained_fraction:.3f} < {min_open_projection_norm:.3f}."
        )
    open_world_A = normalize(open_world_A, "open_world_A")

    third_world_A = normalize(
        np.cross(pointing_world, open_world_A),
        "third_world_A",
    )

    # Recompute open for exact orthogonality while preserving its line.
    open_world_A = normalize(
        np.cross(third_world_A, pointing_world),
        "open_world_A_reorthogonalized",
    )

    # Positive TCP depth moves along -surface_normal; negative moves outward.
    tcp_depth_cm = float(grasp.get("tcp_depth_cm", DEFAULT_TCP_DEPTH_CM))
    tcp_depth_m = tcp_depth_cm / 100.0
    inward_world = -surface_normal_world

    tcp_world = anchor_world + tcp_depth_m * inward_world
    offset_vector_world = tcp_world - anchor_world

    # -------------------------------------------------------------------------
    # Pose A/B
    # -------------------------------------------------------------------------
    rotation_A = build_tcp_rotation(pointing_world, open_world_A)
    T_world_tcp_A = make_transform(rotation_A, tcp_world)

    open_world_B = -open_world_A
    third_world_B = -third_world_A
    rotation_B = build_tcp_rotation(pointing_world, open_world_B)
    T_world_tcp_B = make_transform(rotation_B, tcp_world)

    # Mapping sanity checks for the confirmed panda_hand_tcp convention.
    tcp_z_world_A = T_world_tcp_A[:3, 2]
    tcp_y_world_A = T_world_tcp_A[:3, 1]

    mapping_pointing_dot = float(np.dot(tcp_z_world_A, pointing_world))
    mapping_open_dot = float(np.dot(tcp_y_world_A, open_world_A))

    debug.update(
        {
            "surface_normal_world": surface_normal_world,
            "pointing_world": pointing_world,
            "open_source": open_source,
            "open_projection_retained_fraction": float(retained_fraction),
            "open_world_A": open_world_A,
            "third_world_A": third_world_A,
            "tcp_depth_m": tcp_depth_m,
            "inward_world": inward_world,
            "mapping_tcp_plus_z_dot_pointing": mapping_pointing_dot,
            "mapping_tcp_plus_y_dot_open": mapping_open_dot,
            "explicit_axis": explicit_debug,
        }
    )

    return {
        "anchor_world": anchor_world,
        "surface_normal_world": surface_normal_world,
        "inward_world": inward_world,
        "tcp_world": tcp_world,
        "tcp_depth_cm": tcp_depth_cm,
        "offset_vector_world": offset_vector_world,
        "grasp_mode": grasp_mode,
        "open_axis_mode": open_axis_mode,
        "open_source": open_source,
        "pointing_world": pointing_world,
        "open_world_A": open_world_A,
        "third_world_A": third_world_A,
        "open_world_B": open_world_B,
        "third_world_B": third_world_B,
        "debug": debug,
        "T_world_tcp_A": T_world_tcp_A,
        "T_world_tcp_B": T_world_tcp_B,
    }


# =============================================================================
# DEBUG OVERLAY / JSON SERIALIZATION
# =============================================================================


def draw_overlay(rgb: np.ndarray, grasps: list[dict[str, Any]]) -> np.ndarray:
    output = rgb.copy()

    for index, grasp in enumerate(grasps):
        anchor = tuple(int(round(float(v))) for v in grasp["anchor_pixel"])

        # BGR: anchor red.
        cv2.circle(output, anchor, 8, (0, 0, 255), -1)
        cv2.circle(output, anchor, 11, (255, 255, 255), 2)

        if grasp["open_axis_mode"] == "explicit_axis":
            start_value = grasp.get("axis_pixel_start")
            end_value = grasp.get("axis_pixel_end")
            if start_value is not None and end_value is not None:
                start = tuple(int(round(float(v))) for v in start_value)
                end = tuple(int(round(float(v))) for v in end_value)

                cv2.circle(output, start, 6, (255, 0, 0), -1)
                cv2.circle(output, end, 6, (0, 255, 0), -1)
                cv2.line(output, start, end, (0, 255, 255), 3)

        tcp_depth_cm = float(grasp.get("tcp_depth_cm", DEFAULT_TCP_DEPTH_CM))
        label = (
            f"G{index} mode={grasp['grasp_mode']} "
            f"open={grasp['open_axis_mode']} "
            f"d={tcp_depth_cm:g}cm"
        )

        position = (
            max(5, anchor[0] + 10),
            max(25, anchor[1] - 10),
        )

        cv2.putText(
            output,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return output


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct simplified Franka grasp poses from RGB + aligned depth + "
            "a manually saved VLM JSON specification."
        )
    )

    parser.add_argument("--rgb", default="camera_rgb.png", help="RGB image shown to the VLM.")
    parser.add_argument("--depth", default="initial_depth.npy", help="Aligned metric depth in metres.")
    parser.add_argument(
        "--json",
        "--spec-json",
        dest="json_path",
        required=True,
        help="JSON manually copied/saved from ChatGPT.",
    )
    parser.add_argument(
        "--camera-json",
        required=True,
        help=(
            "Runtime camera calibration JSON. Must contain <camera-name>.intrinsics "
            "and <camera-name>.extrinsics, where extrinsics is T_world_camera."
        ),
    )
    parser.add_argument(
        "--camera-name",
        default="external_sensor0",
        help="Camera key inside --camera-json. Default: external_sensor0.",
    )
    parser.add_argument("--output-dir", default="vlm_grasps")

    parser.add_argument("--normal-radius-px", type=int, default=DEFAULT_NORMAL_RADIUS_PX)
    parser.add_argument("--normal-depth-gate-m", type=float, default=DEFAULT_NORMAL_DEPTH_GATE_M)
    parser.add_argument(
        "--min-axis-pixel-distance-px", type=float, default=DEFAULT_MIN_AXIS_PIXEL_DISTANCE_PX,
    )
    parser.add_argument(
        "--max-anchor-to-axis-line-px", type=float, default=DEFAULT_MAX_ANCHOR_TO_AXIS_LINE_PX,
    )
    parser.add_argument(
        "--max-axis-point-plane-distance-m", type=float,
        default=DEFAULT_MAX_AXIS_POINT_PLANE_DISTANCE_M,
    )
    parser.add_argument(
        "--min-axis-line-same-surface-fraction", type=float,
        default=DEFAULT_MIN_AXIS_LINE_SAME_SURFACE_FRACTION,
    )
    parser.add_argument(
        "--min-open-projection-norm",
        type=float,
        default=DEFAULT_MIN_OPEN_PROJECTION_NORM,
        help=(
            "Minimum fraction of a candidate opening direction remaining after "
            "projection perpendicular to pointing."
        ),
    )

    args = parser.parse_args()

    rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Could not read RGB: {args.rgb}")

    depth = np.load(args.depth, allow_pickle=False).astype(np.float64)
    height, width = rgb.shape[:2]
    if depth.shape != (height, width):
        raise ValueError(
            "RGB/depth resolution mismatch:\n"
            f"  RGB   = {(height, width)}\n"
            f"  depth = {depth.shape}\n"
            "Use depth aligned to the RGB image."
        )

    json_path = Path(args.json_path)
    specification = json.loads(json_path.read_text())

    if not isinstance(specification, dict):
        raise ValueError("Top-level JSON must be an object.")
    if "grasps" not in specification:
        raise ValueError("Top-level JSON needs key 'grasps'.")

    grasps = specification["grasps"]
    if not isinstance(grasps, list) or len(grasps) == 0:
        raise ValueError("'grasps' must be a non-empty list.")

    # Use the task's runtime calibration for all grasp poses.
    K, T_world_camera = load_camera_calibration(
        args.camera_json,
        args.camera_name,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    pose_list: list[np.ndarray] = []

    for index, grasp in enumerate(grasps):
        try:
            if not isinstance(grasp, dict):
                raise ValueError("Each grasp entry must be a JSON object.")

            validate_grasp_spec(grasp, width, height)

            geometry = construct_grasp_pair(
                grasp=grasp,
                depth=depth,
                K=K,
                T_world_camera=T_world_camera,
                normal_radius_px=args.normal_radius_px,
                normal_depth_gate_m=args.normal_depth_gate_m,
                min_axis_pixel_distance_px=args.min_axis_pixel_distance_px,
                max_anchor_to_axis_line_px=args.max_anchor_to_axis_line_px,
                max_axis_point_plane_distance_m=args.max_axis_point_plane_distance_m,
                min_axis_line_same_surface_fraction=args.min_axis_line_same_surface_fraction,
                min_open_projection_norm=args.min_open_projection_norm,
            )

            results.append(
                {
                    "index": index,
                    "status": "ok",
                    "spec": grasp,
                    "geometry": geometry,
                }
            )

            pose_list.extend(
                [
                    geometry["T_world_tcp_A"],
                    geometry["T_world_tcp_B"],
                ]
            )

        except Exception as error:
            print(f"Grasp {index} rejected: {error}")
            results.append(
                {
                    "index": index,
                    "status": "rejected",
                    "spec": grasp,
                    "error": str(error),
                }
            )

    if pose_list:
        poses = np.stack(pose_list, axis=0)
    else:
        poses = np.empty((0, 4, 4), dtype=np.float64)

    np.save(
        output_dir / "grasp_poses_world.npy",
        poses,
        allow_pickle=False,
    )

    output_json = {
        "camera": {
            "source_json": str(Path(args.camera_json).resolve()),
            "camera_name": args.camera_name,
            "K": K,
            "T_world_camera": T_world_camera,
            "tcp_pointing_local_axis": TCP_POINTING_LOCAL_AXIS,
            "tcp_open_local_axis": TCP_OPEN_LOCAL_AXIS,
        },
        "source_spec_json": str(json_path),
        "results": results,
    }

    (output_dir / "grasp_poses_world.json").write_text(
        json.dumps(json_ready(output_json), indent=2)
    )

    overlay = draw_overlay(rgb, grasps)
    cv2.imwrite(str(output_dir / "grasp_overlay.png"), overlay)

    print(f"Grasps: {len(poses) // 2}/{len(grasps)} accepted; {len(poses)} poses saved.")
    print(f"Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
