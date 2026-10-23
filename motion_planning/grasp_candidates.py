from __future__ import annotations

"""Utilities for single- and multi-grasp inputs.

`grasp_poses` are optional GraspGen hand/TCP poses. When supplied, they
constrain the starting hand pose and define the hand-to-canonical-grasp
attachment; `q_start` seeds optimization without fixing the starting arm joints.
Without supplied poses, the attachment is computed from FK(q_start).
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from .se3 import as_transform


FloatArray = NDArray[np.float64]


def pose7_xyzw_to_transform(value: ArrayLike) -> FloatArray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Pose7 must have shape (7,), got {pose.shape}.")
    if not np.isfinite(pose).all():
        raise ValueError("Pose7 contains NaN or infinity.")
    quaternion = pose[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("Pose7 quaternion has zero norm.")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def coerce_grasp_pose_array(value: ArrayLike) -> FloatArray:
    """Return candidate poses with shape (K,4,4).

    Accepted input shapes:
      - (4,4)
      - (K,4,4)
      - (7,) as [x,y,z,qx,qy,qz,qw]
      - (K,7)
    """
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4, 4):
        array = array[None, ...]
    elif array.ndim == 3 and array.shape[1:] == (4, 4):
        pass
    elif array.shape == (7,):
        array = pose7_xyzw_to_transform(array)[None, ...]
    elif array.ndim == 2 and array.shape[1] == 7:
        array = np.stack([pose7_xyzw_to_transform(row) for row in array], axis=0)
    else:
        raise ValueError(
            "grasp pose input must have shape (4,4), (K,4,4), (7,), or (K,7); "
            f"got {array.shape}."
        )

    if not np.isfinite(array).all():
        raise ValueError("grasp pose input contains NaN or infinity.")
    return np.stack([as_transform(transform) for transform in array], axis=0)


def grasp_poses_to_world(
    poses: ArrayLike,
    *,
    frame: str,
    transform_world_camera: Optional[ArrayLike] = None,
    transform_world_object: Optional[ArrayLike] = None,
) -> FloatArray:
    """Convert GraspGen hand/TCP poses into the world/robot frame."""
    candidates = coerce_grasp_pose_array(poses)
    frame_normalized = str(frame).strip().lower()

    if frame_normalized in {"world", "robot", "base", "world_hand"}:
        return candidates

    if frame_normalized in {"camera", "camera_color_optical_frame"}:
        if transform_world_camera is None:
            raise ValueError("camera-frame grasp poses require T_world_camera.")
        parent = as_transform(transform_world_camera)
    elif frame_normalized in {"object", "controlled_object"}:
        if transform_world_object is None:
            raise ValueError("object-frame grasp poses require T_world_object.")
        parent = as_transform(transform_world_object)
    else:
        raise ValueError(
            "grasp_pose_frame must be 'world', 'camera', or 'object'; "
            f"got {frame!r}."
        )

    return np.einsum("ij,njk->nik", parent, candidates)


def grasp_pose_error(
    achieved_world_hand: ArrayLike,
    expected_world_hand: ArrayLike,
) -> tuple[float, float]:
    """Return position error [m] and rotation error [deg]."""
    achieved = as_transform(achieved_world_hand)
    expected = as_transform(expected_world_hand)
    position_error = float(
        np.linalg.norm(achieved[:3, 3] - expected[:3, 3])
    )
    relative = achieved[:3, :3].T @ expected[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    rotation_error_deg = float(np.degrees(np.arccos(cosine)))
    return position_error, rotation_error_deg


__all__ = [
    "coerce_grasp_pose_array",
    "grasp_pose_error",
    "grasp_poses_to_world",
    "pose7_xyzw_to_transform",
]
