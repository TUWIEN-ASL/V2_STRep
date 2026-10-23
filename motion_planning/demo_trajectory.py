from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .se3 import (
    as_transform,
    interpolate_pose,
    make_transform,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TrajectoryPreparationReport:
    """Summary of repairs applied to a demonstrated pose trajectory."""

    num_frames: int
    invalid_frame_indices: Tuple[int, ...]
    projected_rotation_indices: Tuple[int, ...]
    pca_sign_adjusted_indices: Tuple[int, ...]


@dataclass(frozen=True)
class PreparedDemoTrajectory:
    """
    Prepared demonstrated grasp-frame trajectory.

    poses[t] is T_world_grasp_demo at the original video frame t.
    Frame count and indexing are preserved.
    """

    poses: FloatArray
    frame_indices: IntArray
    report: TrajectoryPreparationReport


def _validate_raw_trajectory_shape(trajectory: ArrayLike) -> FloatArray:
    array = np.asarray(trajectory, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(
            "Demonstration trajectory must have shape (T, 4, 4), "
            f"got {array.shape}."
        )
    if array.shape[0] < 1:
        raise ValueError("Demonstration trajectory must contain at least one frame.")
    return array


def project_rotation_to_so3(rotation: ArrayLike) -> FloatArray:
    """
    Project a finite 3x3 matrix to the nearest proper rotation matrix.

    The projection uses SVD and explicitly enforces determinant +1.
    """
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation contains non-finite values.")

    left, _, right_transpose = np.linalg.svd(matrix)
    projected = left @ right_transpose

    if np.linalg.det(projected) < 0.0:
        left[:, -1] *= -1.0
        projected = left @ right_transpose

    return projected


def _is_rotation_matrix(
    rotation: FloatArray,
    *,
    atol: float,
) -> bool:
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        return False
    orthogonality_error = rotation.T @ rotation - np.eye(3)
    return bool(
        np.allclose(orthogonality_error, 0.0, atol=atol)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=atol)
    )


def _interpolate_missing_poses(
    repaired_poses: FloatArray,
    valid_mask: NDArray[np.bool_],
) -> FloatArray:
    """
    Fill invalid frames while preserving original frame indices.

    Translation is linearly interpolated. Rotation uses SE(3) pose
    interpolation between the nearest valid frames. Invalid leading or
    trailing frames copy the nearest valid pose.
    """
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        raise ValueError("No valid pose exists in the demonstration trajectory.")

    result = repaired_poses.copy()
    first_valid = int(valid_indices[0])
    last_valid = int(valid_indices[-1])

    for index in range(0, first_valid):
        result[index] = result[first_valid]

    for index in range(last_valid + 1, result.shape[0]):
        result[index] = result[last_valid]

    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        left_index = int(left)
        right_index = int(right)
        gap = right_index - left_index

        if gap <= 1:
            continue

        for index in range(left_index + 1, right_index):
            fraction = (index - left_index) / gap
            result[index] = interpolate_pose(
                result[left_index],
                result[right_index],
                float(fraction),
            )

    return result


def repair_pose_trajectory(
    trajectory: ArrayLike,
    *,
    rotation_atol: float = 1e-5,
) -> Tuple[FloatArray, Tuple[int, ...], Tuple[int, ...]]:
    """
    Repair a raw (T, 4, 4) demonstrated trajectory.

    Repairs:
    - finite poses with slightly invalid rotations are projected to SO(3);
    - the homogeneous last row is reset exactly;
    - frames containing non-finite translation or rotation values are
      interpolated from neighboring valid poses;
    - original frame count and frame indexing are preserved.

    Returns:
        repaired_trajectory,
        invalid_frame_indices,
        projected_rotation_indices
    """
    raw = _validate_raw_trajectory_shape(trajectory)
    repaired = np.empty_like(raw, dtype=np.float64)
    valid_mask = np.zeros(raw.shape[0], dtype=bool)

    invalid_indices = []
    projected_indices = []

    for index, pose in enumerate(raw):
        rotation = pose[:3, :3]
        translation = pose[:3, 3]

        if not (
            np.all(np.isfinite(rotation))
            and np.all(np.isfinite(translation))
        ):
            invalid_indices.append(index)
            repaired[index] = np.eye(4, dtype=np.float64)
            continue

        if _is_rotation_matrix(rotation, atol=rotation_atol):
            repaired_rotation = rotation
        else:
            repaired_rotation = project_rotation_to_so3(rotation)
            projected_indices.append(index)

        repaired[index] = make_transform(repaired_rotation, translation)
        valid_mask[index] = True

    repaired = _interpolate_missing_poses(repaired, valid_mask)

    for index, pose in enumerate(repaired):
        try:
            as_transform(pose, copy=False)
        except ValueError as error:
            raise ValueError(
                f"Repair produced an invalid pose at frame {index}: {error}"
            ) from error

    return (
        repaired,
        tuple(invalid_indices),
        tuple(projected_indices),
    )


def resolve_pca_axis_sign_continuity(
    trajectory: ArrayLike,
) -> Tuple[FloatArray, Tuple[int, ...]]:
    """
    Resolve frame-to-frame PCA axis sign ambiguity.

    PCA eigenvectors are sign ambiguous. For a right-handed frame whose
    columns are [x, y, z], the four equivalent proper rotations are:

        R @ diag(+1, +1, +1)
        R @ diag(+1, -1, -1)
        R @ diag(-1, +1, -1)
        R @ diag(-1, -1, +1)

    At each frame, this function chooses the equivalent rotation closest to
    the previously selected rotation.

    This resolves sign flips only. It does not resolve eigenvector
    permutations caused by nearly equal PCA eigenvalues.
    """
    poses = _validate_raw_trajectory_shape(trajectory).copy()

    for index, pose in enumerate(poses):
        as_transform(pose, copy=False)

    sign_candidates = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )

    adjusted_indices = []
    previous_rotation = poses[0, :3, :3].copy()

    for index in range(1, poses.shape[0]):
        raw_rotation = poses[index, :3, :3]

        candidates = [
            raw_rotation @ np.diag(signs)
            for signs in sign_candidates
        ]

        # Maximizing trace(previous^T candidate) minimizes relative angle.
        scores = [
            float(np.trace(previous_rotation.T @ candidate))
            for candidate in candidates
        ]
        best_index = int(np.argmax(scores))
        best_rotation = candidates[best_index]

        if best_index != 0:
            adjusted_indices.append(index)

        poses[index, :3, :3] = best_rotation
        previous_rotation = best_rotation

    return poses, tuple(adjusted_indices)


def prepare_demo_trajectory(
    trajectory: ArrayLike,
    *,
    resolve_pca_signs: bool = True,
    rotation_atol: float = 1e-5,
) -> PreparedDemoTrajectory:
    """
    Prepare T_world_grasp_demo while preserving video-frame indexing.

    No global resampling or smoothing is performed here because the phase
    intervals returned by the video VLM refer to original video frames.
    """
    repaired, invalid_indices, projected_indices = repair_pose_trajectory(
        trajectory,
        rotation_atol=rotation_atol,
    )

    if resolve_pca_signs:
        prepared, adjusted_indices = resolve_pca_axis_sign_continuity(repaired)
    else:
        prepared = repaired
        adjusted_indices = ()

    frame_indices = np.arange(prepared.shape[0], dtype=np.int64)
    report = TrajectoryPreparationReport(
        num_frames=prepared.shape[0],
        invalid_frame_indices=invalid_indices,
        projected_rotation_indices=projected_indices,
        pca_sign_adjusted_indices=adjusted_indices,
    )

    return PreparedDemoTrajectory(
        poses=prepared,
        frame_indices=frame_indices,
        report=report,
    )
