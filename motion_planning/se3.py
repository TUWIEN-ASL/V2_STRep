from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_float_array(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _validate_rotation_matrix(
    rotation: ArrayLike,
    *,
    atol: float = 1e-6,
    name: str = "rotation",
) -> FloatArray:
    rotation_array = _as_float_array(rotation, name=name)
    if rotation_array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {rotation_array.shape}.")

    identity_error = rotation_array.T @ rotation_array - np.eye(3)
    if not np.allclose(identity_error, 0.0, atol=atol):
        raise ValueError(f"{name} is not orthonormal within atol={atol}.")

    determinant = float(np.linalg.det(rotation_array))
    if not np.isclose(determinant, 1.0, atol=atol):
        raise ValueError(
            f"{name} must have determinant +1, got {determinant:.8f}."
        )
    return rotation_array


def as_transform(
    transform: ArrayLike,
    *,
    atol: float = 1e-6,
    copy: bool = True,
) -> FloatArray:
    """
    Validate and return a homogeneous transform.

    Convention:
        T_A_B is the pose of frame B expressed in frame A.

    A point p_B is transformed into frame A by:
        p_A = R_A_B @ p_B + t_A_B
    """
    transform_array = _as_float_array(transform, name="transform")
    if transform_array.shape != (4, 4):
        raise ValueError(
            f"transform must have shape (4, 4), got {transform_array.shape}."
        )

    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(transform_array[3], expected_last_row, atol=atol):
        raise ValueError(
            "transform last row must be [0, 0, 0, 1]."
        )

    _validate_rotation_matrix(
        transform_array[:3, :3],
        atol=atol,
        name="transform rotation",
    )
    return transform_array.copy() if copy else transform_array


def make_transform(rotation: ArrayLike, translation: ArrayLike) -> FloatArray:
    """Construct T_A_B from a 3x3 rotation and 3D translation."""
    rotation_array = _validate_rotation_matrix(rotation)
    translation_array = _as_float_array(translation, name="translation")
    if translation_array.shape != (3,):
        raise ValueError(
            f"translation must have shape (3,), got {translation_array.shape}."
        )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_array
    transform[:3, 3] = translation_array
    return transform


def invert_transform(transform: ArrayLike) -> FloatArray:
    """Return T_B_A from T_A_B."""
    transform_array = as_transform(transform, copy=False)
    rotation = transform_array[:3, :3]
    translation = transform_array[:3, 3]

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def compose_transform(*transforms: ArrayLike) -> FloatArray:
    """
    Compose compatible transforms from left to right.

    Example:
        T_A_C = compose_transform(T_A_B, T_B_C)
    """
    if not transforms:
        return np.eye(4, dtype=np.float64)

    result = np.eye(4, dtype=np.float64)
    for index, transform in enumerate(transforms):
        result = result @ as_transform(transform, copy=False)
    return as_transform(result)


def relative_transform(reference_pose: ArrayLike, pose: ArrayLike) -> FloatArray:
    """
    Express `pose` relative to `reference_pose`.

    Given T_A_B and T_A_C, return T_B_C.
    """
    return compose_transform(invert_transform(reference_pose), pose)


def transform_point(transform: ArrayLike, point: ArrayLike) -> FloatArray:
    """Transform a 3D point using T_A_B."""
    transform_array = as_transform(transform, copy=False)
    point_array = _as_float_array(point, name="point")
    if point_array.shape != (3,):
        raise ValueError(f"point must have shape (3,), got {point_array.shape}.")
    return transform_array[:3, :3] @ point_array + transform_array[:3, 3]


def rotation_matrix_to_quaternion_xyzw(rotation: ArrayLike) -> FloatArray:
    """Convert a rotation matrix to a unit quaternion in [x, y, z, w] order."""
    matrix = _validate_rotation_matrix(rotation)
    trace = float(np.trace(matrix))

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))

        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale

    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)

    # Canonicalize the sign for deterministic serialization.
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def quaternion_xyzw_to_rotation_matrix(quaternion: ArrayLike) -> FloatArray:
    """Convert a quaternion in [x, y, z, w] order to a rotation matrix."""
    quaternion_array = _as_float_array(quaternion, name="quaternion")
    if quaternion_array.shape != (4,):
        raise ValueError(
            f"quaternion must have shape (4,), got {quaternion_array.shape}."
        )

    norm = float(np.linalg.norm(quaternion_array))
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero.")

    x, y, z, w = quaternion_array / norm

    rotation = np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
    return _validate_rotation_matrix(rotation)


def _slerp_xyzw(
    quaternion_start: FloatArray,
    quaternion_end: FloatArray,
    fraction: float,
) -> FloatArray:
    """Spherical interpolation between two unit xyzw quaternions."""
    q0 = quaternion_start / np.linalg.norm(quaternion_start)
    q1 = quaternion_end / np.linalg.norm(quaternion_end)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * fraction

    scale_0 = np.sin(theta_0 - theta) / sin_theta_0
    scale_1 = np.sin(theta) / sin_theta_0
    return scale_0 * q0 + scale_1 * q1


def interpolate_pose(
    start_pose: ArrayLike,
    end_pose: ArrayLike,
    fraction: float,
) -> FloatArray:
    """
    Interpolate between two poses.

    Translation is linear. Rotation uses quaternion SLERP.
    fraction must lie in [0, 1].
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must lie in [0, 1], got {fraction}.")

    start = as_transform(start_pose, copy=False)
    end = as_transform(end_pose, copy=False)

    translation = (
        (1.0 - fraction) * start[:3, 3]
        + fraction * end[:3, 3]
    )

    quaternion_start = rotation_matrix_to_quaternion_xyzw(start[:3, :3])
    quaternion_end = rotation_matrix_to_quaternion_xyzw(end[:3, :3])
    quaternion = _slerp_xyzw(quaternion_start, quaternion_end, fraction)
    rotation = quaternion_xyzw_to_rotation_matrix(quaternion)

    return make_transform(rotation, translation)
