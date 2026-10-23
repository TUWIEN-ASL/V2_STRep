from __future__ import annotations

"""Panda trajectory optimization.

This module consumes the task-space ``MotionReference`` and optimizes a
complete robot joint trajectory with PyRoKi/JAXLS.

Heavy optional dependencies (JAXLS, jaxlie, PyRoKi, robot_descriptions) are
imported only by functions that need them. The data structures, metrics, and
serialization utilities remain usable in lightweight environments.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .reference_builder import MAX_AXIS_CONSTRAINTS, MotionReference
from .se3 import as_transform, invert_transform


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

# Kept in sync with reference_builder.py.
AXIS_RELATION_SAME_DIRECTION = 0
AXIS_RELATION_OPPOSITE_DIRECTION = 1
AXIS_RELATION_PARALLEL_EITHER_DIRECTION = 2


@dataclass(frozen=True)
class TrajectoryOptimizerConfig:
    """Robot settings, numerical weights, and timing for trajectory optimization.

    JAXLS minimizes squared residuals. The task ``*_scale`` values multiply
    residuals directly, while the per-phase tracking weights multiply squared
    task cost through an internal ``sqrt(weight)`` residual factor.
    """

    dt: float = 0.05

    # Task tracking. These values scale residuals.
    position_scale: float = 40.0
    axis_scale: float = 25.0
    full_rotation_scale: float = 4.0
    remaining_rotation_scale: float = 1.5

    # Pose-driven grasp initialization.  When a desired Panda TCP grasp pose is
    # supplied, q[0] is no longer fixed to an externally solved IK vector.
    # Instead the first ``fixed_start_samples`` TCP poses are constrained to the
    # supplied 6D grasp pose while their joint configurations remain optimization
    # variables.  These scales only condition the equality residual numerically.
    grasp_position_scale: float = 100.0
    grasp_rotation_scale: float = 10.0

    # Per-sample tracking cost weights. These multiply the squared task cost,
    # so the residual is scaled internally by sqrt(weight). Attractor samples
    # are identified by demo_relative_valid_mask=False. Individual phases can
    # override the type-level defaults using either a phase_id (recommended)
    # or a zero-based phase index encoded as a string.
    demo_tracking_weight: float = 1.0
    attractor_tracking_weight: float = 1.0
    phase_tracking_weights: tuple[tuple[str, float], ...] = ()

    # Joint regularization.
    joint_smoothness_scale: float = 3.0
    acceleration_scale: float = 0.5
    rest_scale: float = 0.005

    # One scalar angle is optimized for each reference phase whose
    # phase_symmetry.mode is "shared_offset".
    phase_symmetry_max_angle_rad: float = float(np.pi)
    phase_symmetry_rest_scale: float = 0.0

    # Boundary conditions.
    fixed_start_samples: int = 2
    zero_final_velocity: bool = True
    panda_arm_dof: int = 7

    # Keep every Panda arm joint this far inside its URDF position limits.
    # safe_lower = urdf_lower + margin
    # safe_upper = urdf_upper - margin
    joint_limit_margin_deg: float = 10.0

    # Collision checking. Collision is enabled only when a sphere JSON is given.
    table_z_world: float = 0.0
    self_collision_margin_m: float = 0.02
    self_collision_scale: float = 5.0
    world_collision_margin_m: float = 0.03

    # Z translation of panda_hand_tcp_joint (meters), matching the physical gripper.
    tcp_offset_m: float = 0.1234

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        for name in (
            "position_scale",
            "axis_scale",
            "full_rotation_scale",
            "remaining_rotation_scale",
            "grasp_position_scale",
            "grasp_rotation_scale",
            "demo_tracking_weight",
            "attractor_tracking_weight",
            "joint_smoothness_scale",
            "acceleration_scale",
            "rest_scale",
            "phase_symmetry_rest_scale",
            "joint_limit_margin_deg",
            "self_collision_margin_m",
            "self_collision_scale",
            "world_collision_margin_m",
            "tcp_offset_m",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative.")
        normalized_phase_weights: list[tuple[str, float]] = []
        seen_selectors: set[str] = set()
        for entry in self.phase_tracking_weights:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError(
                    "Each phase_tracking_weights entry must be (phase_selector, weight)."
                )
            selector, weight = entry
            selector_text = str(selector).strip()
            if not selector_text:
                raise ValueError("Phase-weight selectors must not be empty.")
            numeric_weight = float(weight)
            if numeric_weight < 0.0 or not np.isfinite(numeric_weight):
                raise ValueError(
                    "Phase tracking weights must be finite and non-negative."
                )
            if selector_text in seen_selectors:
                raise ValueError(
                    f"Duplicate phase tracking weight selector: {selector_text!r}."
                )
            seen_selectors.add(selector_text)
            normalized_phase_weights.append((selector_text, numeric_weight))
        object.__setattr__(
            self,
            "phase_tracking_weights",
            tuple(normalized_phase_weights),
        )

        if (
            self.phase_symmetry_max_angle_rad <= 0.0
            or not np.isfinite(self.phase_symmetry_max_angle_rad)
        ):
            raise ValueError(
                "phase_symmetry_max_angle_rad must be finite and positive."
            )

        if self.fixed_start_samples < 1:
            raise ValueError("fixed_start_samples must be at least 1.")
        if self.panda_arm_dof < 1:
            raise ValueError("panda_arm_dof must be positive.")


def safe_arm_joint_limits(
    robot: Any,
    arm_dof: int,
    margin_deg: float,
) -> tuple[FloatArray, FloatArray]:
    """Return URDF arm-joint limits shrunk inward by ``margin_deg``."""
    if arm_dof < 1:
        raise ValueError("arm_dof must be positive.")

    lower = np.asarray(robot.joints.lower_limits, dtype=np.float64).reshape(-1)
    upper = np.asarray(robot.joints.upper_limits, dtype=np.float64).reshape(-1)

    if len(lower) < arm_dof or len(upper) < arm_dof:
        raise ValueError(
            "Robot exposes fewer joint limits than panda_arm_dof: "
            f"{len(lower)} lower / {len(upper)} upper versus {arm_dof}."
        )

    margin_rad = float(np.deg2rad(float(margin_deg)))
    safe_lower = lower[:arm_dof] + margin_rad
    safe_upper = upper[:arm_dof] - margin_rad

    if not np.isfinite(safe_lower).all() or not np.isfinite(safe_upper).all():
        raise ValueError("Safe joint limits contain NaN or infinity.")
    if np.any(safe_lower >= safe_upper):
        bad = np.flatnonzero(safe_lower >= safe_upper)
        raise ValueError(
            "joint_limit_margin_deg is too large for joint(s): "
            + ", ".join(f"q{int(i)+1}" for i in bad)
        )

    return safe_lower, safe_upper



def build_sample_tracking_weights(
    reference: MotionReference,
    config: TrajectoryOptimizerConfig,
) -> FloatArray:
    """Build one squared-cost tracking weight per reference sample.

    Type-level defaults distinguish learned-demo samples from generated
    attractor samples. Explicit phase overrides are applied last. A selector
    first matches an exact ``phase_id``; otherwise it may be a zero-based phase
    index such as ``"0"``.
    """
    sample_count = reference.sample_count
    demo_mask = np.asarray(reference.demo_relative_valid_mask, dtype=bool)
    if demo_mask.shape != (sample_count,):
        raise ValueError(
            "demo_relative_valid_mask shape does not match reference sample count."
        )

    weights = np.where(
        demo_mask,
        float(config.demo_tracking_weight),
        float(config.attractor_tracking_weight),
    ).astype(np.float64, copy=False)

    phase_id_to_index = {phase.phase_id: i for i, phase in enumerate(reference.phases)}
    for selector, weight in config.phase_tracking_weights:
        if selector in phase_id_to_index:
            phase_number = phase_id_to_index[selector]
        else:
            try:
                phase_number = int(selector)
            except ValueError as error:
                available = ", ".join(phase_id_to_index)
                raise ValueError(
                    f"Unknown phase selector {selector!r}. Available phase IDs: "
                    f"{available or '<none>'}."
                ) from error
            if phase_number < 0 or phase_number >= len(reference.phases):
                raise ValueError(
                    f"Phase index {phase_number} is outside [0, "
                    f"{len(reference.phases) - 1}]."
                )

        weights[np.asarray(reference.phase_index) == phase_number] = float(weight)

    if weights.shape != (sample_count,):
        raise RuntimeError("Internal error while constructing sample tracking weights.")
    return weights


@dataclass(frozen=True)
class TrajectoryTrackingMetrics:
    position_errors_m: FloatArray
    axis_errors_deg: FloatArray
    full_rotation_errors_deg: FloatArray
    remaining_rotation_errors_deg: FloatArray
    joint_velocity_rad_s: FloatArray
    joint_acceleration_rad_s2: FloatArray

    def __post_init__(self) -> None:
        position = np.asarray(self.position_errors_m, dtype=np.float64)
        axis = np.asarray(self.axis_errors_deg, dtype=np.float64)
        full = np.asarray(self.full_rotation_errors_deg, dtype=np.float64)
        remaining = np.asarray(self.remaining_rotation_errors_deg, dtype=np.float64)
        velocity = np.asarray(self.joint_velocity_rad_s, dtype=np.float64)
        acceleration = np.asarray(self.joint_acceleration_rad_s2, dtype=np.float64)

        if position.ndim != 1:
            raise ValueError("position_errors_m must be one-dimensional.")
        n = position.shape[0]
        if axis.shape != (n, MAX_AXIS_CONSTRAINTS):
            raise ValueError(
                "axis_errors_deg must have shape "
                f"({n}, {MAX_AXIS_CONSTRAINTS}), got {axis.shape}."
            )
        for name, value in (
            ("full_rotation_errors_deg", full),
            ("remaining_rotation_errors_deg", remaining),
        ):
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {value.shape}.")
        if velocity.ndim != 2 or acceleration.ndim != 2:
            raise ValueError("Joint velocity and acceleration arrays must be 2-D.")

        object.__setattr__(self, "position_errors_m", position.copy())
        object.__setattr__(self, "axis_errors_deg", axis.copy())
        object.__setattr__(self, "full_rotation_errors_deg", full.copy())
        object.__setattr__(self, "remaining_rotation_errors_deg", remaining.copy())
        object.__setattr__(self, "joint_velocity_rad_s", velocity.copy())
        object.__setattr__(self, "joint_acceleration_rad_s2", acceleration.copy())


@dataclass(frozen=True)
class TrajectoryOptimizationResult:
    planned_joints: FloatArray
    achieved_hand_poses_world: FloatArray
    achieved_grasp_poses_world: FloatArray
    achieved_effect_poses_world: FloatArray
    transform_hand_grasp: FloatArray
    transform_grasp_effect: FloatArray
    phase_symmetry_angles_rad: FloatArray
    phase_symmetry_active_phase_mask: BoolArray
    metrics: TrajectoryTrackingMetrics
    config: TrajectoryOptimizerConfig
    target_link_name: str
    urdf_description: str

    def __post_init__(self) -> None:
        joints = np.asarray(self.planned_joints, dtype=np.float64)
        if joints.ndim != 2:
            raise ValueError("planned_joints must have shape (N, dof).")
        n = joints.shape[0]
        if n < 1:
            raise ValueError("planned_joints must contain at least one sample.")

        pose_fields = (
            "achieved_hand_poses_world",
            "achieved_grasp_poses_world",
            "achieved_effect_poses_world",
        )
        for name in pose_fields:
            poses = np.asarray(getattr(self, name), dtype=np.float64)
            if poses.shape != (n, 4, 4):
                raise ValueError(f"{name} must have shape ({n}, 4, 4), got {poses.shape}.")

            # Final serialization boundary: project harmless floating-point
            # orthogonality drift onto SE(3). This is intentionally repeated
            # here even though FK/composition are already sanitized, so result
            # construction cannot fail because of float32 JAX output.
            poses = np.stack(
                [project_transform_to_se3(pose) for pose in poses],
                axis=0,
            )
            for sample_index, pose in enumerate(poses):
                try:
                    as_transform(pose, copy=False)
                except ValueError as error:
                    raise ValueError(
                        f"{name}[{sample_index}] remains invalid after SE(3) "
                        f"projection: {error}"
                    ) from error
            object.__setattr__(self, name, poses.copy())

        if self.metrics.position_errors_m.shape != (n,):
            raise ValueError("metrics sample count does not match planned_joints.")

        symmetry_angles = np.asarray(
            self.phase_symmetry_angles_rad,
            dtype=np.float64,
        ).reshape(-1)
        symmetry_active = np.asarray(
            self.phase_symmetry_active_phase_mask,
            dtype=bool,
        ).reshape(-1)
        if symmetry_angles.shape != symmetry_active.shape:
            raise ValueError(
                "phase_symmetry_angles_rad and "
                "phase_symmetry_active_phase_mask must have the same shape."
            )
        if not np.all(np.isfinite(symmetry_angles)):
            raise ValueError("phase_symmetry_angles_rad contains non-finite values.")

        object.__setattr__(self, "planned_joints", joints.copy())
        object.__setattr__(
            self,
            "phase_symmetry_angles_rad",
            symmetry_angles.copy(),
        )
        object.__setattr__(
            self,
            "phase_symmetry_active_phase_mask",
            symmetry_active.copy(),
        )
        object.__setattr__(
            self,
            "transform_hand_grasp",
            project_transform_to_se3(self.transform_hand_grasp),
        )
        object.__setattr__(
            self,
            "transform_grasp_effect",
            project_transform_to_se3(self.transform_grasp_effect),
        )


# ---------------------------------------------------------------------------
# Pure NumPy geometry and diagnostics
# ---------------------------------------------------------------------------


def rotation_error_deg(rotation_a: ArrayLike, rotation_b: ArrayLike) -> float:
    """Geodesic angle between two 3x3 rotation matrices."""
    a = np.asarray(rotation_a, dtype=np.float64)
    b = np.asarray(rotation_b, dtype=np.float64)
    if a.shape != (3, 3) or b.shape != (3, 3):
        raise ValueError("rotation_error_deg expects two (3, 3) matrices.")
    relative = a.T @ b
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def axis_relation_error_deg(
    controlled_direction: ArrayLike,
    reference_direction: ArrayLike,
    relation_index: int,
) -> float:
    """Angular error for one reference axis relation."""
    controlled = np.asarray(controlled_direction, dtype=np.float64)
    reference = np.asarray(reference_direction, dtype=np.float64)
    if controlled.shape != (3,) or reference.shape != (3,):
        raise ValueError("Axis directions must have shape (3,).")
    controlled_norm = np.linalg.norm(controlled)
    reference_norm = np.linalg.norm(reference)
    if controlled_norm < 1e-12 or reference_norm < 1e-12:
        raise ValueError("Axis directions must be nonzero.")
    controlled = controlled / controlled_norm
    reference = reference / reference_norm
    dot = float(np.clip(np.dot(controlled, reference), -1.0, 1.0))

    if relation_index == AXIS_RELATION_SAME_DIRECTION:
        cosine = dot
    elif relation_index == AXIS_RELATION_OPPOSITE_DIRECTION:
        cosine = -dot
    elif relation_index == AXIS_RELATION_PARALLEL_EITHER_DIRECTION:
        cosine = abs(dot)
    else:
        raise ValueError(f"Unknown axis relation index: {relation_index}.")
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def compute_joint_dynamics(
    planned_joints: ArrayLike,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    joints = np.asarray(planned_joints, dtype=np.float64)
    if joints.ndim != 2:
        raise ValueError("planned_joints must have shape (N, dof).")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if joints.shape[0] < 2:
        velocity = np.empty((0, joints.shape[1]), dtype=np.float64)
    else:
        velocity = np.diff(joints, axis=0) / dt
    if velocity.shape[0] < 2:
        acceleration = np.empty((0, joints.shape[1]), dtype=np.float64)
    else:
        acceleration = np.diff(velocity, axis=0) / dt
    return velocity, acceleration


def project_transform_to_se3(transform: ArrayLike) -> FloatArray:
    """Project a numerically noisy homogeneous matrix onto SE(3).

    PyRoKi/JAXLie FK is often evaluated in float32.  The resulting matrix can
    differ from an exactly orthonormal rotation by a few parts in 1e-6, and
    composing several such matrices can slightly amplify that drift.  This
    function keeps the translation unchanged and projects only the 3x3 block
    to the nearest proper rotation using the SVD/polar projection.

    It is intended for post-optimization FK output and serialization.  It does
    not change the optimization variables or residuals.
    """
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transform contains non-finite values.")

    u, _, vt = np.linalg.svd(matrix[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = matrix[:3, 3]
    return result


def compute_hand_grasp_attachment(
    transform_world_hand_initial: ArrayLike,
    transform_world_grasp_initial: ArrayLike,
) -> FloatArray:
    """Return T_hand_grasp from an already-grasped robot configuration."""
    hand = as_transform(transform_world_hand_initial)
    grasp = as_transform(transform_world_grasp_initial)
    return invert_transform(hand) @ grasp


def compose_control_poses(
    hand_poses_world: ArrayLike,
    transform_hand_grasp: ArrayLike,
    transform_grasp_effect: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    hand = np.asarray(hand_poses_world, dtype=np.float64)
    if hand.ndim != 3 or hand.shape[1:] != (4, 4):
        raise ValueError("hand_poses_world must have shape (N, 4, 4).")
    hand_grasp = as_transform(transform_hand_grasp)
    grasp_effect = as_transform(transform_grasp_effect)

    # Sanitize the post-FK matrices before and after composition.  This removes
    # harmless float32 orthogonality drift without changing translations.
    hand = np.stack([project_transform_to_se3(pose) for pose in hand], axis=0)
    grasp_raw = np.einsum("nij,jk->nik", hand, hand_grasp)
    grasp = np.stack(
        [project_transform_to_se3(pose) for pose in grasp_raw],
        axis=0,
    )
    effect_raw = np.einsum("nij,jk->nik", grasp, grasp_effect)
    effect = np.stack(
        [project_transform_to_se3(pose) for pose in effect_raw],
        axis=0,
    )
    return grasp, effect



def phase_symmetry_active_phase_mask(
    reference: MotionReference,
) -> BoolArray:
    """Validate shared-offset phases and return one active flag per phase.

    Supported semantics generated by the reference builder:
      - one constant scalar angle for a complete phase,
      - a world-frame axis stored in phase_symmetry_axis_world,
      - full-demo rotation tracking,
      - no simultaneous axis constraints or remaining-rotation constraint.
    """
    sample_count = reference.sample_count
    phase_count = len(reference.phases)

    sample_mask = np.asarray(
        reference.phase_symmetry_mask,
        dtype=bool,
    )
    phase_index = np.asarray(
        reference.phase_index,
        dtype=np.int64,
    )
    axes = np.asarray(
        reference.phase_symmetry_axis_world,
        dtype=np.float64,
    )

    if sample_mask.shape != (sample_count,):
        raise ValueError(
            "phase_symmetry_mask shape does not match reference sample count."
        )
    if phase_index.shape != (sample_count,):
        raise ValueError(
            "phase_index shape does not match reference sample count."
        )
    if axes.shape != (sample_count, 3):
        raise ValueError(
            "phase_symmetry_axis_world must have shape "
            f"({sample_count}, 3), got {axes.shape}."
        )
    if np.any(phase_index < 0) or np.any(phase_index >= phase_count):
        raise ValueError("phase_index contains an invalid phase number.")

    active_phase_mask = np.zeros(phase_count, dtype=bool)

    for phase_number in np.unique(phase_index[sample_mask]):
        phase_number = int(phase_number)
        phase_samples = phase_index == phase_number

        # A shared offset belongs to the complete retained phase, not a
        # frame-wise subset.
        if not np.all(sample_mask[phase_samples]):
            raise NotImplementedError(
                f"Phase {reference.phases[phase_number].phase_id!r} has a "
                "partial phase_symmetry mask. V1 requires one shared offset "
                "for the complete phase."
            )

        if not np.all(
            np.asarray(reference.full_rotation_mask, dtype=bool)[phase_samples]
        ):
            raise NotImplementedError(
                f"Phase {reference.phases[phase_number].phase_id!r} uses "
                "shared_offset without full-demo rotation tracking."
            )

        if np.any(
            np.asarray(reference.axis_constraint_mask, dtype=bool)[phase_samples]
        ):
            raise NotImplementedError(
                f"Phase {reference.phases[phase_number].phase_id!r} combines "
                "shared_offset with axis constraints. This combination is not "
                "implemented in V1."
            )

        if np.any(
            np.asarray(
                reference.remaining_rotation_fixed_mask,
                dtype=bool,
            )[phase_samples]
        ):
            raise NotImplementedError(
                f"Phase {reference.phases[phase_number].phase_id!r} combines "
                "shared_offset with remaining_rotation=fixed_from_demo."
            )

        phase_axes = axes[phase_samples]
        if not np.all(np.isfinite(phase_axes)):
            raise ValueError(
                f"Phase {reference.phases[phase_number].phase_id!r} has a "
                "non-finite shared symmetry axis."
            )

        norms = np.linalg.norm(phase_axes, axis=1)
        if np.any(norms < 1e-9):
            raise ValueError(
                f"Phase {reference.phases[phase_number].phase_id!r} has a "
                "zero shared symmetry axis."
            )

        normalized = phase_axes / norms[:, None]
        if not np.allclose(
            normalized,
            normalized[0][None, :],
            atol=1e-6,
            rtol=0.0,
        ):
            raise NotImplementedError(
                f"Phase {reference.phases[phase_number].phase_id!r} does not "
                "use one constant world-frame symmetry axis."
            )

        active_phase_mask[phase_number] = True

    target = np.asarray(
        reference.transform_world_target_current,
        dtype=np.float64,
    )
    if target.shape != (4, 4) or not np.all(np.isfinite(target)):
        raise ValueError(
            "transform_world_target_current must be a finite (4, 4) transform "
            "when shared phase symmetry is used."
        )

    return active_phase_mask


def _axis_angle_rotation_matrix(
    axis: ArrayLike,
    angle_rad: float,
) -> FloatArray:
    axis_array = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis_array))
    if norm < 1e-12:
        raise ValueError("Rotation axis must be nonzero.")
    x, y, z = axis_array / norm
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    one_minus_c = 1.0 - c
    return np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ],
        dtype=np.float64,
    )


def apply_phase_symmetry_to_reference(
    reference: MotionReference,
    phase_angles_rad: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, BoolArray]:
    """Apply optimized shared phase rotations to reference targets.

    Returns:
        position_targets_world,
        full_rotation_targets_world,
        desired_grasp_poses_world,
        desired_effect_poses_world,
        active_phase_mask
    """
    active_phase_mask = phase_symmetry_active_phase_mask(reference)
    phase_count = len(reference.phases)
    angles = np.asarray(phase_angles_rad, dtype=np.float64).reshape(-1)
    if angles.shape != (phase_count,):
        raise ValueError(
            f"phase_angles_rad must have shape ({phase_count},), "
            f"got {angles.shape}."
        )

    position_targets = np.asarray(
        reference.position_targets_world,
        dtype=np.float64,
    ).copy()
    rotation_targets = np.asarray(
        reference.full_rotation_targets_world,
        dtype=np.float64,
    ).copy()
    desired_grasp = np.asarray(
        reference.transform_world_grasp_gauge,
        dtype=np.float64,
    ).copy()
    desired_effect = np.asarray(
        reference.transform_world_effect_gauge,
        dtype=np.float64,
    ).copy()

    phase_index = np.asarray(reference.phase_index, dtype=np.int64)
    symmetry_axes = np.asarray(
        reference.phase_symmetry_axis_world,
        dtype=np.float64,
    )
    target_origin = np.asarray(
        reference.transform_world_target_current,
        dtype=np.float64,
    )[:3, 3]

    for phase_number in np.flatnonzero(active_phase_mask):
        samples = phase_index == int(phase_number)
        first_sample = int(np.flatnonzero(samples)[0])
        rotation_delta = _axis_angle_rotation_matrix(
            symmetry_axes[first_sample],
            float(angles[phase_number]),
        )

        position_targets[samples] = (
            target_origin[None, :]
            + np.einsum(
                "ij,nj->ni",
                rotation_delta,
                position_targets[samples] - target_origin[None, :],
            )
        )
        rotation_targets[samples] = np.einsum(
            "ij,njk->nik",
            rotation_delta,
            rotation_targets[samples],
        )

        for pose_trajectory in (desired_grasp, desired_effect):
            pose_trajectory[samples, :3, 3] = (
                target_origin[None, :]
                + np.einsum(
                    "ij,nj->ni",
                    rotation_delta,
                    pose_trajectory[samples, :3, 3]
                    - target_origin[None, :],
                )
            )
            pose_trajectory[samples, :3, :3] = np.einsum(
                "ij,njk->nik",
                rotation_delta,
                pose_trajectory[samples, :3, :3],
            )

    return (
        position_targets,
        rotation_targets,
        desired_grasp,
        desired_effect,
        active_phase_mask,
    )


def compute_tracking_metrics(
    reference: MotionReference,
    achieved_effect_poses_world: ArrayLike,
    planned_joints: ArrayLike,
    dt: float,
    *,
    position_targets_world: Optional[ArrayLike] = None,
    full_rotation_targets_world: Optional[ArrayLike] = None,
) -> TrajectoryTrackingMetrics:
    achieved = np.asarray(achieved_effect_poses_world, dtype=np.float64)
    if achieved.shape != reference.transform_world_effect_gauge.shape:
        raise ValueError(
            "achieved_effect_poses_world must match the Step-7 trajectory shape."
        )
    n = achieved.shape[0]

    position_targets = (
        np.asarray(reference.position_targets_world, dtype=np.float64)
        if position_targets_world is None
        else np.asarray(position_targets_world, dtype=np.float64)
    )
    rotation_targets = (
        np.asarray(reference.full_rotation_targets_world, dtype=np.float64)
        if full_rotation_targets_world is None
        else np.asarray(full_rotation_targets_world, dtype=np.float64)
    )
    if position_targets.shape != (n, 3):
        raise ValueError(
            f"position_targets_world must have shape ({n}, 3), "
            f"got {position_targets.shape}."
        )
    if rotation_targets.shape != (n, 3, 3):
        raise ValueError(
            f"full_rotation_targets_world must have shape ({n}, 3, 3), "
            f"got {rotation_targets.shape}."
        )

    position_errors = np.full(n, np.nan, dtype=np.float64)
    active_position = np.asarray(reference.position_mask, dtype=bool)
    position_errors[active_position] = np.linalg.norm(
        achieved[active_position, :3, 3]
        - position_targets[active_position],
        axis=1,
    )

    axis_errors = np.full((n, MAX_AXIS_CONSTRAINTS), np.nan, dtype=np.float64)
    for sample in range(n):
        rotation = achieved[sample, :3, :3]
        for slot in range(MAX_AXIS_CONSTRAINTS):
            if not reference.axis_constraint_mask[sample, slot]:
                continue
            axis_index = int(reference.axis_controlled_axis_index[sample, slot])
            relation_index = int(reference.axis_relation_index[sample, slot])
            axis_errors[sample, slot] = axis_relation_error_deg(
                rotation[:, axis_index],
                reference.axis_reference_directions_world[sample, slot],
                relation_index,
            )

    full_errors = np.full(n, np.nan, dtype=np.float64)
    for sample in np.flatnonzero(reference.full_rotation_mask):
        full_errors[sample] = rotation_error_deg(
            achieved[sample, :3, :3],
            rotation_targets[sample],
        )

    remaining_errors = np.full(n, np.nan, dtype=np.float64)
    for sample in np.flatnonzero(reference.remaining_rotation_fixed_mask):
        remaining_errors[sample] = rotation_error_deg(
            achieved[sample, :3, :3],
            rotation_targets[sample],
        )

    velocity, acceleration = compute_joint_dynamics(planned_joints, dt)
    return TrajectoryTrackingMetrics(
        position_errors_m=position_errors,
        axis_errors_deg=axis_errors,
        full_rotation_errors_deg=full_errors,
        remaining_rotation_errors_deg=remaining_errors,
        joint_velocity_rad_s=velocity,
        joint_acceleration_rad_s2=acceleration,
    )


def validate_reference_for_optimizer(reference: MotionReference) -> None:
    """Validate the motion-reference features supported by this optimizer."""
    phase_symmetry_active_phase_mask(reference)


# ---------------------------------------------------------------------------
# PyRoKi helpers
# ---------------------------------------------------------------------------


def adapt_start_configuration(q_given: ArrayLike, dof: int) -> FloatArray:
    q = np.asarray(q_given, dtype=np.float64).reshape(-1)
    if q.size == dof:
        return q.copy()
    if q.size > dof:
        warnings.warn(f"Given {q.size} joints but robot has {dof}; truncating.")
        return q[:dof].copy()
    result = np.zeros(dof, dtype=np.float64)
    result[: q.size] = q
    warnings.warn(
        f"Given {q.size} joints but robot has {dof}; padding missing joints with zero."
    )
    return result


def resolve_target_link(robot: Any, requested: str) -> tuple[str, int]:
    candidates = [requested, "panda_hand_tcp", "panda_hand", "panda_link8"]
    for name in candidates:
        if name in robot.links.names:
            return name, robot.links.names.index(name)
    raise ValueError(
        f"None of {candidates} exists. Available links:\n{robot.links.names}"
    )


def fk_matrix(robot: Any, q: ArrayLike, link_index: int) -> FloatArray:
    import jax.numpy as jnp
    import jaxlie

    link_poses = robot.forward_kinematics(jnp.asarray(q))
    matrix = np.asarray(
        jaxlie.SE3(link_poses[link_index]).as_matrix(),
        dtype=np.float64,
    )
    return project_transform_to_se3(matrix)


def fk_trajectory(robot: Any, joints: ArrayLike, link_index: int) -> FloatArray:
    joints_array = np.asarray(joints, dtype=np.float64)
    if joints_array.ndim != 2:
        raise ValueError("joints must have shape (N, dof).")
    return np.stack(
        [fk_matrix(robot, q, link_index) for q in joints_array],
        axis=0,
    )


def _jax_tracking_residual(
    *,
    jnp: Any,
    effect_translation: Any,
    effect_rotation_matrix: Any,
    position_targets: Any,
    position_mask: Any,
    axis_mask: Any,
    controlled_axis_index: Any,
    relation_index: Any,
    axis_directions: Any,
    full_rotation_targets: Any,
    full_rotation_mask: Any,
    remaining_rotation_mask: Any,
    position_scale: Any,
    axis_scale: Any,
    full_rotation_scale: Any,
    remaining_rotation_scale: Any,
    sample_tracking_weight: Any = 1.0,
) -> Any:
    """Fixed-size residual, vectorized over any leading sample dimensions."""
    p_residual = (
        effect_translation - position_targets
    ) * position_mask[..., None] * position_scale

    safe_axis_index = jnp.maximum(controlled_axis_index, 0)
    basis = jnp.eye(3, dtype=effect_rotation_matrix.dtype)[safe_axis_index]
    # R @ e_axis, supporting leading sample and axis-slot dimensions.
    current_axes = jnp.einsum("...ij,...sj->...si", effect_rotation_matrix, basis)
    safe_axis_directions = jnp.where(
        axis_mask[..., None],
        axis_directions,
        jnp.zeros_like(axis_directions),
    )

    same = current_axes - safe_axis_directions
    opposite = current_axes + safe_axis_directions
    parallel = jnp.cross(current_axes, safe_axis_directions)
    axis_residual = jnp.where(
        (relation_index == AXIS_RELATION_SAME_DIRECTION)[..., None],
        same,
        jnp.where(
            (relation_index == AXIS_RELATION_OPPOSITE_DIRECTION)[..., None],
            opposite,
            parallel,
        ),
    )
    # Inactive reference slots contain NaN direction placeholders. Use where
    # instead of multiplying by zero because NaN * 0 remains NaN.
    axis_residual = jnp.where(
        axis_mask[..., None],
        axis_residual * axis_scale,
        jnp.zeros_like(axis_residual),
    )

    rotation_difference = effect_rotation_matrix - full_rotation_targets
    full_residual = (
        rotation_difference
        * full_rotation_mask[..., None, None]
        * full_rotation_scale
    )
    remaining_residual = (
        rotation_difference
        * remaining_rotation_mask[..., None, None]
        * remaining_rotation_scale
    )

    residual = jnp.concatenate(
        [
            p_residual.reshape((*p_residual.shape[:-1], 3)),
            axis_residual.reshape((*axis_residual.shape[:-2], 3 * MAX_AXIS_CONSTRAINTS)),
            full_residual.reshape((*full_residual.shape[:-2], 9)),
            remaining_residual.reshape((*remaining_residual.shape[:-2], 9)),
        ],
        axis=-1,
    )
    # JAXLS squares residuals. sqrt(weight) therefore makes the user-facing
    # phase weight multiply the squared tracking cost exactly once.
    return residual * jnp.sqrt(sample_tracking_weight)[..., None]


def solve_reference_ik(
    robot: Any,
    q_seed: FloatArray,
    target_link_index: int,
    transform_hand_grasp: FloatArray,
    reference: MotionReference,
    sample_index: int,
    config: TrajectoryOptimizerConfig,
) -> FloatArray:
    """Solve a single-sample IK problem using the trajectory tracking residuals."""
    import jax.numpy as jnp
    import jaxlie
    import jaxls
    import pyroki as pk

    q_var = robot.joint_var_cls(0)
    T_H_G = jaxlie.SE3.from_matrix(jnp.asarray(transform_hand_grasp))
    T_G_E = jaxlie.SE3.from_matrix(jnp.asarray(reference.transform_grasp_effect))

    @jaxls.Cost.factory(name="single_sample_effect_reference")
    def reference_cost(
        vals,
        robot_arg,
        var,
        T_H_G_arg,
        T_G_E_arg,
        link_index_arg,
        p_target,
        p_mask,
        axis_mask_arg,
        controlled_axis_arg,
        relation_arg,
        axis_direction_arg,
        rotation_target,
        rotation_mask,
        remaining_mask,
        p_scale,
        a_scale,
        r_scale,
        remaining_scale,
        sample_weight_arg,
    ):
        q = vals[var]
        link_poses = robot_arg.forward_kinematics(q)
        T_W_H = jaxlie.SE3(link_poses[link_index_arg])
        T_W_E = T_W_H @ T_H_G_arg @ T_G_E_arg
        residual = _jax_tracking_residual(
            jnp=jnp,
            effect_translation=T_W_E.translation(),
            effect_rotation_matrix=T_W_E.rotation().as_matrix(),
            position_targets=p_target,
            position_mask=p_mask,
            axis_mask=axis_mask_arg,
            controlled_axis_index=controlled_axis_arg,
            relation_index=relation_arg,
            axis_directions=axis_direction_arg,
            full_rotation_targets=rotation_target,
            full_rotation_mask=rotation_mask,
            remaining_rotation_mask=remaining_mask,
            position_scale=p_scale,
            axis_scale=a_scale,
            full_rotation_scale=r_scale,
            remaining_rotation_scale=remaining_scale,
            sample_tracking_weight=sample_weight_arg,
        )
        return residual.flatten()

    i = int(sample_index)
    costs = [
        reference_cost(
            robot,
            q_var,
            T_H_G,
            T_G_E,
            jnp.asarray(target_link_index, dtype=jnp.int32),
            jnp.asarray(reference.position_targets_world[i]),
            jnp.asarray(reference.position_mask[i]),
            jnp.asarray(reference.axis_constraint_mask[i]),
            jnp.asarray(reference.axis_controlled_axis_index[i]),
            jnp.asarray(reference.axis_relation_index[i]),
            jnp.asarray(reference.axis_reference_directions_world[i]),
            jnp.asarray(reference.full_rotation_targets_world[i]),
            jnp.asarray(reference.full_rotation_mask[i]),
            jnp.asarray(reference.remaining_rotation_fixed_mask[i]),
            jnp.asarray(config.position_scale),
            jnp.asarray(config.axis_scale),
            jnp.asarray(config.full_rotation_scale),
            jnp.asarray(config.remaining_rotation_scale),
            jnp.asarray(1.0),
        ),
        pk.costs.rest_cost(
            q_var,
            jnp.asarray(q_seed),
            jnp.asarray(max(config.rest_scale, 0.01)),
        ),
        pk.costs.limit_constraint(robot, q_var),
    ]

    safe_lower_np, safe_upper_np = safe_arm_joint_limits(
        robot,
        config.panda_arm_dof,
        config.joint_limit_margin_deg,
    )
    safe_lower = jnp.asarray(safe_lower_np)
    safe_upper = jnp.asarray(safe_upper_np)

    @jaxls.Cost.factory(
        kind="constraint_geq_zero",
        name="single_sample_safe_arm_joint_lower_bound",
    )
    def safe_arm_joint_lower_bound(vals, var):
        return (
            vals[var][..., : config.panda_arm_dof] - safe_lower
        ).flatten()

    @jaxls.Cost.factory(
        kind="constraint_geq_zero",
        name="single_sample_safe_arm_joint_upper_bound",
    )
    def safe_arm_joint_upper_bound(vals, var):
        return (
            safe_upper - vals[var][..., : config.panda_arm_dof]
        ).flatten()

    costs.append(safe_arm_joint_lower_bound(q_var))
    costs.append(safe_arm_joint_upper_bound(q_var))

    problem = jaxls.LeastSquaresProblem(costs=costs, variables=[q_var]).analyze()
    solution = problem.solve(
        verbose=False,
        initial_vals=jaxls.VarValues.make((q_var.with_value(jnp.asarray(q_seed)),))
    )
    return np.asarray(solution[q_var], dtype=np.float64)


def solve_hand_pose_ik(
    robot: Any,
    q_seed: FloatArray,
    target_link_index: int,
    target_hand_pose_world: ArrayLike,
    config: TrajectoryOptimizerConfig,
) -> FloatArray:
    """Return an IK warm start for a desired Panda hand/TCP pose.

    This solve is *initialization only*.  The returned joint vector is not fixed
    during trajectory optimization; the 6D hand pose is enforced again inside
    ``optimize_joint_trajectory`` while q[0] remains a decision variable.
    """
    import jax.numpy as jnp
    import jaxlie
    import jaxls
    import pyroki as pk

    target = as_transform(target_hand_pose_world)
    q_var = robot.joint_var_cls(0)
    target_position = jnp.asarray(target[:3, 3])
    target_rotation = jaxlie.SO3.from_matrix(jnp.asarray(target[:3, :3]))

    @jaxls.Cost.factory(name="grasp_pose_ik_warm_start")
    def grasp_pose_ik_cost(
        vals,
        robot_arg,
        var,
        link_index_arg,
        target_position_arg,
        target_rotation_arg,
        position_scale,
        rotation_scale,
    ):
        q = vals[var]
        link_poses = robot_arg.forward_kinematics(q)
        T_W_H = jaxlie.SE3(link_poses[link_index_arg])
        position_residual = (
            T_W_H.translation() - target_position_arg
        ) * position_scale
        rotation_residual = (
            target_rotation_arg.inverse() @ T_W_H.rotation()
        ).log() * rotation_scale
        return jnp.concatenate((position_residual, rotation_residual), axis=-1)

    costs = [
        grasp_pose_ik_cost(
            robot,
            q_var,
            jnp.asarray(target_link_index, dtype=jnp.int32),
            target_position,
            target_rotation,
            jnp.asarray(config.grasp_position_scale),
            jnp.asarray(config.grasp_rotation_scale),
        ),
        pk.costs.rest_cost(
            q_var,
            jnp.asarray(q_seed),
            jnp.asarray(max(config.rest_scale, 0.01)),
        ),
        pk.costs.limit_constraint(robot, q_var),
    ]

    safe_lower_np, safe_upper_np = safe_arm_joint_limits(
        robot,
        config.panda_arm_dof,
        config.joint_limit_margin_deg,
    )
    safe_lower = jnp.asarray(safe_lower_np)
    safe_upper = jnp.asarray(safe_upper_np)

    @jaxls.Cost.factory(
        kind="constraint_geq_zero",
        name="grasp_pose_ik_safe_arm_joint_lower_bound",
    )
    def safe_arm_joint_lower_bound(vals, var):
        return (
            vals[var][..., : config.panda_arm_dof] - safe_lower
        ).flatten()

    @jaxls.Cost.factory(
        kind="constraint_geq_zero",
        name="grasp_pose_ik_safe_arm_joint_upper_bound",
    )
    def safe_arm_joint_upper_bound(vals, var):
        return (
            safe_upper - vals[var][..., : config.panda_arm_dof]
        ).flatten()

    costs.append(safe_arm_joint_lower_bound(q_var))
    costs.append(safe_arm_joint_upper_bound(q_var))

    problem = jaxls.LeastSquaresProblem(costs=costs, variables=[q_var]).analyze()
    solution = problem.solve(
        verbose=False,
        initial_vals=jaxls.VarValues.make((q_var.with_value(jnp.asarray(q_seed)),))
    )
    return np.asarray(solution[q_var], dtype=np.float64)


def make_initial_trajectory(
    robot: Any,
    q_start: FloatArray,
    target_link_index: int,
    transform_hand_grasp: FloatArray,
    reference: MotionReference,
    config: TrajectoryOptimizerConfig,
    supplied_initial_trajectory: Optional[ArrayLike] = None,
) -> FloatArray:
    n = reference.sample_count
    dof = q_start.shape[0]
    if supplied_initial_trajectory is not None:
        initial = np.asarray(supplied_initial_trajectory, dtype=np.float64)
        if initial.shape != (n, dof):
            raise ValueError(
                f"Initial trajectory must have shape {(n, dof)}, got {initial.shape}."
            )
        return initial.copy()

    q_final = solve_reference_ik(
        robot,
        q_start,
        target_link_index,
        transform_hand_grasp,
        reference,
        n - 1,
        config,
    )
    return np.linspace(q_start, q_final, n, dtype=np.float64)


def load_collision_objects(
    robot: Any,
    urdf: Any,
    sphere_json: Optional[Path],
    table_z_world: float,
) -> tuple[Any, list[Any]]:
    import pyroki as pk

    if sphere_json is None:
        return None, []
    sphere_path = Path(sphere_json)
    if not sphere_path.exists():
        raise FileNotFoundError(sphere_path)
    sphere_data = json.loads(sphere_path.read_text(encoding="utf-8"))
    robot_collision = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_data,
        urdf=urdf,
    )
    table = pk.collision.HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, table_z_world], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    return robot_collision, [table]


def optimize_joint_trajectory(
    robot: Any,
    q_start: FloatArray,
    target_link_index: int,
    transform_hand_grasp: FloatArray,
    reference: MotionReference,
    initial_trajectory: FloatArray,
    config: TrajectoryOptimizerConfig,
    *,
    robot_collision: Any = None,
    world_collision: Sequence[Any] = (),
    desired_start_hand_pose_world: Optional[ArrayLike] = None,
) -> tuple[FloatArray, FloatArray]:
    """Run full trajectory optimization, including shared phase offsets."""
    import jax
    import jax.numpy as jnp
    import jaxlie
    import jaxls
    import pyroki as pk

    validate_reference_for_optimizer(reference)
    n = reference.sample_count
    if n < 2:
        raise ValueError("Trajectory optimization requires at least two samples.")

    phase_count = len(reference.phases)
    if phase_count < 1:
        raise ValueError("Trajectory optimization requires at least one phase.")

    active_phase_mask = phase_symmetry_active_phase_mask(reference)

    class PhaseSymmetryVar(
        jaxls.Var[jax.Array],
        default_factory=lambda: jnp.zeros((1,), dtype=jnp.float32),
    ):
        """One world-axis rotation angle in radians per reference phase."""

    traj_var = robot.joint_var_cls(jnp.arange(n))
    phase_var_all = PhaseSymmetryVar(jnp.arange(phase_count))
    phase_var_per_sample = PhaseSymmetryVar(
        jnp.asarray(reference.phase_index, dtype=jnp.int32)
    )
    robot_batched = jax.tree.map(lambda x: x[None], robot)

    T_H_G = jaxlie.SE3.from_matrix(jnp.asarray(transform_hand_grasp))
    T_G_E = jaxlie.SE3.from_matrix(jnp.asarray(reference.transform_grasp_effect))
    T_H_G_batched = jax.tree.map(lambda x: x[None], T_H_G)
    T_G_E_batched = jax.tree.map(lambda x: x[None], T_G_E)

    target_link_index_jax = jnp.asarray(target_link_index, dtype=jnp.int32)
    sample_tracking_weights = build_sample_tracking_weights(reference, config)
    target_origin_world = np.asarray(
        reference.transform_world_target_current,
        dtype=np.float64,
    )[:3, 3]

    @jaxls.Cost.factory(name="step8_effect_tracking")
    def effect_tracking_cost(
        vals,
        robot_arg,
        var,
        phase_var,
        T_H_G_arg,
        T_G_E_arg,
        link_index_arg,
        positions_arg,
        position_mask_arg,
        axis_mask_arg,
        controlled_axis_arg,
        relation_arg,
        axis_directions_arg,
        full_rotation_targets_arg,
        full_rotation_mask_arg,
        remaining_rotation_mask_arg,
        phase_symmetry_mask_arg,
        phase_symmetry_axis_arg,
        target_origin_arg,
        p_scale,
        a_scale,
        r_scale,
        remaining_scale,
        sample_weight_arg,
    ):
        q = vals[var]
        link_poses = robot_arg.forward_kinematics(q)
        T_W_H = jaxlie.SE3(link_poses[..., link_index_arg, :])
        T_W_E = T_W_H @ T_H_G_arg @ T_G_E_arg

        phase_angle = vals[phase_var][..., 0]
        phase_angle = jnp.where(
            phase_symmetry_mask_arg,
            phase_angle,
            jnp.zeros_like(phase_angle),
        )
        safe_axis = jnp.where(
            phase_symmetry_mask_arg[..., None],
            phase_symmetry_axis_arg,
            jnp.asarray([0.0, 0.0, 1.0]),
        )
        safe_axis = safe_axis / jnp.maximum(
            jnp.linalg.norm(safe_axis, axis=-1, keepdims=True),
            1e-12,
        )
        symmetry_rotation = jaxlie.SO3.exp(
            safe_axis * phase_angle[..., None]
        ).as_matrix()

        shifted_position_targets = (
            target_origin_arg
            + jnp.einsum(
                "...ij,...j->...i",
                symmetry_rotation,
                positions_arg - target_origin_arg,
            )
        )
        shifted_rotation_targets = jnp.einsum(
            "...ij,...jk->...ik",
            symmetry_rotation,
            full_rotation_targets_arg,
        )

        residual = _jax_tracking_residual(
            jnp=jnp,
            effect_translation=T_W_E.translation(),
            effect_rotation_matrix=T_W_E.rotation().as_matrix(),
            position_targets=shifted_position_targets,
            position_mask=position_mask_arg,
            axis_mask=axis_mask_arg,
            controlled_axis_index=controlled_axis_arg,
            relation_index=relation_arg,
            axis_directions=axis_directions_arg,
            full_rotation_targets=shifted_rotation_targets,
            full_rotation_mask=full_rotation_mask_arg,
            remaining_rotation_mask=remaining_rotation_mask_arg,
            position_scale=p_scale,
            axis_scale=a_scale,
            full_rotation_scale=r_scale,
            remaining_rotation_scale=remaining_scale,
            sample_tracking_weight=sample_weight_arg,
        )
        return residual.flatten()

    costs: list[Any] = [
        effect_tracking_cost(
            robot_batched,
            traj_var,
            phase_var_per_sample,
            T_H_G_batched,
            T_G_E_batched,
            target_link_index_jax,
            jnp.asarray(reference.position_targets_world),
            jnp.asarray(reference.position_mask),
            jnp.asarray(reference.axis_constraint_mask),
            jnp.asarray(reference.axis_controlled_axis_index),
            jnp.asarray(reference.axis_relation_index),
            jnp.asarray(reference.axis_reference_directions_world),
            jnp.asarray(reference.full_rotation_targets_world),
            jnp.asarray(reference.full_rotation_mask),
            jnp.asarray(reference.remaining_rotation_fixed_mask),
            jnp.asarray(reference.phase_symmetry_mask),
            jnp.asarray(reference.phase_symmetry_axis_world),
            jnp.asarray(target_origin_world)[None, :],
            jnp.asarray(config.position_scale),
            jnp.asarray(config.axis_scale),
            jnp.asarray(config.full_rotation_scale),
            jnp.asarray(config.remaining_rotation_scale),
            jnp.asarray(sample_tracking_weights),
        ),
        pk.costs.rest_cost(
            traj_var,
            jnp.asarray(q_start)[None],
            jnp.asarray(config.rest_scale),
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, n)),
            robot.joint_var_cls(jnp.arange(0, n - 1)),
            jnp.asarray(config.joint_smoothness_scale),
        ),
        pk.costs.limit_constraint(robot_batched, traj_var),
        pk.costs.limit_velocity_constraint(
            robot_batched,
            robot.joint_var_cls(jnp.arange(1, n)),
            robot.joint_var_cls(jnp.arange(0, n - 1)),
            config.dt,
        ),
    ]

    # Additional controller-safe position envelope for the Panda arm.
    # This is stricter than the raw URDF limit_constraint above.
    safe_lower_np, safe_upper_np = safe_arm_joint_limits(
        robot,
        config.panda_arm_dof,
        config.joint_limit_margin_deg,
    )
    # Keep the arm-limit vectors as closure constants. JAXLS interprets array
    # Cost arguments as carrying batch axes, while these are common 7-D limits.
    safe_lower = jnp.asarray(safe_lower_np)
    safe_upper = jnp.asarray(safe_upper_np)

    @jaxls.Cost.factory(
        kind="constraint_geq_zero",
        name="safe_arm_joint_lower_bound",
    )
    def safe_arm_joint_lower_bound(vals, var):
        return (
            vals[var][..., : config.panda_arm_dof] - safe_lower
        ).flatten()

    @jaxls.Cost.factory(
        kind="constraint_geq_zero",
        name="safe_arm_joint_upper_bound",
    )
    def safe_arm_joint_upper_bound(vals, var):
        return (
            safe_upper - vals[var][..., : config.panda_arm_dof]
        ).flatten()

    costs.append(safe_arm_joint_lower_bound(traj_var))
    costs.append(safe_arm_joint_upper_bound(traj_var))

    active_phase_ids = np.flatnonzero(active_phase_mask)
    inactive_phase_ids = np.flatnonzero(~active_phase_mask)

    if active_phase_ids.size:
        active_phase_var = PhaseSymmetryVar(
            jnp.asarray(active_phase_ids, dtype=jnp.int32)
        )

        @jaxls.Cost.factory(
            kind="constraint_geq_zero",
            name="phase_symmetry_lower_bound",
        )
        def phase_symmetry_lower_bound(vals, var, limit):
            return (vals[var][..., 0] + limit).flatten()

        @jaxls.Cost.factory(
            kind="constraint_geq_zero",
            name="phase_symmetry_upper_bound",
        )
        def phase_symmetry_upper_bound(vals, var, limit):
            return (limit - vals[var][..., 0]).flatten()

        symmetry_limit = jnp.asarray(config.phase_symmetry_max_angle_rad)
        costs.append(
            phase_symmetry_lower_bound(active_phase_var, symmetry_limit)
        )
        costs.append(
            phase_symmetry_upper_bound(active_phase_var, symmetry_limit)
        )

        if config.phase_symmetry_rest_scale > 0.0:
            @jaxls.Cost.factory(name="phase_symmetry_rest")
            def phase_symmetry_rest(vals, var, scale):
                return vals[var][..., 0].flatten() * scale

            costs.append(
                phase_symmetry_rest(
                    active_phase_var,
                    jnp.asarray(config.phase_symmetry_rest_scale),
                )
            )

    if inactive_phase_ids.size:
        inactive_phase_var = PhaseSymmetryVar(
            jnp.asarray(inactive_phase_ids, dtype=jnp.int32)
        )

        @jaxls.Cost.factory(
            kind="constraint_eq_zero",
            name="inactive_phase_symmetry_zero",
        )
        def inactive_phase_symmetry_zero(vals, var):
            return vals[var][..., 0].flatten()

        costs.append(inactive_phase_symmetry_zero(inactive_phase_var))

    if n >= 5 and config.acceleration_scale > 0.0:
        costs.append(
            pk.costs.five_point_acceleration_cost(
                robot.joint_var_cls(jnp.arange(2, n - 2)),
                robot.joint_var_cls(jnp.arange(4, n)),
                robot.joint_var_cls(jnp.arange(3, n - 1)),
                robot.joint_var_cls(jnp.arange(1, n - 3)),
                robot.joint_var_cls(jnp.arange(0, n - 4)),
                config.dt,
                jnp.asarray(config.acceleration_scale),
            )
        )

    fixed_count = min(config.fixed_start_samples, n)
    if desired_start_hand_pose_world is None:
        # Legacy joint-driven mode: preserve the previous exact q-start behavior.
        @jaxls.Cost.factory(kind="constraint_eq_zero", name="fixed_start")
        def fixed_start(vals, var, q0):
            return (vals[var] - q0).flatten()

        costs.append(
            fixed_start(
                robot.joint_var_cls(jnp.arange(fixed_count)),
                jnp.asarray(q_start)[None],
            )
        )
    else:
        # Pose-driven mode: q[0] (and the optional short initial hold) remains an
        # optimization variable.  Only FK(q) is constrained to the desired 6D
        # Panda TCP grasp pose.
        desired_start_hand_pose = as_transform(desired_start_hand_pose_world)
        target_position = jnp.asarray(desired_start_hand_pose[:3, 3])
        target_rotation = jaxlie.SO3.from_matrix(
            jnp.asarray(desired_start_hand_pose[:3, :3])
        )

        # JAXLS treats the leading axis of every Cost argument as its batch
        # axis.  The trajectory variable below has batch size ``fixed_count``
        # (normally 2), while a raw position vector has shape (3,) and an SO3
        # stores a quaternion with shape (4,).  Passing those raw values makes
        # JAXLS incorrectly interpret 3 and 4 as batch sizes.  Add an explicit
        # singleton batch axis so the one desired grasp pose broadcasts across
        # every fixed-start sample.
        target_position_batched = target_position[None, :]
        target_rotation_batched = jax.tree.map(
            lambda x: x[None, ...], target_rotation
        )

        @jaxls.Cost.factory(
            kind="constraint_eq_zero",
            name="fixed_start_hand_pose",
        )
        def fixed_start_hand_pose(
            vals,
            robot_arg,
            var,
            link_index_arg,
            target_position_arg,
            target_rotation_arg,
            position_scale,
            rotation_scale,
        ):
            q = vals[var]
            link_poses = robot_arg.forward_kinematics(q)
            T_W_H = jaxlie.SE3(link_poses[..., link_index_arg, :])
            position_residual = (
                T_W_H.translation() - target_position_arg
            ) * position_scale
            rotation_residual = (
                target_rotation_arg.inverse() @ T_W_H.rotation()
            ).log() * rotation_scale
            return jnp.concatenate(
                (position_residual, rotation_residual),
                axis=-1,
            ).flatten()

        costs.append(
            fixed_start_hand_pose(
                robot_batched,
                robot.joint_var_cls(jnp.arange(fixed_count)),
                target_link_index_jax,
                target_position_batched,
                target_rotation_batched,
                jnp.asarray(config.grasp_position_scale),
                jnp.asarray(config.grasp_rotation_scale),
            )
        )

    if config.zero_final_velocity:
        @jaxls.Cost.factory(kind="constraint_eq_zero", name="zero_final_velocity")
        def zero_final_velocity(vals, last_var, previous_var):
            return (vals[last_var] - vals[previous_var]).flatten()

        costs.append(
            zero_final_velocity(
                robot.joint_var_cls(n - 1),
                robot.joint_var_cls(n - 2),
            )
        )

    dof = int(robot.joints.num_actuated_joints)
    if dof > config.panda_arm_dof:
        fixed_ids = jnp.arange(config.panda_arm_dof, dof)
        fixed_values = jnp.asarray(q_start[config.panda_arm_dof:])

        @jaxls.Cost.factory(kind="constraint_eq_zero", name="locked_non_arm_joints")
        def locked_non_arm_joints(vals, var, ids, values):
            return (vals[var][..., ids] - values).flatten()

        costs.append(locked_non_arm_joints(traj_var, fixed_ids, fixed_values))

    if robot_collision is not None:
        collision_batched = jax.tree.map(lambda x: x[None], robot_collision)
        costs.append(
            pk.costs.self_collision_cost(
                robot_batched,
                collision_batched,
                traj_var,
                config.self_collision_margin_m,
                config.self_collision_scale,
            )
        )

        @jaxls.Cost.factory(
            kind="constraint_geq_zero",
            name="swept_world_collision",
        )
        def swept_world_collision(
            vals,
            robot_arg,
            collision_arg,
            obstacle_arg,
            previous_var,
            current_var,
            margin,
        ):
            swept = collision_arg.get_swept_capsules(
                robot_arg,
                vals[previous_var],
                vals[current_var],
            )
            distance = pk.collision.collide(
                swept.reshape((-1, 1)),
                obstacle_arg.reshape((1, -1)),
            )
            return distance.flatten() - margin

        for obstacle in world_collision:
            obstacle_batched = jax.tree.map(lambda x: x[None], obstacle)
            costs.append(
                swept_world_collision(
                    robot_batched,
                    collision_batched,
                    obstacle_batched,
                    robot.joint_var_cls(jnp.arange(0, n - 1)),
                    robot.joint_var_cls(jnp.arange(1, n)),
                    config.world_collision_margin_m,
                )
            )

    problem = jaxls.LeastSquaresProblem(
        costs=costs,
        variables=[traj_var, phase_var_all],
    ).analyze()
    solution = problem.solve(
        verbose=False,
        initial_vals=jaxls.VarValues.make(
            (
                traj_var.with_value(jnp.asarray(initial_trajectory)),
                phase_var_all.with_value(
                    jnp.zeros((phase_count, 1), dtype=jnp.float32)
                ),
            )
        )
    )
    planned_joints = np.asarray(solution[traj_var], dtype=np.float64)

    # Never silently accept a numerical solution outside the configured safe
    # arm-joint envelope.
    arm_planned = planned_joints[:, : config.panda_arm_dof]
    tolerance_rad = 1e-3
    violation = (
        (arm_planned < safe_lower_np[None, :] - tolerance_rad)
        | (arm_planned > safe_upper_np[None, :] + tolerance_rad)
    )
    if np.any(violation):
        sample_ids, joint_ids = np.nonzero(violation)
        first_sample = int(sample_ids[0])
        first_joint = int(joint_ids[0])
        raise RuntimeError(
            "Optimized trajectory violates the safe joint-limit envelope: "
            f"sample={first_sample}, q{first_joint + 1}="
            f"{arm_planned[first_sample, first_joint]:.6f} rad, "
            f"allowed=[{safe_lower_np[first_joint]:.6f}, "
            f"{safe_upper_np[first_joint]:.6f}] rad."
        )

    phase_angles = np.asarray(
        solution[phase_var_all],
        dtype=np.float64,
    ).reshape(phase_count)
    return planned_joints, phase_angles


def run_trajectory_optimization(
    reference: MotionReference,
    q_start_given: ArrayLike,
    *,
    urdf_description: str = "panda_description",
    target_link: str = "panda_hand_tcp",
    config: Optional[TrajectoryOptimizerConfig] = None,
    initial_trajectory: Optional[ArrayLike] = None,
    sphere_json: Optional[Path] = None,
    desired_start_hand_pose_world: Optional[ArrayLike] = None,
    transform_world_grasp_current: Optional[ArrayLike] = None,
) -> TrajectoryOptimizationResult:
    """Load the Panda model, optimize, evaluate FK, and return a result bundle."""
    import pyroki as pk
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    cfg = config or TrajectoryOptimizerConfig()
    validate_reference_for_optimizer(reference)

    urdf = load_robot_description(urdf_description)

    urdf.joint_map["panda_hand_tcp_joint"].origin[2, 3] = cfg.tcp_offset_m

    robot = pk.Robot.from_urdf(urdf)
    q_start = adapt_start_configuration(
        q_start_given,
        int(robot.joints.num_actuated_joints),
    )
    target_link_name, target_link_index = resolve_target_link(robot, target_link)

    desired_start_hand_pose: Optional[FloatArray]
    if desired_start_hand_pose_world is None:
        desired_start_hand_pose = None
        attachment_hand_pose = fk_matrix(robot, q_start, target_link_index)
        q_grasp_seed = q_start
    else:
        desired_start_hand_pose = as_transform(desired_start_hand_pose_world)
        attachment_hand_pose = desired_start_hand_pose
        # IK is used only to produce a numerically useful warm start.  q_grasp_seed
        # is deliberately not imposed as a boundary condition below.
        q_grasp_seed = solve_hand_pose_ik(
            robot,
            q_start,
            target_link_index,
            desired_start_hand_pose,
            cfg,
        )

    # The rigid hand-to-canonical-grasp attachment must be defined from the
    # desired GraspGen TCP pose, not from an approximate IK/FK realization.
    if transform_world_grasp_current is None:
        # Backward-compatible fallback.
        attachment_grasp_pose = reference.transform_world_grasp_gauge[0]
    else:
        attachment_grasp_pose = as_transform(transform_world_grasp_current)

    T_H_G = compute_hand_grasp_attachment(
        attachment_hand_pose,
        attachment_grasp_pose,
    )

    warm_start = make_initial_trajectory(
        robot,
        q_grasp_seed,
        target_link_index,
        T_H_G,
        reference,
        cfg,
        supplied_initial_trajectory=initial_trajectory,
    )

    robot_collision, world_collision = load_collision_objects(
        robot,
        urdf,
        sphere_json,
        cfg.table_z_world,
    )
    planned_joints, phase_symmetry_angles = optimize_joint_trajectory(
        robot,
        q_grasp_seed,
        target_link_index,
        T_H_G,
        reference,
        warm_start,
        cfg,
        robot_collision=robot_collision,
        world_collision=world_collision,
        desired_start_hand_pose_world=desired_start_hand_pose,
    )

    hand_poses = fk_trajectory(robot, planned_joints, target_link_index)
    grasp_poses, effect_poses = compose_control_poses(
        hand_poses,
        T_H_G,
        reference.transform_grasp_effect,
    )
    (
        optimized_position_targets,
        optimized_rotation_targets,
        _,
        _,
        active_phase_mask,
    ) = apply_phase_symmetry_to_reference(
        reference,
        phase_symmetry_angles,
    )
    metrics = compute_tracking_metrics(
        reference,
        effect_poses,
        planned_joints,
        cfg.dt,
        position_targets_world=optimized_position_targets,
        full_rotation_targets_world=optimized_rotation_targets,
    )
    return TrajectoryOptimizationResult(
        planned_joints=planned_joints,
        achieved_hand_poses_world=hand_poses,
        achieved_grasp_poses_world=grasp_poses,
        achieved_effect_poses_world=effect_poses,
        transform_hand_grasp=T_H_G,
        transform_grasp_effect=reference.transform_grasp_effect,
        phase_symmetry_angles_rad=phase_symmetry_angles,
        phase_symmetry_active_phase_mask=active_phase_mask,
        metrics=metrics,
        config=cfg,
        target_link_name=target_link_name,
        urdf_description=urdf_description,
    )


# ---------------------------------------------------------------------------
# Multi-start trajectory optimization
# ---------------------------------------------------------------------------


def _finite_max(values: ArrayLike, default: float = 0.0) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(default)
    return float(np.max(finite))


def compute_tracking_selection_cost(
    reference: MotionReference,
    result: TrajectoryOptimizationResult,
) -> dict[str, float]:
    """Evaluate one optimized trajectory using the optimizer tracking residuals.

    This reconstructs the task-tracking part of the existing JAXLS objective
    using NumPy. It intentionally uses the same:
      - position residual,
      - axis relation residual,
      - full rotation-matrix residual,
      - remaining-rotation residual,
      - per-sample tracking weights,
      - configured residual scales.

    The function also reports motion-quality diagnostics. Joint limits,
    velocity limits, fixed-start constraints, zero-final-velocity constraints,
    and optional swept collision constraints are already handled inside the
    optimizer.
    """
    achieved = np.asarray(
        result.achieved_effect_poses_world,
        dtype=np.float64,
    )
    if achieved.shape != reference.transform_world_effect_gauge.shape:
        raise ValueError(
            "Achieved effect trajectory shape does not match the Step-7 reference."
        )

    config = result.config
    sample_weights = build_sample_tracking_weights(reference, config)
    sample_count = achieved.shape[0]
    (
        optimized_position_targets,
        optimized_rotation_targets,
        _,
        _,
        _,
    ) = apply_phase_symmetry_to_reference(
        reference,
        result.phase_symmetry_angles_rad,
    )

    # Position residual: exactly the same scaled vector used in JAXLS.
    position_mask = np.asarray(reference.position_mask, dtype=bool)
    position_residual = np.zeros((sample_count, 3), dtype=np.float64)
    position_residual[position_mask] = (
        achieved[position_mask, :3, 3]
        - optimized_position_targets[position_mask]
    ) * float(config.position_scale)

    # Axis-relation residual: same / opposite use vector differences;
    # parallel-either-direction uses a cross product.
    axis_mask = np.asarray(reference.axis_constraint_mask, dtype=bool)
    controlled_indices = np.asarray(
        reference.axis_controlled_axis_index,
        dtype=np.int64,
    )
    relation_indices = np.asarray(
        reference.axis_relation_index,
        dtype=np.int64,
    )
    reference_directions = np.asarray(
        reference.axis_reference_directions_world,
        dtype=np.float64,
    )

    axis_residual = np.zeros(
        (sample_count, MAX_AXIS_CONSTRAINTS, 3),
        dtype=np.float64,
    )
    for sample_index in range(sample_count):
        rotation = achieved[sample_index, :3, :3]
        for slot in range(MAX_AXIS_CONSTRAINTS):
            if not axis_mask[sample_index, slot]:
                continue

            axis_index = int(controlled_indices[sample_index, slot])
            relation_index = int(relation_indices[sample_index, slot])
            current_axis = rotation[:, axis_index]
            reference_axis = reference_directions[sample_index, slot]

            if relation_index == AXIS_RELATION_SAME_DIRECTION:
                residual = current_axis - reference_axis
            elif relation_index == AXIS_RELATION_OPPOSITE_DIRECTION:
                residual = current_axis + reference_axis
            elif relation_index == AXIS_RELATION_PARALLEL_EITHER_DIRECTION:
                residual = np.cross(current_axis, reference_axis)
            else:
                raise ValueError(
                    f"Unknown axis relation index: {relation_index}."
                )

            axis_residual[sample_index, slot] = (
                residual * float(config.axis_scale)
            )

    achieved_rotation = achieved[:, :3, :3]
    target_rotation = optimized_rotation_targets
    rotation_difference = achieved_rotation - target_rotation

    full_mask = np.asarray(reference.full_rotation_mask, dtype=bool)
    full_residual = np.zeros((sample_count, 3, 3), dtype=np.float64)
    full_residual[full_mask] = (
        rotation_difference[full_mask]
        * float(config.full_rotation_scale)
    )

    remaining_mask = np.asarray(
        reference.remaining_rotation_fixed_mask,
        dtype=bool,
    )
    remaining_residual = np.zeros((sample_count, 3, 3), dtype=np.float64)
    remaining_residual[remaining_mask] = (
        rotation_difference[remaining_mask]
        * float(config.remaining_rotation_scale)
    )

    # JAXLS squares residuals, and sqrt(sample_weight) multiplies each sample.
    per_sample_tracking_cost = sample_weights * (
        np.sum(position_residual**2, axis=1)
        + np.sum(axis_residual**2, axis=(1, 2))
        + np.sum(full_residual**2, axis=(1, 2))
        + np.sum(remaining_residual**2, axis=(1, 2))
    )
    tracking_cost = float(np.sum(per_sample_tracking_cost))

    joints = np.asarray(result.planned_joints, dtype=np.float64)
    arm_joints = joints[:, : int(config.panda_arm_dof)]
    q_start = arm_joints[0]

    rest_cost = float(
        float(config.rest_scale) ** 2
        * np.sum((arm_joints - q_start[None, :]) ** 2)
    )

    if len(arm_joints) >= 2:
        joint_steps = np.diff(arm_joints, axis=0)
        smoothness_cost = float(
            float(config.joint_smoothness_scale) ** 2
            * np.sum(joint_steps**2)
        )
        joint_path_length = float(
            np.sum(np.linalg.norm(joint_steps, axis=1))
        )
    else:
        smoothness_cost = 0.0
        joint_path_length = 0.0

    acceleration = np.asarray(
        result.metrics.joint_acceleration_rad_s2,
        dtype=np.float64,
    )
    if acceleration.size:
        arm_acceleration = acceleration[:, : int(config.panda_arm_dof)]
        maximum_acceleration = float(np.max(np.abs(arm_acceleration)))
        rms_acceleration = float(np.sqrt(np.mean(arm_acceleration**2)))
    else:
        maximum_acceleration = 0.0
        rms_acceleration = 0.0

    # This comparable score contains terms that can be reconstructed exactly
    # from the saved result. The five-point acceleration term and optional
    # self-collision soft cost are reported separately rather than approximated.
    symmetry_rest_cost = float(
        float(config.phase_symmetry_rest_scale) ** 2
        * np.sum(
            result.phase_symmetry_angles_rad[
                result.phase_symmetry_active_phase_mask
            ]
            ** 2
        )
    )
    comparable_objective = (
        tracking_cost
        + rest_cost
        + smoothness_cost
        + symmetry_rest_cost
    )

    return {
        "tracking_cost": tracking_cost,
        "rest_cost": rest_cost,
        "smoothness_cost": smoothness_cost,
        "phase_symmetry_rest_cost": symmetry_rest_cost,
        "comparable_objective": comparable_objective,
        "joint_path_length_rad": joint_path_length,
        "max_joint_acceleration_rad_s2": maximum_acceleration,
        "rms_joint_acceleration_rad_s2": rms_acceleration,
        "max_position_error_m": _finite_max(
            result.metrics.position_errors_m
        ),
        "max_axis_error_deg": _finite_max(
            result.metrics.axis_errors_deg
        ),
        "max_full_rotation_error_deg": _finite_max(
            result.metrics.full_rotation_errors_deg
        ),
        "max_remaining_rotation_error_deg": _finite_max(
            result.metrics.remaining_rotation_errors_deg
        ),
    }


def _select_and_save_best_candidate(
    records: list[dict[str, Any]],
    successful_results: Mapping[int, TrajectoryOptimizationResult],
    reference: MotionReference,
    *,
    output_root: Path,
    result_output_dir: Optional[str | Path],
    requested_thresholds: Sequence[Optional[float]],
) -> tuple[TrajectoryOptimizationResult, int, list[dict[str, Any]]]:
    """Rank completed candidates and save the selected result and comparison."""
    successful_records = [
        record
        for record in records
        if record.get("optimization_success", False)
    ]
    if not successful_records:
        comparison_path = output_root / "candidate_comparison.json"
        comparison_path.write_text(
            json.dumps(records, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(
            "All multi-start trajectory optimizations failed. "
            f"See {comparison_path}."
        )

    feasible_records = [
        record
        for record in successful_records
        if record.get("task_feasible", False)
    ]
    ranking_pool = feasible_records or successful_records

    if not feasible_records and any(
        threshold is not None
        for threshold in requested_thresholds
    ):
        warnings.warn(
            "No trajectory passed every requested task-error threshold. "
            "Selecting the best successful trajectory instead."
        )

    def ranking_key(record: Mapping[str, Any]) -> tuple[float, ...]:
        grasp_score = record.get("graspgen_score")
        score_tiebreak = (
            0.0
            if grasp_score is None
            else -float(grasp_score)
        )
        return (
            float(record["comparable_objective"]),
            float(record["tracking_cost"]),
            float(record["max_position_error_m"]),
            float(record["max_axis_error_deg"]),
            float(record["max_full_rotation_error_deg"]),
            float(record["max_remaining_rotation_error_deg"]),
            float(record["joint_path_length_rad"]),
            float(record["max_joint_acceleration_rad_s2"]),
            score_tiebreak,
        )

    best_record = min(ranking_pool, key=ranking_key)
    best_index = int(best_record["candidate_index"])
    best_result = successful_results[best_index]
    result_dir = output_root / "best" if result_output_dir is None else Path(result_output_dir)
    best_record["output_directory"] = str(result_dir)

    for record in records:
        record["selected_best"] = (
            int(record["candidate_index"]) == best_index
        )

    comparison_path = output_root / "candidate_comparison.json"
    comparison_path.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    save_optimization_result(
        result_dir,
        best_result,
        reference=reference,
    )

    print(
        f"[Optimize] Selected candidate {best_index + 1}/{len(records)}: "
        f"cost={best_record['comparable_objective']:.4g}; results: {result_dir}",
        flush=True,
    )

    return best_result, best_index, records


def run_multi_start_trajectory_optimization(
    reference: MotionReference,
    q_start_candidates: ArrayLike,
    *,
    output_dir: str | Path,
    result_output_dir: Optional[str | Path] = None,
    urdf_description: str = "panda_description",
    target_link: str = "panda_hand_tcp",
    config: Optional[TrajectoryOptimizerConfig] = None,
    sphere_json: Optional[Path] = None,
    graspgen_scores: Optional[ArrayLike] = None,
    expected_hand_poses_world: Optional[ArrayLike] = None,
    max_position_error_m: Optional[float] = None,
    max_axis_error_deg: Optional[float] = None,
    max_rotation_error_deg: Optional[float] = None,
    max_grasp_pose_position_error_m: Optional[float] = None,
    max_grasp_pose_rotation_error_deg: Optional[float] = None,
    clear_jax_caches_between_candidates: bool = True,
) -> tuple[TrajectoryOptimizationResult, int, list[dict[str, Any]]]:
    """Optimize a complete trajectory for each grasp candidate.

    If ``expected_hand_poses_world`` is supplied, each pose is authoritative and
    the matching ``q_start_candidates`` row is only a warm-start seed.  The
    optimizer jointly solves q[0:T] while constraining FK(q[0]) to that 6D pose.
    Without expected poses this function preserves the legacy fixed-q behavior.

    Candidate scores, validation results, and failures are saved in
    output_dir/candidate_comparison.json. The selected full result is saved in
    result_output_dir (default: output_dir/best/). Its record points there;
    other records have no saved trajectory directory.

    Selection:
      1. Optional hard task-error thresholds.
      2. Lowest reconstructed comparable objective.
      3. Lowest maximum tracking errors.
      4. Shorter joint path.
      5. Lower maximum acceleration.
      6. Higher GraspGen score as the final tie-breaker.

    The q7 +/-90-degree condition is not checked over the full trajectory here.
    It should already have been applied to the initial grasp IK candidates.
    """
    candidates = np.asarray(q_start_candidates, dtype=np.float64)
    if candidates.ndim != 2:
        raise ValueError(
            "q_start_candidates must have shape (K, dof), "
            f"got {candidates.shape}."
        )
    if len(candidates) == 0:
        raise ValueError("q_start_candidates is empty.")
    if not np.isfinite(candidates).all():
        raise ValueError("q_start_candidates contains NaN or infinity.")

    cfg = config or TrajectoryOptimizerConfig()
    # Validate global reference semantics once before entering the candidate loop.
    # This avoids repeating the same reference-level failure for every q start.
    phase_symmetry_active_phase_mask(reference)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if graspgen_scores is None:
        scores = np.full(len(candidates), np.nan, dtype=np.float64)
    else:
        scores = np.asarray(graspgen_scores, dtype=np.float64).reshape(-1)
        if len(scores) != len(candidates):
            raise ValueError(
                "graspgen_scores length does not match q_start_candidates: "
                f"{len(scores)} versus {len(candidates)}."
            )

    expected_hand_poses: Optional[FloatArray]
    if expected_hand_poses_world is None:
        expected_hand_poses = None
    else:
        expected_hand_poses = np.asarray(
            expected_hand_poses_world,
            dtype=np.float64,
        )
        if expected_hand_poses.shape != (len(candidates), 4, 4):
            raise ValueError(
                "expected_hand_poses_world must have shape (K, 4, 4) matching "
                "q_start_candidates, got "
                f"{expected_hand_poses.shape} for K={len(candidates)}."
            )
        if not np.isfinite(expected_hand_poses).all():
            raise ValueError(
                "expected_hand_poses_world contains NaN or infinity."
            )

    records: list[dict[str, Any]] = []
    successful_results: dict[int, TrajectoryOptimizationResult] = {}

    print(f"[Optimize] Optimizing {len(candidates)} candidates...", flush=True)

    for candidate_index, q_start in enumerate(candidates):
        candidate_info = {
            "candidate_index": candidate_index,
            "q_start": q_start.tolist(),
            "graspgen_score": (
                float(scores[candidate_index])
                if np.isfinite(scores[candidate_index]) else None
            ),
        }
        try:
            desired_start_hand_pose = (
                None
                if expected_hand_poses is None
                else expected_hand_poses[candidate_index]
            )
            result = run_trajectory_optimization(
                reference=reference,
                q_start_given=q_start,
                urdf_description=urdf_description,
                target_link=target_link,
                config=cfg,
                sphere_json=sphere_json,
                desired_start_hand_pose_world=desired_start_hand_pose,
            )

            evaluation = compute_tracking_selection_cost(
                reference,
                result,
            )

            grasp_pose_position_error_m: Optional[float] = None
            grasp_pose_rotation_error_deg: Optional[float] = None
            if expected_hand_poses is not None:
                expected_hand_pose = expected_hand_poses[candidate_index]
                achieved_start_hand_pose = np.asarray(
                    result.achieved_hand_poses_world[0],
                    dtype=np.float64,
                )
                grasp_pose_position_error_m = float(
                    np.linalg.norm(
                        achieved_start_hand_pose[:3, 3]
                        - expected_hand_pose[:3, 3]
                    )
                )
                grasp_pose_rotation_error_deg = rotation_error_deg(
                    achieved_start_hand_pose[:3, :3],
                    expected_hand_pose[:3, :3],
                )

            maximum_rotation_error = max(
                evaluation["max_full_rotation_error_deg"],
                evaluation["max_remaining_rotation_error_deg"],
            )
            thresholds = (
                ("max_grasp_pose_position_error_m", grasp_pose_position_error_m, max_grasp_pose_position_error_m),
                ("max_grasp_pose_rotation_error_deg", grasp_pose_rotation_error_deg, max_grasp_pose_rotation_error_deg),
                ("max_position_error_m", evaluation["max_position_error_m"], max_position_error_m),
                ("max_axis_error_deg", evaluation["max_axis_error_deg"], max_axis_error_deg),
                ("max_rotation_error_deg", maximum_rotation_error, max_rotation_error_deg),
            )
            rejection_reasons = [
                name for name, error, limit in thresholds
                if error is not None and limit is not None and error > float(limit)
            ]
            feasible = not rejection_reasons

            record: dict[str, Any] = {
                **candidate_info,
                "optimization_success": True,
                "task_feasible": feasible,
                "rejection_reasons": rejection_reasons,
                "grasp_pose_position_error_m": grasp_pose_position_error_m,
                "grasp_pose_rotation_error_deg": grasp_pose_rotation_error_deg,
                "expected_hand_pose_world": (
                    None
                    if expected_hand_poses is None
                    else expected_hand_poses[candidate_index].tolist()
                ),
                "output_directory": None,
                **evaluation,
            }

            successful_results[candidate_index] = result

            print(
                f"[Optimize] Candidate {candidate_index + 1}/{len(candidates)}: "
                f"cost={evaluation['comparable_objective']:.4g}, "
                f"max position={1000 * evaluation['max_position_error_m']:.2f} mm, "
                f"max axis={evaluation['max_axis_error_deg']:.2f} deg, "
                f"feasible={feasible}",
                flush=True,
            )

        except Exception as error:
            record = {
                **candidate_info,
                "optimization_success": False,
                "task_feasible": False,
                "error": repr(error),
            }
            print(
                f"[Optimize] Candidate {candidate_index + 1}/{len(candidates)} "
                f"failed: {error!r}",
                flush=True,
            )

        records.append(record)

        if clear_jax_caches_between_candidates:
            # The result contains only NumPy arrays, so clearing compilation
            # caches here does not invalidate an already completed candidate.
            try:
                import gc
                import jax

                jax.clear_caches()
                gc.collect()
            except Exception:
                pass

    return _select_and_save_best_candidate(
        records,
        successful_results,
        reference,
        output_root=output_root,
        result_output_dir=result_output_dir,
        requested_thresholds=(
            max_position_error_m,
            max_axis_error_deg,
            max_rotation_error_deg,
            max_grasp_pose_position_error_m,
            max_grasp_pose_rotation_error_deg,
        ),
    )


# ---------------------------------------------------------------------------
# Serialization and summaries
# ---------------------------------------------------------------------------


def _nan_stat(values: FloatArray, function: Any) -> Optional[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(function(finite))


def result_summary(result: TrajectoryOptimizationResult) -> dict[str, Any]:
    velocity = result.metrics.joint_velocity_rad_s
    acceleration = result.metrics.joint_acceleration_rad_s2
    return {
        "samples": int(result.planned_joints.shape[0]),
        "dof": int(result.planned_joints.shape[1]),
        "target_link_name": result.target_link_name,
        "urdf_description": result.urdf_description,
        "phase_symmetry": {
            "active_phase_mask": (
                result.phase_symmetry_active_phase_mask.tolist()
            ),
            "angles_rad": result.phase_symmetry_angles_rad.tolist(),
            "angles_deg": np.degrees(
                result.phase_symmetry_angles_rad
            ).tolist(),
        },
        "position_error_m": {
            "mean": _nan_stat(result.metrics.position_errors_m, np.mean),
            "max": _nan_stat(result.metrics.position_errors_m, np.max),
        },
        "axis_error_deg": {
            "mean": _nan_stat(result.metrics.axis_errors_deg, np.mean),
            "max": _nan_stat(result.metrics.axis_errors_deg, np.max),
        },
        "full_rotation_error_deg": {
            "mean": _nan_stat(result.metrics.full_rotation_errors_deg, np.mean),
            "max": _nan_stat(result.metrics.full_rotation_errors_deg, np.max),
        },
        "remaining_rotation_error_deg": {
            "mean": _nan_stat(result.metrics.remaining_rotation_errors_deg, np.mean),
            "max": _nan_stat(result.metrics.remaining_rotation_errors_deg, np.max),
        },
        "joint_velocity_rad_s": {
            "max_abs": None if velocity.size == 0 else float(np.max(np.abs(velocity))),
        },
        "joint_acceleration_rad_s2": {
            "max_abs": (
                None if acceleration.size == 0 else float(np.max(np.abs(acceleration)))
            ),
        },
        "config": asdict(result.config),
    }


# Ordered NPZ fields shared by result saving and loading.
_OPTIMIZATION_ARRAY_FIELDS = (
    "planned_joints",
    "achieved_hand_poses_world",
    "achieved_grasp_poses_world",
    "achieved_effect_poses_world",
    "transform_hand_grasp",
    "transform_grasp_effect",
)
_TRACKING_METRIC_FIELDS = (
    "position_errors_m",
    "axis_errors_deg",
    "full_rotation_errors_deg",
    "remaining_rotation_errors_deg",
    "joint_velocity_rad_s",
    "joint_acceleration_rad_s2",
)


def save_optimization_result(
    output_dir: str | Path,
    result: TrajectoryOptimizationResult,
    *,
    reference: Optional[MotionReference] = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output / "optimization_result.npz",
        **{name: getattr(result, name) for name in _OPTIMIZATION_ARRAY_FIELDS},
        phase_symmetry_angles_rad=result.phase_symmetry_angles_rad,
        phase_symmetry_active_phase_mask=(
            result.phase_symmetry_active_phase_mask
        ),
        **{name: getattr(result.metrics, name) for name in _TRACKING_METRIC_FIELDS},
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format_version": 2,
                    "config": asdict(result.config),
                    "target_link_name": result.target_link_name,
                    "urdf_description": result.urdf_description,
                }
            )
        ),
    )

    if reference is not None:
        _, _, _, desired_effect_poses, _ = apply_phase_symmetry_to_reference(
            reference,
            result.phase_symmetry_angles_rad,
        )
        np.save(
            output / "desired_effect_poses_world.npy",
            desired_effect_poses,
        )

    (output / "optimization_metrics.json").write_text(
        json.dumps(result_summary(result), indent=2),
        encoding="utf-8",
    )


def load_optimization_result(path: str | Path) -> TrajectoryOptimizationResult:
    input_path = Path(path)
    bundle_path = (
        input_path / "optimization_result.npz"
        if input_path.is_dir()
        else input_path
    )
    with np.load(bundle_path, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata_json"].item()))
        format_version = int(metadata.get("format_version", -1))
        if format_version not in (1, 2):
            raise ValueError(
                f"Unsupported optimization-result format: "
                f"{metadata.get('format_version')}."
            )
        config = TrajectoryOptimizerConfig(**metadata["config"])
        metrics = TrajectoryTrackingMetrics(**{
            name: np.asarray(bundle[name], dtype=np.float64)
            for name in _TRACKING_METRIC_FIELDS
        })
        return TrajectoryOptimizationResult(
            **{
                name: np.asarray(bundle[name], dtype=np.float64)
                for name in _OPTIMIZATION_ARRAY_FIELDS
            },
            phase_symmetry_angles_rad=(
                np.asarray(
                    bundle["phase_symmetry_angles_rad"],
                    dtype=np.float64,
                )
                if format_version >= 2
                else np.empty(0, dtype=np.float64)
            ),
            phase_symmetry_active_phase_mask=(
                np.asarray(
                    bundle["phase_symmetry_active_phase_mask"],
                    dtype=bool,
                )
                if format_version >= 2
                else np.empty(0, dtype=bool)
            ),
            metrics=metrics,
            config=config,
            target_link_name=str(metadata["target_link_name"]),
            urdf_description=str(metadata["urdf_description"]),
        )


__all__ = [
    "AXIS_RELATION_OPPOSITE_DIRECTION",
    "AXIS_RELATION_PARALLEL_EITHER_DIRECTION",
    "AXIS_RELATION_SAME_DIRECTION",
    "TrajectoryOptimizationResult",
    "TrajectoryOptimizerConfig",
    "TrajectoryTrackingMetrics",
    "adapt_start_configuration",
    "apply_phase_symmetry_to_reference",
    "axis_relation_error_deg",
    "compose_control_poses",
    "compute_hand_grasp_attachment",
    "compute_joint_dynamics",
    "compute_tracking_metrics",
    "compute_tracking_selection_cost",
    "fk_matrix",
    "fk_trajectory",
    "load_optimization_result",
    "make_initial_trajectory",
    "optimize_joint_trajectory",
    "phase_symmetry_active_phase_mask",
    "resolve_target_link",
    "result_summary",
    "rotation_error_deg",
    "run_multi_start_trajectory_optimization",
    "run_trajectory_optimization",
    "save_optimization_result",
    "solve_reference_ik",
    "validate_reference_for_optimizer",
]
