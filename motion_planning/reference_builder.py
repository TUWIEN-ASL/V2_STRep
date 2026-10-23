from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .demo_trajectory import PreparedDemoTrajectory, prepare_demo_trajectory
from .scene_geometry import GeometricFeature, SceneGeometry
from .schemas import (
    AxisName,
    AxisRelation,
    MotionSource,
    PhaseSpec,
    PhaseSymmetryMode,
    ReferenceFrame,
    RemainingRotation,
    RotationMode,
    TargetRepresentation,
    VideoStructureSpec,
)
from .se3 import (
    as_transform,
    compose_transform,
    interpolate_pose,
    invert_transform,
    make_transform,
    relative_transform,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

MAX_AXIS_CONSTRAINTS = 2

_AXIS_TO_INDEX = {
    AxisName.X: 0,
    AxisName.Y: 1,
    AxisName.Z: 2,
}
_INDEX_TO_AXIS = {value: key for key, value in _AXIS_TO_INDEX.items()}

_REFERENCE_FRAME_TO_INDEX = {
    ReferenceFrame.GRASP: 0,
    ReferenceFrame.TARGET: 1,
}

_RELATION_TO_INDEX = {
    AxisRelation.SAME_DIRECTION: 0,
    AxisRelation.OPPOSITE_DIRECTION: 1,
    AxisRelation.PARALLEL_EITHER_DIRECTION: 2,
}
_INDEX_TO_RELATION = {
    value: key for key, value in _RELATION_TO_INDEX.items()
}

_ROTATION_MODE_TO_INDEX = {
    RotationMode.FREE: 0,
    RotationMode.AXIS_CONSTRAINED: 1,
    RotationMode.FULL_DEMO: 2,
}

_REMAINING_ROTATION_TO_INDEX = {
    RemainingRotation.FREE: 0,
    RemainingRotation.FIXED_FROM_DEMO: 1,
    RemainingRotation.NONE: 2,
}


# Shapes after the leading sample dimension, in validation and save order.
_REFERENCE_SAMPLE_SHAPES = {
    "position_targets_world": (3,),
    "position_mask": (),
    "demo_relative_transform": (4, 4),
    "demo_relative_valid_mask": (),
    "demo_relative_reference_frame_index": (),
    "rotation_mode_index": (),
    "remaining_rotation_index": (),
    "full_rotation_targets_world": (3, 3),
    "full_rotation_mask": (),
    "remaining_rotation_fixed_mask": (),
    "axis_constraint_mask": (MAX_AXIS_CONSTRAINTS,),
    "axis_controlled_axis_index": (MAX_AXIS_CONSTRAINTS,),
    "axis_reference_frame_index": (MAX_AXIS_CONSTRAINTS,),
    "axis_reference_axis_index": (MAX_AXIS_CONSTRAINTS,),
    "axis_relation_index": (MAX_AXIS_CONSTRAINTS,),
    "axis_reference_directions_world": (MAX_AXIS_CONSTRAINTS, 3),
    "phase_symmetry_mask": (),
    "phase_symmetry_axis_world": (3,),
    "phase_index": (),
    "source_frame_indices": (),
}


@dataclass(frozen=True)
class PhaseReferenceRange:
    """One contiguous phase range in the concatenated motion reference.

    `sample_stop` is Python-exclusive. `cycle_sample_start` and
    `cycle_sample_stop` are also global Python-style indices when present.
    """

    phase_id: str
    name: str
    sample_start: int
    sample_stop: int
    source_frame_start: int
    source_frame_end: int
    motion_source: MotionSource
    reference_frame: Optional[ReferenceFrame]
    attractor_goal_phase_id: Optional[str]
    rotation_mode: RotationMode
    remaining_rotation: RemainingRotation
    repeatable: bool
    cycle_sample_start: Optional[int]
    cycle_sample_stop: Optional[int]

    @property
    def sample_count(self) -> int:
        return self.sample_stop - self.sample_start


@dataclass(frozen=True)
class ReferenceBuildDiagnostics:
    """Diagnostics for cross-environment semantic transfer.

    These values check consistency of the two scene descriptions and
    the demonstrated canonical grasp trajectory. They do not evaluate robot
    reachability, collision, joint limits, or timing; those belong to optimization.
    """

    current_grasp_was_inferred: bool
    demo_initial_grasp_position_error_m: float
    demo_initial_grasp_rotation_error_rad: float
    phase_entry_position_jumps_m: FloatArray
    phase_entry_rotation_jumps_rad: FloatArray

    def __post_init__(self) -> None:
        if not np.isfinite(self.demo_initial_grasp_position_error_m):
            raise ValueError("demo_initial_grasp_position_error_m must be finite.")
        if not np.isfinite(self.demo_initial_grasp_rotation_error_rad):
            raise ValueError("demo_initial_grasp_rotation_error_rad must be finite.")

        position = np.asarray(self.phase_entry_position_jumps_m, dtype=np.float64)
        rotation = np.asarray(self.phase_entry_rotation_jumps_rad, dtype=np.float64)
        if position.ndim != 1 or rotation.ndim != 1:
            raise ValueError("Phase-entry diagnostics must be one-dimensional arrays.")
        if position.shape != rotation.shape:
            raise ValueError("Position and rotation jump arrays must have equal shape.")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
            raise ValueError("Phase-entry diagnostics contain non-finite values.")
        object.__setattr__(self, "phase_entry_position_jumps_m", position.copy())
        object.__setattr__(self, "phase_entry_rotation_jumps_rad", rotation.copy())


@dataclass(frozen=True)
class MotionReference:
    """Task-space reference consumed by the trajectory optimizer.

    The controlled frame is the effect frame. Position targets are its origins.
    Axis constraints compare an axis of the achieved effect frame against the
    packaged world direction.

    `transform_world_grasp_gauge` and `transform_world_effect_gauge` are full
    deterministic poses for visualization and warm starts. Their rotations are
    not automatically full rotational constraints. The masks and rotation-mode
    arrays decide what the optimizer should enforce.
    """

    transform_world_grasp_gauge: FloatArray
    transform_world_effect_gauge: FloatArray
    position_targets_world: FloatArray
    position_mask: BoolArray

    # The learned demonstration trajectory before current-scene instantiation.
    # For grasp-relative learned phases this is T_G0_G(t). For target-relative
    # learned phases this is T_Tdemo_Edemo(t). Attractor samples are marked
    # invalid because they are generated from the current scene, not copied
    # from the demonstration.
    demo_relative_transform: FloatArray
    demo_relative_valid_mask: BoolArray
    demo_relative_reference_frame_index: IntArray

    rotation_mode_index: IntArray
    remaining_rotation_index: IntArray
    full_rotation_targets_world: FloatArray
    full_rotation_mask: BoolArray
    remaining_rotation_fixed_mask: BoolArray

    axis_constraint_mask: BoolArray
    axis_controlled_axis_index: IntArray
    axis_reference_frame_index: IntArray
    axis_reference_axis_index: IntArray
    axis_relation_index: IntArray
    axis_reference_directions_world: FloatArray

    phase_symmetry_mask: BoolArray
    phase_symmetry_axis_world: FloatArray

    phase_index: IntArray
    source_frame_indices: IntArray
    phases: Tuple[PhaseReferenceRange, ...]

    transform_grasp_effect: FloatArray
    transform_grasp_effect_demo: FloatArray
    transform_world_target_current: FloatArray
    transform_world_target_demo: FloatArray
    target_representation: TargetRepresentation
    controlled_frame: str
    diagnostics: ReferenceBuildDiagnostics

    def __post_init__(self) -> None:
        grasp = np.asarray(self.transform_world_grasp_gauge, dtype=np.float64)
        effect = np.asarray(self.transform_world_effect_gauge, dtype=np.float64)
        if grasp.ndim != 3 or grasp.shape[1:] != (4, 4):
            raise ValueError(
                "transform_world_grasp_gauge must have shape (N, 4, 4), "
                f"got {grasp.shape}."
            )
        if effect.shape != grasp.shape:
            raise ValueError("Grasp and effect gauge trajectories must have equal shape.")
        sample_count = grasp.shape[0]
        if sample_count < 1:
            raise ValueError("MotionReference must contain at least one sample.")

        for name, trailing_shape in _REFERENCE_SAMPLE_SHAPES.items():
            shape = (sample_count, *trailing_shape)
            array = np.asarray(getattr(self, name))
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")

        for pose in grasp:
            as_transform(pose, copy=False)
        for pose in effect:
            as_transform(pose, copy=False)

        demo_relative = np.asarray(self.demo_relative_transform, dtype=np.float64)
        demo_relative_valid = np.asarray(self.demo_relative_valid_mask, dtype=bool)
        for pose in demo_relative[demo_relative_valid]:
            as_transform(pose, copy=False)

        transform_grasp_effect = as_transform(self.transform_grasp_effect)
        transform_grasp_effect_demo = as_transform(self.transform_grasp_effect_demo)
        target_current = as_transform(self.transform_world_target_current)
        target_demo = as_transform(self.transform_world_target_demo)

        reconstructed_effect = np.einsum(
            "nij,jk->nik", grasp, transform_grasp_effect
        )
        if not np.allclose(reconstructed_effect, effect, atol=1e-7):
            raise ValueError(
                "Effect gauge trajectory is inconsistent with "
                "T_world_grasp @ T_grasp_effect."
            )

        if not np.all(np.asarray(self.position_mask, dtype=bool)):
            # The first implementation tracks an effect position for every sample.
            raise ValueError("Step 7 currently expects position_mask to be true everywhere.")

        expected_phase_start = 0
        for phase_number, phase in enumerate(self.phases):
            if phase.sample_start != expected_phase_start:
                raise ValueError("Phase ranges must be contiguous and ordered.")
            if phase.sample_stop <= phase.sample_start:
                raise ValueError(f"Phase {phase.phase_id!r} has an empty range.")
            sample_slice = slice(phase.sample_start, phase.sample_stop)
            if not np.all(np.asarray(self.phase_index)[sample_slice] == phase_number):
                raise ValueError("phase_index is inconsistent with phase ranges.")
            expected_phase_start = phase.sample_stop
        if expected_phase_start != sample_count:
            raise ValueError("Phase ranges do not cover the complete reference.")

        relative_reference = np.asarray(
            self.demo_relative_reference_frame_index, dtype=np.int64
        )
        relative_valid = np.asarray(self.demo_relative_valid_mask, dtype=bool)
        if np.any(relative_valid & ~np.isin(relative_reference, [0, 1])):
            raise ValueError(
                "Valid demo-relative samples must use grasp(0) or target(1)."
            )
        if np.any((~relative_valid) & (relative_reference != -1)):
            raise ValueError(
                "Generated/non-demo samples must use relative reference index -1."
            )

        directions = np.asarray(self.axis_reference_directions_world, dtype=np.float64)
        axis_mask = np.asarray(self.axis_constraint_mask, dtype=bool)
        if np.any(axis_mask):
            active_directions = directions[axis_mask]
            norms = np.linalg.norm(active_directions, axis=-1)
            if not np.all(np.isfinite(active_directions)) or not np.allclose(
                norms, 1.0, atol=1e-6
            ):
                raise ValueError("Active axis-reference directions must be finite unit vectors.")

        object.__setattr__(self, "transform_world_grasp_gauge", grasp.copy())
        object.__setattr__(self, "transform_world_effect_gauge", effect.copy())
        object.__setattr__(self, "transform_grasp_effect", transform_grasp_effect)
        object.__setattr__(self, "transform_grasp_effect_demo", transform_grasp_effect_demo)
        object.__setattr__(self, "transform_world_target_current", target_current)
        object.__setattr__(self, "transform_world_target_demo", target_demo)

        for field_name in _REFERENCE_SAMPLE_SHAPES:
            object.__setattr__(self, field_name, np.asarray(getattr(self, field_name)).copy())

    @property
    def sample_count(self) -> int:
        return int(self.transform_world_grasp_gauge.shape[0])

    def phase_by_id(self, phase_id: str) -> PhaseReferenceRange:
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        raise KeyError(f"Unknown phase_id: {phase_id}")

    def axis_name(self, sample_index: int, slot: int) -> AxisName:
        return _INDEX_TO_AXIS[int(self.axis_controlled_axis_index[sample_index, slot])]

    def axis_relation(self, sample_index: int, slot: int) -> AxisRelation:
        return _INDEX_TO_RELATION[int(self.axis_relation_index[sample_index, slot])]


@dataclass(frozen=True)
class _BuiltPhase:
    phase: PhaseSpec
    transform_world_grasp: FloatArray
    transform_world_effect: FloatArray
    source_frame_indices: IntArray
    demo_relative_transform: FloatArray
    demo_relative_valid_mask: BoolArray


def _normalize(vector: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    norm = float(np.linalg.norm(array))
    if norm < 1e-12:
        raise ValueError(f"{name} has near-zero norm.")
    return array / norm


def _rotation_error_angle(rotation_a: FloatArray, rotation_b: FloatArray) -> float:
    relative = rotation_a.T @ rotation_b
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def _smoothstep(value: float) -> float:
    return 3.0 * value * value - 2.0 * value * value * value


def _minimal_rotation_between_vectors(
    vector_from: ArrayLike,
    vector_to: ArrayLike,
) -> FloatArray:
    """Return the minimum-angle rotation mapping vector_from onto vector_to.

    The returned rotation contains no additional twist about vector_to.
    """
    a = _normalize(vector_from, name="vector_from")
    b = _normalize(vector_to, name="vector_to")

    cross = np.cross(a, b)
    sin_angle = float(np.linalg.norm(cross))
    cos_angle = float(np.clip(np.dot(a, b), -1.0, 1.0))

    # Same direction.
    if sin_angle < 1e-10 and cos_angle > 0.0:
        return np.eye(3, dtype=np.float64)

    # Opposite direction. The angle is pi, but the rotation axis is not unique.
    # Choose one deterministic axis perpendicular to the source direction.
    if sin_angle < 1e-10 and cos_angle < 0.0:
        candidate = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(candidate, a))) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        axis = _normalize(
            np.cross(a, candidate),
            name="180-degree rotation axis",
        )
        return (
            -np.eye(3, dtype=np.float64)
            + 2.0 * np.outer(axis, axis)
        )

    # Rodrigues formula.
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )

    return (
        np.eye(3, dtype=np.float64)
        + skew
        + skew @ skew
        * ((1.0 - cos_angle) / (sin_angle * sin_angle))
    )



def _rotation_about_axis(axis: ArrayLike, angle_rad: float) -> FloatArray:
    """Return a 3x3 rotation by ``angle_rad`` around ``axis``.

    ``axis`` is expressed in the frame in which the returned rotation acts.
    """
    unit_axis = _normalize(axis, name="rotation axis")
    x, y, z = unit_axis
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    one_minus_cosine = 1.0 - cosine

    return np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=np.float64,
    )


def _axis_transfer_rotation(
    demo_axis_world: ArrayLike,
    current_axis_world: ArrayLike,
    demo_target_origin_world: ArrayLike,
    current_target_origin_world: ArrayLike,
    demo_controlled_origin_world: ArrayLike,
    current_controlled_origin_world: ArrayLike,
) -> FloatArray:
    """Resolve the complete cross-scene rotation for an AXIS target.

    An axis target defines only an origin and a z direction, so aligning target
    z alone leaves one unknown rotation about z. Resolve that missing gauge from
    the controlled object's INITIAL position around the target axis:

        1. minimally rotate demo target z onto current target z;
        2. project the demo/current target-to-controlled vectors into the plane
           perpendicular to the current target z;
        3. rotate about current target z so the projected demo radial direction
           matches the current radial direction.

    The returned rotation is one rigid rotation for the whole transferred
    target-relative phase. If either radial direction is degenerate (controlled
    origin lies on/very near the axis), fall back to minimum z alignment.
    """
    demo_axis = _normalize(demo_axis_world, name="demo target axis")
    current_axis = _normalize(current_axis_world, name="current target axis")

    rotation_axis_align = _minimal_rotation_between_vectors(
        demo_axis,
        current_axis,
    )

    demo_target_origin = np.asarray(
        demo_target_origin_world,
        dtype=np.float64,
    ).reshape(3)
    current_target_origin = np.asarray(
        current_target_origin_world,
        dtype=np.float64,
    ).reshape(3)
    demo_controlled_origin = np.asarray(
        demo_controlled_origin_world,
        dtype=np.float64,
    ).reshape(3)
    current_controlled_origin = np.asarray(
        current_controlled_origin_world,
        dtype=np.float64,
    ).reshape(3)

    demo_radial_world = demo_controlled_origin - demo_target_origin
    current_radial_world = current_controlled_origin - current_target_origin

    # First put the demonstrated radial vector into the current axis frame.
    demo_radial_aligned = rotation_axis_align @ demo_radial_world

    # Only the component perpendicular to the axis resolves rotation about z.
    demo_radial_perp = (
        demo_radial_aligned
        - current_axis * float(np.dot(demo_radial_aligned, current_axis))
    )
    current_radial_perp = (
        current_radial_world
        - current_axis * float(np.dot(current_radial_world, current_axis))
    )

    demo_norm = float(np.linalg.norm(demo_radial_perp))
    current_norm = float(np.linalg.norm(current_radial_perp))
    radial_epsilon_m = 1e-6
    if demo_norm < radial_epsilon_m or current_norm < radial_epsilon_m:
        return rotation_axis_align

    demo_radial_unit = demo_radial_perp / demo_norm
    current_radial_unit = current_radial_perp / current_norm

    cosine = float(
        np.clip(np.dot(demo_radial_unit, current_radial_unit), -1.0, 1.0)
    )
    sine = float(
        np.dot(
            current_axis,
            np.cross(demo_radial_unit, current_radial_unit),
        )
    )
    twist_angle = float(np.arctan2(sine, cosine))

    rotation_radial_align = _rotation_about_axis(current_axis, twist_angle)
    return rotation_radial_align @ rotation_axis_align


def _target_transfer_frame(feature: GeometricFeature) -> FloatArray:
    """Return the deterministic geometric frame stored for a target.

    POINT uses world-aligned axes because it has no semantic orientation.

    Other target representations may contain computational gauge axes in
    transform_world_gauge. These axes are useful for representing local
    geometry, but cross-scene trajectory transfer must respect
    TargetRepresentation semantics and must not assume every gauge axis is
    physically meaningful. See _target_relative_current_frame().
    """
    if feature.transform_world_gauge is not None:
        return as_transform(feature.transform_world_gauge)
    if feature.point_world is None:
        raise ValueError("Target geometry has neither a gauge frame nor a point.")
    return make_transform(np.eye(3, dtype=np.float64), feature.point_world)


def _target_relative_current_frame(
    transform_world_target_demo: ArrayLike,
    transform_world_target_current: ArrayLike,
    target_representation: TargetRepresentation,
    *,
    controlled_origin_demo_world: Optional[ArrayLike] = None,
    controlled_origin_current_world: Optional[ArrayLike] = None,
) -> FloatArray:
    """Build the current target frame used to instantiate demo-relative motion.

    Transfer only the target degrees of freedom that are semantically defined:

    - POINT: translation only.
    - POINT_NORMAL / PLANE: translation plus the minimum rotation that aligns
      demo target z with current target z. The independently generated current
      x/y gauge is ignored.
    - AXIS: translation + target-z alignment, then resolve the otherwise
      undefined rotation about target z from the demo/current INITIAL
      controlled-object position around the target axis. This makes a rigidly
      reoriented articulated object (for example, a door around a hinge) rotate
      the whole learned target-relative trajectory by the same scene rotation.
    - OBJECT_POSE_6D: full current target pose.

    For z-only surface representations, the current transfer frame reuses the
    demo x/y gauge after applying minimum z alignment. For AXIS, it additionally
    rotates that gauge about current z using the controlled-object radial
    direction. In every case the resulting frame is composed with the stored
    demo-relative trajectory; the target's arbitrary reconstruction gauge never
    directly becomes a task constraint.
    """
    demo_target = as_transform(transform_world_target_demo)
    current_target = as_transform(transform_world_target_current)

    if target_representation is TargetRepresentation.POINT:
        result = np.eye(4, dtype=np.float64)
        result[:3, 3] = current_target[:3, 3]
        return result

    if target_representation in {
        TargetRepresentation.POINT_NORMAL,
        TargetRepresentation.PLANE,
    }:
        demo_rotation = demo_target[:3, :3]
        demo_z = demo_rotation[:, 2]
        current_z = current_target[:3, 2]

        rotation_demo_to_current = _minimal_rotation_between_vectors(
            demo_z,
            current_z,
        )

        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rotation_demo_to_current @ demo_rotation
        result[:3, 3] = current_target[:3, 3]
        return result

    if target_representation is TargetRepresentation.AXIS:
        demo_rotation = demo_target[:3, :3]
        demo_z = demo_rotation[:, 2]
        current_z = current_target[:3, 2]

        if (
            controlled_origin_demo_world is not None
            and controlled_origin_current_world is not None
        ):
            rotation_demo_to_current = _axis_transfer_rotation(
                demo_z,
                current_z,
                demo_target[:3, 3],
                current_target[:3, 3],
                controlled_origin_demo_world,
                controlled_origin_current_world,
            )
        else:
            # Backward-compatible fallback for callers that only provide target
            # geometry. Without a controlled-object radial direction, rotation
            # about the target axis is mathematically unobservable.
            rotation_demo_to_current = _minimal_rotation_between_vectors(
                demo_z,
                current_z,
            )

        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rotation_demo_to_current @ demo_rotation
        result[:3, 3] = current_target[:3, 3]
        return result

    if target_representation is TargetRepresentation.OBJECT_POSE_6D:
        return current_target.copy()

    raise ValueError(
        "Unsupported target representation for target-relative transfer: "
        f"{target_representation!r}"
    )


def infer_current_grasp_pose(scene: SceneGeometry) -> FloatArray:
    """Infer T_world_grasp from the effect pose and T_grasp_effect."""
    effect = scene.effect
    if effect.transform_world_gauge is None:
        raise ValueError(
            "Cannot infer current grasp pose: effect.transform_world_gauge is missing."
        )
    if effect.transform_grasp_feature is None:
        raise ValueError(
            "Cannot infer current grasp pose: effect.transform_grasp_feature is missing."
        )
    return compose_transform(
        effect.transform_world_gauge,
        invert_transform(effect.transform_grasp_feature),
    )


def _prepare_demo(
    trajectory: Union[PreparedDemoTrajectory, ArrayLike],
) -> PreparedDemoTrajectory:
    if isinstance(trajectory, PreparedDemoTrajectory):
        return trajectory
    return prepare_demo_trajectory(trajectory)


def _phase_demo_slice(
    prepared: PreparedDemoTrajectory,
    phase: PhaseSpec,
) -> Tuple[FloatArray, IntArray]:
    if phase.start_frame < 0 or phase.end_frame >= prepared.poses.shape[0]:
        raise ValueError(
            f"Phase {phase.phase_id!r} uses frames {phase.start_frame}-"
            f"{phase.end_frame}, but the demo contains frames 0-"
            f"{prepared.poses.shape[0] - 1}."
        )
    frame_slice = slice(phase.start_frame, phase.end_frame + 1)
    return (
        prepared.poses[frame_slice].copy(),
        prepared.frame_indices[frame_slice].copy(),
    )


def _build_grasp_relative_phase(
    phase: PhaseSpec,
    demo_grasp_poses: FloatArray,
    frame_indices: IntArray,
    transform_world_grasp_entry: FloatArray,
    transform_grasp_effect: FloatArray,
) -> _BuiltPhase:
    inverse_demo_entry = invert_transform(demo_grasp_poses[0])
    relative_poses = np.stack(
        [compose_transform(inverse_demo_entry, pose) for pose in demo_grasp_poses],
        axis=0,
    )
    grasp_reference = np.stack(
        [compose_transform(transform_world_grasp_entry, relative) for relative in relative_poses],
        axis=0,
    )
    effect_reference = np.einsum(
        "nij,jk->nik", grasp_reference, transform_grasp_effect
    )
    return _BuiltPhase(
        phase=phase,
        transform_world_grasp=grasp_reference,
        transform_world_effect=effect_reference,
        source_frame_indices=frame_indices,
        demo_relative_transform=relative_poses,
        demo_relative_valid_mask=np.ones(len(relative_poses), dtype=bool),
    )


def _build_target_relative_phase(
    phase: PhaseSpec,
    demo_grasp_poses: FloatArray,
    frame_indices: IntArray,
    transform_grasp_effect_demo: FloatArray,
    transform_grasp_effect_current: FloatArray,
    transform_world_target_demo: FloatArray,
    transform_world_target_current: FloatArray,
    target_representation: TargetRepresentation,
    *,
    controlled_origin_demo_world: Optional[ArrayLike] = None,
    controlled_origin_current_world: Optional[ArrayLike] = None,
    plane_press_down_m: float = 0.0,
) -> _BuiltPhase:
    """Transfer one demonstrated target-relative phase to the current scene.

    Cross-scene transfer uses only the target DOFs that are semantically defined
    by target_representation. point_normal and plane use target position +
    semantic z only. axis additionally resolves its otherwise undefined twist
    around z from the initial controlled-object radial direction. The target's
    reconstructed x/y gauge never directly rotates the transferred trajectory.
    """
    demo_effect_poses = np.einsum(
        "nij,jk->nik",
        demo_grasp_poses,
        transform_grasp_effect_demo,
    )

    # Store the learned effect motion relative to the DEMO target frame:
    #     T_Tdemo_Edemo(t) = inv(T_W_Tdemo) @ T_W_Edemo(t)
    inverse_demo_target = invert_transform(transform_world_target_demo)
    relative_effect_poses = np.stack(
        [
            compose_transform(inverse_demo_target, pose)
            for pose in demo_effect_poses
        ],
        axis=0,
    )

    if target_representation is TargetRepresentation.PLANE:
        # Hold this phase's entry height; positive press-down follows -plane Z.
        anchor_distance_m = float(relative_effect_poses[0, 2, 3])
        relative_effect_poses[:, 2, 3] = anchor_distance_m - plane_press_down_m

    # Build the representation-aware current target frame used only for
    # cross-scene trajectory instantiation.
    transform_world_target_instantiation = _target_relative_current_frame(
        transform_world_target_demo,
        transform_world_target_current,
        target_representation,
        controlled_origin_demo_world=controlled_origin_demo_world,
        controlled_origin_current_world=controlled_origin_current_world,
    )

    effect_reference = np.stack(
        [
            compose_transform(
                transform_world_target_instantiation,
                relative_pose,
            )
            for relative_pose in relative_effect_poses
        ],
        axis=0,
    )

    transform_effect_grasp_current = invert_transform(
        transform_grasp_effect_current
    )
    grasp_reference = np.einsum(
        "nij,jk->nik",
        effect_reference,
        transform_effect_grasp_current,
    )

    return _BuiltPhase(
        phase=phase,
        transform_world_grasp=grasp_reference,
        transform_world_effect=effect_reference,
        source_frame_indices=frame_indices,
        demo_relative_transform=relative_effect_poses,
        demo_relative_valid_mask=np.ones(
            len(relative_effect_poses),
            dtype=bool,
        ),
    )


def _build_attractor_phase(
    phase: PhaseSpec,
    transform_world_grasp_start: FloatArray,
    transform_world_grasp_goal: FloatArray,
    transform_grasp_effect: FloatArray,
) -> _BuiltPhase:
    sample_count = phase.frame_count
    if sample_count == 1:
        grasp_reference = transform_world_grasp_goal[None].copy()
    else:
        fractions = np.linspace(0.0, 1.0, sample_count)
        grasp_reference = np.stack(
            [
                interpolate_pose(
                    transform_world_grasp_start,
                    transform_world_grasp_goal,
                    _smoothstep(float(fraction)),
                )
                for fraction in fractions
            ],
            axis=0,
        )
    effect_reference = np.einsum(
        "nij,jk->nik", grasp_reference, transform_grasp_effect
    )
    return _BuiltPhase(
        phase=phase,
        transform_world_grasp=grasp_reference,
        transform_world_effect=effect_reference,
        source_frame_indices=np.arange(
            phase.start_frame,
            phase.end_frame + 1,
            dtype=np.int64,
        ),
        demo_relative_transform=np.repeat(
            np.eye(4, dtype=np.float64)[None], sample_count, axis=0
        ),
        demo_relative_valid_mask=np.zeros(sample_count, dtype=bool),
    )


def _target_axis_world(
    scene_target: GeometricFeature,
    axis: AxisName,
) -> FloatArray:
    representation = scene_target.representation
    if representation is TargetRepresentation.POINT:
        raise ValueError("Point targets do not provide a target axis.")
    if (
        representation
        in {
            TargetRepresentation.POINT_NORMAL,
            TargetRepresentation.AXIS,
            TargetRepresentation.PLANE,
        }
        and axis is not AxisName.Z
    ):
        raise ValueError(
            f"Target representation {representation.value!r} provides only semantic z."
        )

    if axis is AxisName.Z:
        if representation in {
            TargetRepresentation.POINT_NORMAL,
            TargetRepresentation.PLANE,
        } and scene_target.normal_world is not None:
            return _normalize(scene_target.normal_world, name="target normal")
        if representation is TargetRepresentation.AXIS and scene_target.axis_world is not None:
            return _normalize(scene_target.axis_world, name="target axis")

    target_frame = _target_transfer_frame(scene_target)
    return _normalize(
        target_frame[:3, _AXIS_TO_INDEX[axis]],
        name=f"target {axis.value}-axis",
    )


def _phase_range(
    phase: PhaseSpec,
    sample_start: int,
    sample_stop: int,
) -> PhaseReferenceRange:
    cycle_start: Optional[int] = None
    cycle_stop: Optional[int] = None
    if phase.cycle_interval is not None:
        cycle_start = sample_start + (
            phase.cycle_interval.start_frame - phase.start_frame
        )
        cycle_stop = sample_start + (
            phase.cycle_interval.end_frame - phase.start_frame + 1
        )

    return PhaseReferenceRange(
        phase_id=phase.phase_id,
        name=phase.name,
        sample_start=sample_start,
        sample_stop=sample_stop,
        source_frame_start=phase.start_frame,
        source_frame_end=phase.end_frame,
        motion_source=phase.motion_source,
        reference_frame=phase.reference_frame,
        attractor_goal_phase_id=phase.attractor_goal_phase_id,
        rotation_mode=phase.rotation_policy.mode,
        remaining_rotation=phase.rotation_policy.remaining_rotation,
        repeatable=phase.repeatable,
        cycle_sample_start=cycle_start,
        cycle_sample_stop=cycle_stop,
    )


def _assemble_motion_reference(
    built_phases: list[_BuiltPhase],
    current_target: GeometricFeature,
    transform_world_grasp_start: FloatArray,
    *,
    transform_grasp_effect_demo: FloatArray,
    transform_grasp_effect_current: FloatArray,
    transform_world_target_demo: FloatArray,
    transform_world_target_current: FloatArray,
    current_grasp_was_inferred: bool,
    demo_initial_grasp_position_error_m: float,
    demo_initial_grasp_rotation_error_rad: float,
) -> MotionReference:
    """Combine built phases, constraint masks, and diagnostics into one reference."""
    total_samples = sum(item.transform_world_grasp.shape[0] for item in built_phases)
    if total_samples < 1:
        raise ValueError("No phase samples were built.")

    grasp_trajectory = np.concatenate(
        [item.transform_world_grasp for item in built_phases], axis=0
    )
    effect_trajectory = np.concatenate(
        [item.transform_world_effect for item in built_phases], axis=0
    )
    source_frame_indices = np.concatenate(
        [item.source_frame_indices for item in built_phases], axis=0
    )
    demo_relative_transform = np.concatenate(
        [item.demo_relative_transform for item in built_phases], axis=0
    )
    demo_relative_valid_mask = np.concatenate(
        [item.demo_relative_valid_mask for item in built_phases], axis=0
    )
    demo_relative_reference_frame_index = np.full(
        total_samples, -1, dtype=np.int64
    )

    position_targets = effect_trajectory[:, :3, 3].copy()
    position_mask = np.ones(total_samples, dtype=bool)

    rotation_mode_index = np.empty(total_samples, dtype=np.int64)
    remaining_rotation_index = np.empty(total_samples, dtype=np.int64)
    full_rotation_targets = effect_trajectory[:, :3, :3].copy()
    full_rotation_mask = np.zeros(total_samples, dtype=bool)
    remaining_rotation_fixed_mask = np.zeros(total_samples, dtype=bool)

    axis_constraint_mask = np.zeros(
        (total_samples, MAX_AXIS_CONSTRAINTS), dtype=bool
    )
    axis_controlled_axis_index = np.full(
        (total_samples, MAX_AXIS_CONSTRAINTS), -1, dtype=np.int64
    )
    axis_reference_frame_index = np.full(
        (total_samples, MAX_AXIS_CONSTRAINTS), -1, dtype=np.int64
    )
    axis_reference_axis_index = np.full(
        (total_samples, MAX_AXIS_CONSTRAINTS), -1, dtype=np.int64
    )
    axis_relation_index = np.full(
        (total_samples, MAX_AXIS_CONSTRAINTS), -1, dtype=np.int64
    )
    axis_reference_directions_world = np.full(
        (total_samples, MAX_AXIS_CONSTRAINTS, 3), np.nan, dtype=np.float64
    )

    phase_symmetry_mask = np.zeros(total_samples, dtype=bool)
    phase_symmetry_axis_world = np.full(
        (total_samples, 3), np.nan, dtype=np.float64
    )
    phase_index = np.empty(total_samples, dtype=np.int64)

    phase_ranges: list[PhaseReferenceRange] = []
    position_jumps: list[float] = []
    rotation_jumps: list[float] = []

    sample_start = 0
    previous_pose = transform_world_grasp_start

    for phase_number, built in enumerate(built_phases):
        phase = built.phase
        sample_count = built.transform_world_grasp.shape[0]
        sample_stop = sample_start + sample_count
        sample_slice = slice(sample_start, sample_stop)

        first_pose = built.transform_world_grasp[0]
        position_jumps.append(
            float(np.linalg.norm(first_pose[:3, 3] - previous_pose[:3, 3]))
        )
        rotation_jumps.append(
            _rotation_error_angle(previous_pose[:3, :3], first_pose[:3, :3])
        )

        rotation_mode_index[sample_slice] = _ROTATION_MODE_TO_INDEX[
            phase.rotation_policy.mode
        ]
        remaining_rotation_index[sample_slice] = _REMAINING_ROTATION_TO_INDEX[
            phase.rotation_policy.remaining_rotation
        ]
        full_rotation_mask[sample_slice] = (
            phase.rotation_policy.mode is RotationMode.FULL_DEMO
        )
        remaining_rotation_fixed_mask[sample_slice] = (
            phase.rotation_policy.mode is RotationMode.AXIS_CONSTRAINED
            and phase.rotation_policy.remaining_rotation
            is RemainingRotation.FIXED_FROM_DEMO
        )
        phase_index[sample_slice] = phase_number
        if phase.motion_source is MotionSource.LEARN_DEMO:
            if phase.reference_frame is None:
                raise ValueError(
                    f"Learned phase {phase.phase_id!r} is missing reference_frame."
                )
            demo_relative_reference_frame_index[sample_slice] = (
                _REFERENCE_FRAME_TO_INDEX[phase.reference_frame]
            )

        # Axis constraints act on the controlled/effect frame.
        # For reference_frame="grasp", preserve the controlled-object axis at
        # phase entry (the semantic meaning used by the VLM), rather than the
        # auxiliary grasp-gauge axis. This avoids a spurious orientation jump
        # when T_grasp_effect has a non-identity rotation.
        phase_entry_controlled = built.transform_world_effect[0]
        for slot, constraint in enumerate(phase.rotation_policy.axis_constraints):
            if slot >= MAX_AXIS_CONSTRAINTS:
                raise ValueError(
                    f"Phase {phase.phase_id!r} has more than "
                    f"{MAX_AXIS_CONSTRAINTS} axis constraints."
                )
            if constraint.reference_frame is ReferenceFrame.GRASP:
                direction = phase_entry_controlled[
                    :3, _AXIS_TO_INDEX[constraint.reference_axis]
                ]
            elif constraint.reference_frame is ReferenceFrame.TARGET:
                direction = _target_axis_world(current_target, constraint.reference_axis)
            else:
                raise ValueError(
                    f"Unsupported axis reference frame: {constraint.reference_frame}."
                )
            direction = _normalize(direction, name="axis reference direction")

            axis_constraint_mask[sample_slice, slot] = True
            axis_controlled_axis_index[sample_slice, slot] = _AXIS_TO_INDEX[
                constraint.controlled_axis
            ]
            axis_reference_frame_index[sample_slice, slot] = (
                _REFERENCE_FRAME_TO_INDEX[constraint.reference_frame]
            )
            axis_reference_axis_index[sample_slice, slot] = _AXIS_TO_INDEX[
                constraint.reference_axis
            ]
            axis_relation_index[sample_slice, slot] = _RELATION_TO_INDEX[
                constraint.relation
            ]
            axis_reference_directions_world[sample_slice, slot, :] = direction

        symmetry = phase.rotation_policy.phase_symmetry
        if symmetry.mode is PhaseSymmetryMode.SHARED_OFFSET:
            if symmetry.reference_frame is not ReferenceFrame.TARGET:
                raise ValueError("shared_offset currently requires target reference frame.")
            if symmetry.reference_axis is None:
                raise ValueError("shared_offset is missing its reference axis.")
            symmetry_axis = _target_axis_world(current_target, symmetry.reference_axis)
            phase_symmetry_mask[sample_slice] = True
            phase_symmetry_axis_world[sample_slice] = symmetry_axis

        phase_ranges.append(_phase_range(phase, sample_start, sample_stop))
        previous_pose = built.transform_world_grasp[-1]
        sample_start = sample_stop

    diagnostics = ReferenceBuildDiagnostics(
        current_grasp_was_inferred=current_grasp_was_inferred,
        demo_initial_grasp_position_error_m=(
            demo_initial_grasp_position_error_m
        ),
        demo_initial_grasp_rotation_error_rad=(
            demo_initial_grasp_rotation_error_rad
        ),
        phase_entry_position_jumps_m=np.asarray(position_jumps, dtype=np.float64),
        phase_entry_rotation_jumps_rad=np.asarray(rotation_jumps, dtype=np.float64),
    )

    return MotionReference(
        transform_world_grasp_gauge=grasp_trajectory,
        transform_world_effect_gauge=effect_trajectory,
        position_targets_world=position_targets,
        position_mask=position_mask,
        demo_relative_transform=demo_relative_transform,
        demo_relative_valid_mask=demo_relative_valid_mask,
        demo_relative_reference_frame_index=(
            demo_relative_reference_frame_index
        ),
        rotation_mode_index=rotation_mode_index,
        remaining_rotation_index=remaining_rotation_index,
        full_rotation_targets_world=full_rotation_targets,
        full_rotation_mask=full_rotation_mask,
        remaining_rotation_fixed_mask=remaining_rotation_fixed_mask,
        axis_constraint_mask=axis_constraint_mask,
        axis_controlled_axis_index=axis_controlled_axis_index,
        axis_reference_frame_index=axis_reference_frame_index,
        axis_reference_axis_index=axis_reference_axis_index,
        axis_relation_index=axis_relation_index,
        axis_reference_directions_world=axis_reference_directions_world,
        phase_symmetry_mask=phase_symmetry_mask,
        phase_symmetry_axis_world=phase_symmetry_axis_world,
        phase_index=phase_index,
        source_frame_indices=source_frame_indices,
        phases=tuple(phase_ranges),
        transform_grasp_effect=transform_grasp_effect_current,
        transform_grasp_effect_demo=transform_grasp_effect_demo,
        transform_world_target_current=transform_world_target_current,
        transform_world_target_demo=transform_world_target_demo,
        target_representation=current_target.representation,
        controlled_frame="effect",
        diagnostics=diagnostics,
    )


def build_motion_reference(
    demo_grasp_trajectory: Union[PreparedDemoTrajectory, ArrayLike],
    spec: VideoStructureSpec,
    demo_scene: SceneGeometry,
    current_scene: SceneGeometry,
    *,
    transform_world_grasp_current: Optional[ArrayLike] = None,
    plane_press_down_m: float = 0.0,
) -> MotionReference:
    """Build a cross-environment task-space motion reference.

    Scene geometry is reconstructed independently for the demonstration
    and current initial scenes. This function then:

    * stores each demonstrated learned phase in its semantic relative frame;
    * transfers grasp-relative phases from the current phase-entry grasp;
    * transfers target-relative effect phases from the demo target geometry to
      the current target geometry;
    * creates current-scene attractors to later transferred phase entries; and
    * packages rotational relationships without applying robot FK.

    The full demonstration trajectory is used for motion. Only the initial demo
    RGB-D frame is needed for `demo_scene`, because it supplies the fixed demo
    effect attachment and demo target reference geometry.
    Plane target-relative phases hold their demonstrated entry normal distance,
    reduced by plane_press_down_m. Approach attractors inherit that shifted goal.
    """
    if not np.isfinite(plane_press_down_m) or plane_press_down_m < 0.0:
        raise ValueError("plane_press_down_m must be finite and non-negative.")
    prepared = _prepare_demo(demo_grasp_trajectory)

    for label, scene in (("demo", demo_scene), ("current", current_scene)):
        if scene.target.representation is not spec.target_representation:
            raise ValueError(
                f"Video specification and {label} SceneGeometry disagree on "
                "target representation: "
                f"{spec.target_representation.value!r} versus "
                f"{scene.target.representation.value!r}."
            )
        if scene.effect.transform_grasp_feature is None:
            raise ValueError(
                f"{label.capitalize()} scene effect is missing "
                "transform_grasp_feature (T_grasp_effect)."
            )

    if demo_scene.effect.representation is not current_scene.effect.representation:
        raise ValueError(
            "Demo and current scenes disagree on effect representation: "
            f"{demo_scene.effect.representation.value!r} versus "
            f"{current_scene.effect.representation.value!r}."
        )

    transform_grasp_effect_demo_array = as_transform(
        demo_scene.effect.transform_grasp_feature
    )
    transform_grasp_effect_current = as_transform(
        current_scene.effect.transform_grasp_feature
    )

    current_grasp_was_inferred = transform_world_grasp_current is None
    transform_world_grasp_start = (
        infer_current_grasp_pose(current_scene)
        if transform_world_grasp_current is None
        else as_transform(transform_world_grasp_current)
    )

    transform_world_target_demo_array = _target_transfer_frame(demo_scene.target)
    transform_world_target_current = _target_transfer_frame(current_scene.target)

    # Initial controlled-object origins resolve the otherwise unobservable
    # rotation about an AXIS target. This is intentionally based on the current
    # observed scene, not on the future target-phase entry, so target-relative
    # phases can still be precomputed before attractor construction.
    if demo_scene.effect.transform_world_gauge is None:
        raise ValueError(
            "Demo scene effect is missing transform_world_gauge required for "
            "axis transfer."
        )
    if current_scene.effect.transform_world_gauge is None:
        raise ValueError(
            "Current scene effect is missing transform_world_gauge required for "
            "axis transfer."
        )
    controlled_origin_demo_world = as_transform(
        demo_scene.effect.transform_world_gauge
    )[:3, 3].copy()
    controlled_origin_current_world = as_transform(
        current_scene.effect.transform_world_gauge
    )[:3, 3].copy()

    # This should be near zero when demo_scene was built from the same initial
    # frame as demo_grasp_trajectory[0]. Keep it diagnostic rather than a hard
    # failure because reconstructed trajectories may contain small noise.
    demo_grasp_from_scene = infer_current_grasp_pose(demo_scene)
    demo_initial_pose = prepared.poses[0]
    demo_initial_grasp_position_error_m = float(
        np.linalg.norm(
            demo_grasp_from_scene[:3, 3] - demo_initial_pose[:3, 3]
        )
    )
    demo_initial_grasp_rotation_error_rad = _rotation_error_angle(
        demo_grasp_from_scene[:3, :3], demo_initial_pose[:3, :3]
    )

    # Precompute every target-relative learned phase. Attractor phases need the
    # entry pose of a later target-relative phase before sequential assembly.
    target_phase_cache: dict[str, _BuiltPhase] = {}
    for phase in spec.phases:
        if (
            phase.motion_source is MotionSource.LEARN_DEMO
            and phase.reference_frame is ReferenceFrame.TARGET
        ):
            demo_slice, frame_indices = _phase_demo_slice(prepared, phase)
            target_phase_cache[phase.phase_id] = _build_target_relative_phase(
                phase,
                demo_slice,
                frame_indices,
                transform_grasp_effect_demo_array,
                transform_grasp_effect_current,
                transform_world_target_demo_array,
                transform_world_target_current,
                spec.target_representation,
                controlled_origin_demo_world=controlled_origin_demo_world,
                controlled_origin_current_world=controlled_origin_current_world,
                plane_press_down_m=plane_press_down_m,
            )

    built_phases: list[_BuiltPhase] = []
    current_entry = transform_world_grasp_start

    for phase in spec.phases:
        if phase.motion_source is MotionSource.ATTRACTOR:
            goal_id = phase.attractor_goal_phase_id
            if goal_id is None or goal_id not in target_phase_cache:
                raise ValueError(
                    f"Attractor phase {phase.phase_id!r} does not identify a "
                    "precomputable target-relative learned goal phase."
                )
            built = _build_attractor_phase(
                phase,
                current_entry,
                target_phase_cache[goal_id].transform_world_grasp[0],
                transform_grasp_effect_current,
            )
        elif phase.reference_frame is ReferenceFrame.GRASP:
            demo_slice, frame_indices = _phase_demo_slice(prepared, phase)
            built = _build_grasp_relative_phase(
                phase,
                demo_slice,
                frame_indices,
                current_entry,
                transform_grasp_effect_current,
            )
        elif phase.reference_frame is ReferenceFrame.TARGET:
            built = target_phase_cache[phase.phase_id]
        else:
            raise ValueError(
                f"Unsupported phase configuration for {phase.phase_id!r}."
            )

        built_phases.append(built)
        current_entry = built.transform_world_grasp[-1]

    return _assemble_motion_reference(
        built_phases,
        current_scene.target,
        transform_world_grasp_start,
        transform_grasp_effect_demo=transform_grasp_effect_demo_array,
        transform_grasp_effect_current=transform_grasp_effect_current,
        transform_world_target_demo=transform_world_target_demo_array,
        transform_world_target_current=transform_world_target_current,
        current_grasp_was_inferred=current_grasp_was_inferred,
        demo_initial_grasp_position_error_m=demo_initial_grasp_position_error_m,
        demo_initial_grasp_rotation_error_rad=demo_initial_grasp_rotation_error_rad,
    )


def _phase_range_to_dict(phase: PhaseReferenceRange) -> dict[str, Any]:
    return {
        "phase_id": phase.phase_id,
        "name": phase.name,
        "sample_start": phase.sample_start,
        "sample_stop": phase.sample_stop,
        "source_frame_start": phase.source_frame_start,
        "source_frame_end": phase.source_frame_end,
        "motion_source": phase.motion_source.value,
        "reference_frame": (
            None if phase.reference_frame is None else phase.reference_frame.value
        ),
        "attractor_goal_phase_id": phase.attractor_goal_phase_id,
        "rotation_mode": phase.rotation_mode.value,
        "remaining_rotation": phase.remaining_rotation.value,
        "repeatable": phase.repeatable,
        "cycle_sample_start": phase.cycle_sample_start,
        "cycle_sample_stop": phase.cycle_sample_stop,
    }


def save_motion_reference(path: Union[str, Path], reference: MotionReference) -> None:
    """Save the motion-reference interface as a compressed NPZ bundle."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format_version": 2,
        "controlled_frame": reference.controlled_frame,
        "target_representation": reference.target_representation.value,
        "phases": [_phase_range_to_dict(phase) for phase in reference.phases],
        "diagnostics": {
            "current_grasp_was_inferred": (
                reference.diagnostics.current_grasp_was_inferred
            ),
            "demo_initial_grasp_position_error_m": (
                reference.diagnostics.demo_initial_grasp_position_error_m
            ),
            "demo_initial_grasp_rotation_error_rad": (
                reference.diagnostics.demo_initial_grasp_rotation_error_rad
            ),
        },
        "index_maps": {
            name: {member.value: index for member, index in mapping.items()}
            for name, mapping in (
                ("axis", _AXIS_TO_INDEX),
                ("reference_frame", _REFERENCE_FRAME_TO_INDEX),
                ("axis_relation", _RELATION_TO_INDEX),
                ("rotation_mode", _ROTATION_MODE_TO_INDEX),
                ("remaining_rotation", _REMAINING_ROTATION_TO_INDEX),
            )
        },
    }

    array_fields = (
        "transform_world_grasp_gauge", "transform_world_effect_gauge",
        *_REFERENCE_SAMPLE_SHAPES,
        "transform_grasp_effect", "transform_grasp_effect_demo",
        "transform_world_target_current", "transform_world_target_demo",
    )
    np.savez_compressed(
        output,
        **{name: getattr(reference, name) for name in array_fields},
        diagnostic_phase_entry_position_jumps_m=(
            reference.diagnostics.phase_entry_position_jumps_m
        ),
        diagnostic_phase_entry_rotation_jumps_rad=(
            reference.diagnostics.phase_entry_rotation_jumps_rad
        ),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


__all__ = [
    "MAX_AXIS_CONSTRAINTS",
    "MotionReference",
    "PhaseReferenceRange",
    "ReferenceBuildDiagnostics",
    "build_motion_reference",
    "infer_current_grasp_pose",
    "save_motion_reference",
]
