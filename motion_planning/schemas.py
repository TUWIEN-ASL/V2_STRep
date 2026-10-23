from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class _StringEnum(str, Enum):
    """Enum whose values serialize naturally as JSON strings."""

    def __str__(self) -> str:
        return self.value


class TargetRepresentation(_StringEnum):
    POINT = "point"
    POINT_NORMAL = "point_normal"
    AXIS = "axis"
    PLANE = "plane"
    OBJECT_POSE_6D = "object_pose_6d"


class MotionSource(_StringEnum):
    LEARN_DEMO = "learn_demo"
    ATTRACTOR = "attractor"


class ReferenceFrame(_StringEnum):
    GRASP = "grasp"
    TARGET = "target"


class RotationMode(_StringEnum):
    FREE = "free"
    AXIS_CONSTRAINED = "axis_constrained"
    FULL_DEMO = "full_demo"


class RemainingRotation(_StringEnum):
    FREE = "free"
    FIXED_FROM_DEMO = "fixed_from_demo"
    NONE = "none"


class PhaseSymmetryMode(_StringEnum):
    NONE = "none"
    SHARED_OFFSET = "shared_offset"


class AxisName(_StringEnum):
    X = "x"
    Y = "y"
    Z = "z"


class AxisRelation(_StringEnum):
    SAME_DIRECTION = "same_direction"
    OPPOSITE_DIRECTION = "opposite_direction"
    PARALLEL_EITHER_DIRECTION = "parallel_either_direction"


@dataclass(frozen=True)
class EntitySpec:
    description: str


@dataclass(frozen=True)
class ControlledPointSpec:
    required: bool
    description: Optional[str]


@dataclass(frozen=True)
class CycleInterval:
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class AxisConstraint:
    controlled_axis: AxisName
    reference_frame: ReferenceFrame
    reference_axis: AxisName
    relation: AxisRelation
    controlled_positive_end_description: Optional[str]
    reference_positive_end_description: Optional[str]


@dataclass(frozen=True)
class DemoAngleReference:
    """Visual reference used to compare demo and current-scene azimuth."""

    comparison_frame: int
    anchor: str
    anchor_description: str
    radial_direction_description: str


@dataclass(frozen=True)
class PhaseSymmetry:
    mode: PhaseSymmetryMode
    reference_frame: Optional[ReferenceFrame]
    reference_axis: Optional[AxisName]
    demo_angle_reference: Optional[DemoAngleReference] = None


@dataclass(frozen=True)
class RotationPolicy:
    mode: RotationMode
    axis_constraints: Tuple[AxisConstraint, ...]
    remaining_rotation: RemainingRotation
    phase_symmetry: PhaseSymmetry


@dataclass(frozen=True)
class PhaseSpec:
    phase_id: str
    name: str
    start_frame: int
    end_frame: int
    motion_source: MotionSource
    reference_frame: Optional[ReferenceFrame]
    attractor_goal_phase_id: Optional[str]
    repeatable: bool
    cycle_interval: Optional[CycleInterval]
    rotation_policy: RotationPolicy

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass(frozen=True)
class VideoStructureSpec:
    controlled_entity: EntitySpec
    target_entity: EntitySpec
    controlled_point: ControlledPointSpec
    target_representation: TargetRepresentation
    phases: Tuple[PhaseSpec, ...]

    def phase_by_id(self, phase_id: str) -> PhaseSpec:
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        raise KeyError(f"Unknown phase_id: {phase_id}")
