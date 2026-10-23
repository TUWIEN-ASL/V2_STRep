from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence, Type, TypeVar

from .schemas import (
    AxisConstraint,
    AxisName,
    AxisRelation,
    ControlledPointSpec,
    CycleInterval,
    DemoAngleReference,
    EntitySpec,
    MotionSource,
    PhaseSpec,
    PhaseSymmetry,
    PhaseSymmetryMode,
    ReferenceFrame,
    RemainingRotation,
    RotationMode,
    RotationPolicy,
    TargetRepresentation,
    VideoStructureSpec,
)


EnumT = TypeVar("EnumT")


class SpecValidationError(ValueError):
    """Raised when the video-structure JSON violates the planning schema."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        message = "Invalid video structure:\n- " + "\n- ".join(self.errors)
        super().__init__(message)


def _expect_mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be a JSON object.")
        return {}
    return value


def _expect_exact_keys(
    mapping: Mapping[str, Any],
    expected_keys: set[str],
    path: str,
    errors: list[str],
    *,
    optional_keys: Sequence[str] = (),
) -> None:
    actual_keys = set(mapping.keys())
    missing = expected_keys - actual_keys
    unknown = actual_keys - expected_keys - set(optional_keys)

    for key in sorted(missing):
        errors.append(f"{path} is missing required field '{key}'.")
    for key in sorted(unknown):
        errors.append(f"{path} contains unsupported field '{key}'.")


def _expect_string(value: Any, path: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path} must be a non-empty string.")
        return ""
    return value.strip()


def _expect_optional_string(
    value: Any,
    path: str,
    errors: list[str],
) -> Optional[str]:
    if value is None:
        return None
    return _expect_string(value, path, errors)


def _expect_bool(value: Any, path: str, errors: list[str]) -> bool:
    if not isinstance(value, bool):
        errors.append(f"{path} must be true or false.")
        return False
    return value


def _expect_int(value: Any, path: str, errors: list[str]) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{path} must be an integer.")
        return 0
    return value


def _expect_enum(
    enum_type: Type[EnumT],
    value: Any,
    path: str,
    errors: list[str],
) -> Optional[EnumT]:
    if not isinstance(value, str):
        errors.append(f"{path} must be a string.")
        return None
    try:
        return enum_type(value)
    except ValueError:
        choices = ", ".join(repr(member.value) for member in enum_type)
        errors.append(f"{path} must be one of: {choices}.")
        return None


def _expect_optional_enum(
    enum_type: Type[EnumT],
    value: Any,
    path: str,
    errors: list[str],
) -> Optional[EnumT]:
    if value is None:
        return None
    return _expect_enum(enum_type, value, path, errors)


class _ObjectFields:
    """Read object fields using their shared JSON path and error list."""

    def __init__(
        self, value: Any, path: str, errors: list[str], required_keys: set[str],
        *, optional_keys: Sequence[str] = (),
    ) -> None:
        self.mapping = _expect_mapping(value, path, errors)
        self.path = path
        self.errors = errors
        _expect_exact_keys(
            self.mapping, required_keys, path, errors, optional_keys=optional_keys,
        )

    def read(self, name: str, parser: Callable[[Any, str, list[str]], Any]) -> Any:
        return parser(self.mapping.get(name), f"{self.path}.{name}", self.errors)

    def enum(
        self, name: str, enum_type: Type[EnumT], *, optional: bool = False,
    ) -> Optional[EnumT]:
        parser = _expect_optional_enum if optional else _expect_enum
        return parser(
            enum_type, self.mapping.get(name), f"{self.path}.{name}", self.errors,
        )


def _parse_entity(
    value: Any,
    path: str,
    errors: list[str],
) -> EntitySpec:
    fields = _ObjectFields(value, path, errors, {"description"})
    return EntitySpec(description=fields.read("description", _expect_string))


def _parse_cycle_interval(
    value: Any,
    path: str,
    errors: list[str],
) -> Optional[CycleInterval]:
    if value is None:
        return None
    fields = _ObjectFields(value, path, errors, {"start_frame", "end_frame"})
    return CycleInterval(
        start_frame=fields.read("start_frame", _expect_int),
        end_frame=fields.read("end_frame", _expect_int),
    )


def _parse_axis_constraint(
    value: Any,
    path: str,
    errors: list[str],
) -> AxisConstraint:
    fields = _ObjectFields(value, path, errors, {
        "controlled_axis", "reference_frame", "reference_axis", "relation",
        "controlled_positive_end_description", "reference_positive_end_description",
    })
    return AxisConstraint(
        controlled_axis=fields.enum("controlled_axis", AxisName) or AxisName.X,
        reference_frame=fields.enum("reference_frame", ReferenceFrame) or ReferenceFrame.GRASP,
        reference_axis=fields.enum("reference_axis", AxisName) or AxisName.X,
        relation=fields.enum("relation", AxisRelation) or AxisRelation.PARALLEL_EITHER_DIRECTION,
        controlled_positive_end_description=fields.read(
            "controlled_positive_end_description", _expect_optional_string,
        ),
        reference_positive_end_description=fields.read(
            "reference_positive_end_description", _expect_optional_string,
        ),
    )


def _parse_demo_angle_reference(
    value: Any,
    path: str,
    errors: list[str],
) -> Optional[DemoAngleReference]:
    """Parse the optional visual azimuth reference for a shared-offset phase."""
    if value is None:
        return None
    fields = _ObjectFields(value, path, errors, {
        "comparison_frame", "anchor", "anchor_description", "radial_direction_description",
    })
    comparison_frame = fields.read("comparison_frame", _expect_int)
    anchor = fields.read("anchor", _expect_string)
    allowed_anchors = {
        "controlled_point",
        "grasp_origin",
        "controlled_entity_center",
    }
    if anchor and anchor not in allowed_anchors:
        choices = ", ".join(repr(item) for item in sorted(allowed_anchors))
        errors.append(f"{path}.anchor must be one of: {choices}.")

    return DemoAngleReference(
        comparison_frame=comparison_frame,
        anchor=anchor,
        anchor_description=fields.read("anchor_description", _expect_string),
        radial_direction_description=fields.read("radial_direction_description", _expect_string),
    )


def _parse_phase_symmetry(
    value: Any,
    path: str,
    errors: list[str],
) -> PhaseSymmetry:
    fields = _ObjectFields(
        value, path, errors, {"mode", "reference_frame", "reference_axis"},
        optional_keys=("demo_angle_reference",),
    )
    return PhaseSymmetry(
        mode=fields.enum("mode", PhaseSymmetryMode) or PhaseSymmetryMode.NONE,
        reference_frame=fields.enum("reference_frame", ReferenceFrame, optional=True),
        reference_axis=fields.enum("reference_axis", AxisName, optional=True),
        demo_angle_reference=fields.read("demo_angle_reference", _parse_demo_angle_reference),
    )


def _parse_rotation_policy(
    value: Any,
    path: str,
    errors: list[str],
) -> RotationPolicy:
    fields = _ObjectFields(value, path, errors, {
        "mode", "axis_constraints", "remaining_rotation", "phase_symmetry",
    })
    mode = fields.enum("mode", RotationMode)
    raw_constraints = fields.mapping.get("axis_constraints")
    if not isinstance(raw_constraints, list):
        errors.append(f"{path}.axis_constraints must be a JSON array.")
        raw_constraints = []
    axis_constraints = tuple(
        _parse_axis_constraint(item, f"{path}.axis_constraints[{index}]", errors)
        for index, item in enumerate(raw_constraints)
    )
    return RotationPolicy(
        mode=mode or RotationMode.FREE,
        axis_constraints=axis_constraints,
        remaining_rotation=(
            fields.enum("remaining_rotation", RemainingRotation) or RemainingRotation.FREE
        ),
        phase_symmetry=fields.read("phase_symmetry", _parse_phase_symmetry),
    )


def _parse_phase(
    value: Any,
    index: int,
    errors: list[str],
) -> PhaseSpec:
    fields = _ObjectFields(value, f"phases[{index}]", errors, {
        "phase_id", "name", "start_frame", "end_frame", "motion_source",
        "reference_frame", "attractor_goal_phase_id", "repeatable",
        "cycle_interval", "rotation_policy",
    })
    return PhaseSpec(
        phase_id=fields.read("phase_id", _expect_string),
        name=fields.read("name", _expect_string),
        start_frame=fields.read("start_frame", _expect_int),
        end_frame=fields.read("end_frame", _expect_int),
        motion_source=fields.enum("motion_source", MotionSource) or MotionSource.LEARN_DEMO,
        reference_frame=fields.enum("reference_frame", ReferenceFrame, optional=True),
        attractor_goal_phase_id=fields.read("attractor_goal_phase_id", _expect_optional_string),
        repeatable=fields.read("repeatable", _expect_bool),
        cycle_interval=fields.read("cycle_interval", _parse_cycle_interval),
        rotation_policy=fields.read("rotation_policy", _parse_rotation_policy),
    )


def _validate_axis_reference(
    constraint: AxisConstraint,
    *,
    target_representation: TargetRepresentation,
    path: str,
    errors: list[str],
) -> None:
    if constraint.reference_frame is not ReferenceFrame.TARGET:
        return

    if target_representation is TargetRepresentation.POINT:
        errors.append(f"{path} references a target axis, but point targets have no axis.")
        return

    if (
        target_representation
        in {
            TargetRepresentation.POINT_NORMAL,
            TargetRepresentation.AXIS,
            TargetRepresentation.PLANE,
        }
        and constraint.reference_axis is not AxisName.Z
    ):
        errors.append(
            f"{path} may reference only target z for "
            f"target_representation={target_representation.value!r}."
        )


def _validate_rotation_policy(
    phase: PhaseSpec,
    *,
    target_representation: TargetRepresentation,
    path: str,
    errors: list[str],
) -> None:
    policy = phase.rotation_policy
    constraint_count = len(policy.axis_constraints)

    if policy.mode is RotationMode.FREE:
        if constraint_count != 0:
            errors.append(f"{path}.axis_constraints must be empty for free mode.")
        if policy.remaining_rotation is not RemainingRotation.FREE:
            errors.append(f"{path}.remaining_rotation must be 'free' for free mode.")
        if policy.phase_symmetry.mode is not PhaseSymmetryMode.NONE:
            errors.append(f"{path}.phase_symmetry.mode must be 'none' for free mode.")

    elif policy.mode is RotationMode.AXIS_CONSTRAINED:
        if constraint_count not in {1, 2}:
            errors.append(
                f"{path}.axis_constraints must contain one or two constraints "
                "for axis_constrained mode."
            )
        if policy.phase_symmetry.mode is not PhaseSymmetryMode.NONE:
            errors.append(
                f"{path}.phase_symmetry.mode must be 'none' for axis_constrained mode."
            )

        if constraint_count == 1 and policy.remaining_rotation not in {
            RemainingRotation.FREE,
            RemainingRotation.FIXED_FROM_DEMO,
        }:
            errors.append(
                f"{path}.remaining_rotation must be 'free' or 'fixed_from_demo' "
                "when one axis is constrained."
            )

        if constraint_count == 2:
            controlled_axes = {item.controlled_axis for item in policy.axis_constraints}
            reference_keys = {
                (item.reference_frame, item.reference_axis)
                for item in policy.axis_constraints
            }
            if len(controlled_axes) != 2 or len(reference_keys) != 2:
                errors.append(
                    f"{path}.axis_constraints must contain two independent axis "
                    "relations when two are provided."
                )
            if policy.remaining_rotation is not RemainingRotation.NONE:
                errors.append(
                    f"{path}.remaining_rotation must be 'none' when two "
                    "independent axes are constrained."
                )

    elif policy.mode is RotationMode.FULL_DEMO:
        if constraint_count != 0:
            errors.append(f"{path}.axis_constraints must be empty for full_demo mode.")
        if policy.remaining_rotation is not RemainingRotation.NONE:
            errors.append(
                f"{path}.remaining_rotation must be 'none' for full_demo mode."
            )

    symmetry = policy.phase_symmetry
    if symmetry.mode is PhaseSymmetryMode.NONE:
        if symmetry.reference_frame is not None:
            errors.append(
                f"{path}.phase_symmetry.reference_frame must be null "
                "when symmetry mode is 'none'."
            )
        if symmetry.reference_axis is not None:
            errors.append(
                f"{path}.phase_symmetry.reference_axis must be null "
                "when symmetry mode is 'none'."
            )
        if symmetry.demo_angle_reference is not None:
            errors.append(
                f"{path}.phase_symmetry.demo_angle_reference must be null "
                "when symmetry mode is 'none'."
            )
    else:
        if phase.motion_source is not MotionSource.LEARN_DEMO:
            errors.append(
                f"{path}.phase_symmetry shared_offset requires motion_source='learn_demo'."
            )
        if phase.reference_frame is not ReferenceFrame.TARGET:
            errors.append(
                f"{path}.phase_symmetry shared_offset requires reference_frame='target'."
            )
        if policy.mode is not RotationMode.FULL_DEMO:
            errors.append(
                f"{path}.phase_symmetry shared_offset requires rotation mode='full_demo'."
            )
        if symmetry.reference_frame is not ReferenceFrame.TARGET:
            errors.append(
                f"{path}.phase_symmetry.reference_frame must be 'target' "
                "for shared_offset."
            )
        if symmetry.reference_axis is None:
            errors.append(
                f"{path}.phase_symmetry.reference_axis is required for shared_offset."
            )
        elif target_representation is TargetRepresentation.POINT:
            errors.append(
                f"{path}.phase_symmetry cannot reference an axis for a point target."
            )
        elif (
            target_representation
            in {
                TargetRepresentation.POINT_NORMAL,
                TargetRepresentation.AXIS,
                TargetRepresentation.PLANE,
            }
            and symmetry.reference_axis is not AxisName.Z
        ):
            errors.append(
                f"{path}.phase_symmetry may use only target z for "
                f"target_representation={target_representation.value!r}."
            )

        # Legacy demo-angle metadata is optional and ignored by the ICRA
        # execution path. When present, still validate its frame range so older
        # saved specifications remain well formed.
        demo_reference = symmetry.demo_angle_reference
        if demo_reference is not None and not (
            phase.start_frame
            <= demo_reference.comparison_frame
            <= phase.end_frame
        ):
            errors.append(
                f"{path}.phase_symmetry.demo_angle_reference.comparison_frame "
                f"must lie inside [{phase.start_frame}, {phase.end_frame}]."
            )

    for constraint_index, constraint in enumerate(policy.axis_constraints):
        _validate_axis_reference(
            constraint,
            target_representation=target_representation,
            path=f"{path}.axis_constraints[{constraint_index}]",
            errors=errors,
        )


def _validate_spec(
    spec: VideoStructureSpec,
    *,
    total_frames: Optional[int],
    errors: list[str],
) -> None:
    if not spec.phases:
        errors.append("phases must contain at least one phase.")
        return

    if total_frames is not None:
        if not isinstance(total_frames, int) or isinstance(total_frames, bool):
            errors.append("total_frames must be an integer when provided.")
        elif total_frames < 1:
            errors.append("total_frames must be at least 1 when provided.")

    phase_ids: dict[str, int] = {}
    previous_phase: Optional[PhaseSpec] = None

    for index, phase in enumerate(spec.phases):
        path = f"phases[{index}]"

        if phase.phase_id in phase_ids:
            errors.append(
                f"{path}.phase_id duplicates phases[{phase_ids[phase.phase_id]}].phase_id."
            )
        else:
            phase_ids[phase.phase_id] = index

        if phase.start_frame < 0:
            errors.append(f"{path}.start_frame must be non-negative.")
        if phase.end_frame < phase.start_frame:
            errors.append(f"{path}.end_frame must be >= start_frame.")

        if total_frames is not None and isinstance(total_frames, int) and total_frames > 0:
            if phase.end_frame >= total_frames:
                errors.append(
                    f"{path}.end_frame={phase.end_frame} is outside "
                    f"the valid range [0, {total_frames - 1}]."
                )

        if previous_phase is not None:
            if phase.start_frame < previous_phase.start_frame:
                errors.append(f"{path} is not ordered by start_frame.")
            if phase.start_frame <= previous_phase.end_frame:
                errors.append(
                    f"{path} overlaps the previous phase "
                    f"'{previous_phase.phase_id}'."
                )
        previous_phase = phase

        if phase.motion_source is MotionSource.LEARN_DEMO:
            if phase.reference_frame not in {
                ReferenceFrame.GRASP,
                ReferenceFrame.TARGET,
            }:
                errors.append(
                    f"{path}.reference_frame must be 'grasp' or 'target' "
                    "for learn_demo."
                )
            if phase.attractor_goal_phase_id is not None:
                errors.append(
                    f"{path}.attractor_goal_phase_id must be null for learn_demo."
                )

        elif phase.motion_source is MotionSource.ATTRACTOR:
            if phase.reference_frame is not None:
                errors.append(
                    f"{path}.reference_frame must be null for attractor."
                )
            if phase.attractor_goal_phase_id is None:
                errors.append(
                    f"{path}.attractor_goal_phase_id is required for attractor."
                )
            if phase.repeatable:
                errors.append(f"{path}.repeatable must be false for attractor.")
            if phase.cycle_interval is not None:
                errors.append(f"{path}.cycle_interval must be null for attractor.")

        if phase.repeatable:
            if phase.motion_source is not MotionSource.LEARN_DEMO:
                errors.append(f"{path}.repeatable is allowed only for learn_demo.")
            if phase.cycle_interval is None:
                errors.append(
                    f"{path}.cycle_interval is required when repeatable is true."
                )
        elif phase.cycle_interval is not None:
            errors.append(
                f"{path}.cycle_interval must be null when repeatable is false."
            )

        if phase.cycle_interval is not None:
            cycle = phase.cycle_interval
            if cycle.start_frame > cycle.end_frame:
                errors.append(
                    f"{path}.cycle_interval.end_frame must be >= start_frame."
                )
            if (
                cycle.start_frame < phase.start_frame
                or cycle.end_frame > phase.end_frame
            ):
                errors.append(
                    f"{path}.cycle_interval must lie inside the phase interval."
                )

        _validate_rotation_policy(
            phase,
            target_representation=spec.target_representation,
            path=f"{path}.rotation_policy",
            errors=errors,
        )

    for index, phase in enumerate(spec.phases):
        if phase.motion_source is not MotionSource.ATTRACTOR:
            continue

        goal_id = phase.attractor_goal_phase_id
        if goal_id not in phase_ids:
            errors.append(
                f"phases[{index}].attractor_goal_phase_id={goal_id!r} "
                "does not identify an existing phase."
            )
            continue

        goal_index = phase_ids[goal_id]
        goal_phase = spec.phases[goal_index]

        if goal_index <= index:
            errors.append(
                f"phases[{index}].attractor_goal_phase_id must identify "
                "a later phase."
            )
        if goal_phase.motion_source is not MotionSource.LEARN_DEMO:
            errors.append(
                f"Attractor goal '{goal_id}' must use motion_source='learn_demo'."
            )
        if goal_phase.reference_frame is not ReferenceFrame.TARGET:
            errors.append(
                f"Attractor goal '{goal_id}' must use reference_frame='target'."
            )


def parse_video_structure(
    data: Mapping[str, Any],
    *,
    total_frames: Optional[int] = None,
) -> VideoStructureSpec:
    """
    Parse and validate a VIDEO_STRUCTURE_JSON dictionary.

    Args:
        data: Decoded JSON object.
        total_frames: Optional total video-frame count. When supplied, every
            phase and cycle index is checked against [0, total_frames - 1].

    Raises:
        SpecValidationError: If any schema or semantic rule is violated.
    """
    errors: list[str] = []
    mapping = _expect_mapping(data, "root", errors)

    # controlled_point was removed from the VLM contract.  Keep accepting it
    # as an optional legacy field, but never let it affect planning.
    required_keys = {
        "controlled_entity",
        "target_entity",
        "target_representation",
        "phases",
    }
    allowed_keys = required_keys | {"controlled_point"}
    actual_keys = set(mapping.keys())
    for key in sorted(required_keys - actual_keys):
        errors.append(f"root is missing required field '{key}'.")
    for key in sorted(actual_keys - allowed_keys):
        errors.append(f"root contains unsupported field '{key}'.")

    controlled_entity = _parse_entity(
        mapping.get("controlled_entity"),
        "controlled_entity",
        errors,
    )
    target_entity = _parse_entity(
        mapping.get("target_entity"),
        "target_entity",
        errors,
    )
    # The current ICRA pipeline always uses the controlled-object canonical
    # origin as the effect frame.  Controlled-point semantics are disabled.
    controlled_point = ControlledPointSpec(required=False, description=None)
    target_representation = _expect_enum(
        TargetRepresentation,
        mapping.get("target_representation"),
        "target_representation",
        errors,
    )

    raw_phases = mapping.get("phases")
    if not isinstance(raw_phases, list):
        errors.append("phases must be a JSON array.")
        raw_phases = []

    phases = tuple(
        _parse_phase(value, index, errors)
        for index, value in enumerate(raw_phases)
    )

    spec = VideoStructureSpec(
        controlled_entity=controlled_entity,
        target_entity=target_entity,
        controlled_point=controlled_point,
        target_representation=target_representation or TargetRepresentation.POINT,
        phases=phases,
    )

    _validate_spec(spec, total_frames=total_frames, errors=errors)

    if errors:
        raise SpecValidationError(errors)
    return spec
