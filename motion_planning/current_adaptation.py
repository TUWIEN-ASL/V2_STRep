from __future__ import annotations

"""Parsing and validation for the second VLM prompt.

The demonstration prompt owns the motion structure.  The current-scene prompt
may only replace per-phase rotation policies and provide current grounding.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .schemas import (
    PhaseSymmetryMode,
    RemainingRotation,
    RotationMode,
    VideoStructureSpec,
)
from .video_spec import SpecValidationError, parse_video_structure


class CurrentAdaptationValidationError(ValueError):
    """Raised when the current-adaptation JSON violates the two-prompt contract."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        message = "Invalid current adaptation:\n- " + "\n- ".join(self.errors)
        super().__init__(message)


@dataclass(frozen=True)
class CurrentAdaptation:
    task_instruction: str
    is_usable: bool
    failure_reason: Optional[str]
    current_grounding: Mapping[str, Any]
    effective_video_structure_payload: Mapping[str, Any]
    effective_spec: VideoStructureSpec
    policy_changes: tuple[Mapping[str, Any], ...]
    legacy_direct_grounding: bool = False


def _rotation_policy_payload(phase_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = phase_payload.get("rotation_policy")
    return policy if isinstance(policy, Mapping) else {}


def _normalise_for_compare(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _normalise_for_compare(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        return [_normalise_for_compare(item) for item in value]
    return value


def _validate_grounding_contract(
    grounding: Mapping[str, Any],
    demo_spec: VideoStructureSpec,
    errors: list[str],
) -> None:
    target_geometry = grounding.get("target_geometry")
    if not isinstance(target_geometry, Mapping):
        errors.append("current_grounding.target_geometry must be a JSON object.")
    elif target_geometry.get("representation") != demo_spec.target_representation.value:
        errors.append(
            "current_grounding.target_geometry.representation must equal the "
            f"demonstration target representation {demo_spec.target_representation.value!r}."
        )


def _validate_policy_transition(
    demo_phase_payload: Mapping[str, Any],
    current_phase_payload: Mapping[str, Any],
    *,
    phase_id: str,
    errors: list[str],
) -> None:
    demo_policy = _rotation_policy_payload(demo_phase_payload)
    current_policy = _rotation_policy_payload(current_phase_payload)

    demo_mode = demo_policy.get("mode")
    current_mode = current_policy.get("mode")
    if current_mode == RotationMode.FULL_DEMO.value and demo_mode != RotationMode.FULL_DEMO.value:
        errors.append(
            f"phase_rotation_policies[{phase_id!r}] cannot introduce full_demo; "
            "only a demonstration full_demo phase may retain full_demo."
        )

    demo_symmetry = demo_policy.get("phase_symmetry")
    current_symmetry = current_policy.get("phase_symmetry")
    demo_symmetry_mode = (
        demo_symmetry.get("mode") if isinstance(demo_symmetry, Mapping) else None
    )
    current_symmetry_mode = (
        current_symmetry.get("mode") if isinstance(current_symmetry, Mapping) else None
    )
    if (
        current_symmetry_mode == PhaseSymmetryMode.SHARED_OFFSET.value
        and demo_symmetry_mode != PhaseSymmetryMode.SHARED_OFFSET.value
    ):
        errors.append(
            f"phase_rotation_policies[{phase_id!r}] cannot introduce shared_offset."
        )

    # The current optimizer's fixed_from_demo term is a complete orientation
    # target.  It is safe only when the axis constraints themselves are kept.
    if current_policy.get("remaining_rotation") == RemainingRotation.FIXED_FROM_DEMO.value:
        demo_constraints = _normalise_for_compare(demo_policy.get("axis_constraints", []))
        current_constraints = _normalise_for_compare(
            current_policy.get("axis_constraints", [])
        )
        if demo_constraints != current_constraints:
            errors.append(
                f"phase_rotation_policies[{phase_id!r}] changes axis constraints "
                "while using remaining_rotation='fixed_from_demo'. Use 'free' or "
                "'none', or keep the demonstration constraints unchanged."
            )


def parse_current_adaptation(
    payload: Mapping[str, Any],
    demo_video_structure_payload: Mapping[str, Any],
    *,
    total_frames: Optional[int] = None,
    allow_legacy_direct_grounding: bool = True,
) -> CurrentAdaptation:
    """Parse Prompt-2 output and merge its policies into the demo structure.

    A legacy direct scene-grounding JSON is accepted when requested.  In that
    mode the demonstration rotation policies are kept unchanged.
    """
    if not isinstance(payload, Mapping):
        raise CurrentAdaptationValidationError(("root must be a JSON object.",))
    if not isinstance(demo_video_structure_payload, Mapping):
        raise CurrentAdaptationValidationError(
            ("demo_video_structure_payload must be a JSON object.",)
        )

    try:
        demo_spec = parse_video_structure(
            demo_video_structure_payload,
            total_frames=total_frames,
        )
    except SpecValidationError as error:
        raise CurrentAdaptationValidationError(
            tuple(f"demonstration video_structure: {item}" for item in error.errors)
        ) from error

    is_new_format = "current_grounding" in payload or "phase_rotation_policies" in payload
    if not is_new_format:
        if not allow_legacy_direct_grounding:
            raise CurrentAdaptationValidationError(
                ("Prompt-2 JSON is missing current_grounding and phase_rotation_policies.",)
            )
        return CurrentAdaptation(
            task_instruction="",
            is_usable=True,
            failure_reason=None,
            current_grounding=payload,
            effective_video_structure_payload=deepcopy(demo_video_structure_payload),
            effective_spec=demo_spec,
            policy_changes=(),
            legacy_direct_grounding=True,
        )

    errors: list[str] = []
    required_keys = {
        "task_instruction",
        "is_usable",
        "failure_reason",
        "phase_rotation_policies",
        "current_grounding",
    }
    # Accept obsolete correspondence data in older JSON files, but ignore it.
    allowed_keys = required_keys | {"relative_pose_correspondences"}
    missing = required_keys - set(payload.keys())
    unknown = set(payload.keys()) - allowed_keys
    for key in sorted(missing):
        errors.append(f"root is missing required field {key!r}.")
    for key in sorted(unknown):
        errors.append(f"root contains unsupported field {key!r}.")

    instruction = payload.get("task_instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        errors.append("task_instruction must be a non-empty string.")
        instruction = ""
    else:
        instruction = instruction.strip()

    is_usable = payload.get("is_usable")
    if not isinstance(is_usable, bool):
        errors.append("is_usable must be true or false.")
        is_usable = False

    failure_reason = payload.get("failure_reason")
    if failure_reason is not None and (
        not isinstance(failure_reason, str) or not failure_reason.strip()
    ):
        errors.append("failure_reason must be null or a non-empty string.")
        failure_reason = None
    elif isinstance(failure_reason, str):
        failure_reason = failure_reason.strip()

    if is_usable and failure_reason is not None:
        errors.append("failure_reason must be null when is_usable=true.")
    if not is_usable and failure_reason is None:
        errors.append("failure_reason is required when is_usable=false.")

    current_grounding = payload.get("current_grounding")
    if not isinstance(current_grounding, Mapping):
        errors.append("current_grounding must be a JSON object.")
        current_grounding = {}
    _validate_grounding_contract(current_grounding, demo_spec, errors)

    raw_overrides = payload.get("phase_rotation_policies")
    if not isinstance(raw_overrides, list):
        errors.append("phase_rotation_policies must be a JSON array.")
        raw_overrides = []

    demo_phases = demo_video_structure_payload.get("phases")
    if not isinstance(demo_phases, list):
        errors.append("demonstration video_structure.phases must be a JSON array.")
        demo_phases = []

    demo_by_id: dict[str, Mapping[str, Any]] = {}
    for phase in demo_phases:
        if isinstance(phase, Mapping) and isinstance(phase.get("phase_id"), str):
            demo_by_id[str(phase["phase_id"])] = phase

    override_by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(raw_overrides):
        path = f"phase_rotation_policies[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{path} must be a JSON object.")
            continue
        item_keys = set(item.keys())
        if item_keys != {"phase_id", "rotation_policy"}:
            for key in sorted({"phase_id", "rotation_policy"} - item_keys):
                errors.append(f"{path} is missing required field {key!r}.")
            for key in sorted(item_keys - {"phase_id", "rotation_policy"}):
                errors.append(f"{path} contains unsupported field {key!r}.")
        phase_id = item.get("phase_id")
        if not isinstance(phase_id, str) or not phase_id.strip():
            errors.append(f"{path}.phase_id must be a non-empty string.")
            continue
        phase_id = phase_id.strip()
        if phase_id in override_by_id:
            errors.append(f"{path}.phase_id duplicates {phase_id!r}.")
            continue
        if phase_id not in demo_by_id:
            errors.append(f"{path}.phase_id={phase_id!r} is not a demo phase.")
            continue
        rotation_policy = item.get("rotation_policy")
        if not isinstance(rotation_policy, Mapping):
            errors.append(f"{path}.rotation_policy must be a JSON object.")
            continue
        override_by_id[phase_id] = rotation_policy

    demo_ids = set(demo_by_id)
    override_ids = set(override_by_id)
    for phase_id in sorted(demo_ids - override_ids):
        errors.append(
            f"phase_rotation_policies is missing demonstration phase {phase_id!r}."
        )

    effective_payload = deepcopy(demo_video_structure_payload)
    effective_phases = effective_payload.get("phases", [])
    policy_changes: list[Mapping[str, Any]] = []
    if isinstance(effective_phases, list):
        for phase in effective_phases:
            if not isinstance(phase, dict):
                continue
            phase_id = phase.get("phase_id")
            if not isinstance(phase_id, str) or phase_id not in override_by_id:
                continue
            demo_policy = deepcopy(phase.get("rotation_policy"))
            phase["rotation_policy"] = deepcopy(override_by_id[phase_id])
            _validate_policy_transition(
                demo_by_id[phase_id],
                phase,
                phase_id=phase_id,
                errors=errors,
            )
            if _normalise_for_compare(demo_policy) != _normalise_for_compare(
                phase["rotation_policy"]
            ):
                policy_changes.append(
                    {
                        "phase_id": phase_id,
                        "demo_rotation_policy": demo_policy,
                        "current_rotation_policy": deepcopy(
                            phase["rotation_policy"]
                        ),
                    }
                )

    effective_spec = demo_spec
    try:
        effective_spec = parse_video_structure(
            effective_payload,
            total_frames=total_frames,
        )
    except SpecValidationError as error:
        errors.extend(f"effective video_structure: {item}" for item in error.errors)

    if errors:
        raise CurrentAdaptationValidationError(errors)

    return CurrentAdaptation(
        task_instruction=instruction,
        is_usable=bool(is_usable),
        failure_reason=failure_reason,
        current_grounding=current_grounding,
        effective_video_structure_payload=effective_payload,
        effective_spec=effective_spec,
        policy_changes=tuple(policy_changes),
        legacy_direct_grounding=False,
    )


__all__ = [
    "CurrentAdaptation",
    "CurrentAdaptationValidationError",
    "parse_current_adaptation",
]
