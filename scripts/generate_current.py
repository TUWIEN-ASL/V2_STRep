#!/usr/bin/env python3

import json
import argparse
from copy import deepcopy


def demo_to_current(demo):
    """
    Convert demo.json -> current.json for the special case where
    the current scene is exactly the same as the demonstration scene.
    """

    # ---------------------------------------------------------
    # Basic checks
    # ---------------------------------------------------------
    if "video_structure" not in demo:
        raise KeyError("demo.json missing 'video_structure'")

    if "demo_initial_grounding" not in demo:
        raise KeyError("demo.json missing 'demo_initial_grounding'")

    video_structure = demo["video_structure"]
    grounding = demo["demo_initial_grounding"]

    # Make sure target representation is internally consistent.
    demo_representation = video_structure["target_representation"]
    grounding_representation = grounding["target_geometry"]["representation"]

    if demo_representation != grounding_representation:
        raise ValueError(
            "Target representation mismatch:\n"
            f"  video_structure.target_representation = {demo_representation}\n"
            f"  demo_initial_grounding.target_geometry.representation = "
            f"{grounding_representation}"
        )

    # ---------------------------------------------------------
    # Convert phase structure -> current rotation policies
    # ---------------------------------------------------------
    phase_rotation_policies = []

    for phase in video_structure["phases"]:
        phase_rotation_policies.append(
            {
                "phase_id": phase["phase_id"],
                "rotation_policy": deepcopy(phase["rotation_policy"]),
            }
        )

    # ---------------------------------------------------------
    # Construct current.json using EXACT current schema
    # ---------------------------------------------------------
    current = {
        "task_instruction": demo["task_instruction"],

        "is_usable": grounding["is_usable"],

        "failure_reason": grounding["failure_reason"],

        "phase_rotation_policies": phase_rotation_policies,

        "current_grounding": {
            "controlled_entity": deepcopy(
                grounding["controlled_entity"]
            ),

            "allowed_grasp_region": deepcopy(
                grounding["allowed_grasp_region"]
            ),

            "target_entity": deepcopy(
                grounding["target_entity"]
            ),

            "target_geometry": deepcopy(
                grounding["target_geometry"]
            ),
        },
    }

    return current


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate current.json directly from demo.json when "
            "the current scene is identical to the demonstration scene."
        )
    )

    parser.add_argument(
        "demo_json",
        help="Path to demo.json",
    )

    parser.add_argument(
        "output_json",
        nargs="?",
        default="g_current.json",
        help="Output current.json path (default: g_current.json)",
    )

    args = parser.parse_args()

    # Load demo
    with open(args.demo_json, "r", encoding="utf-8") as f:
        demo = json.load(f)

    # Convert
    current = demo_to_current(demo)

    # Save
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            current,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Generated: {args.output_json}")
    print("Current scene grounding copied exactly from demo scene.")


if __name__ == "__main__":
    main()