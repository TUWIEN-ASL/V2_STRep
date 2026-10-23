#!/usr/bin/env python3
"""Run the motion-planning pipeline into one isolated output folder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_planning.full_pipeline import (
    PipelineInputError,
    load_json,
    run_pipeline,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build demo/current geometry, transfer the demonstrated motion, "
            "optimize the Panda joint trajectory, and export Cartesian poses."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace previous run outputs, keeping the active flow and grasp input folders.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all required paths/calibration without building geometry.",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Build through motion_reference.npz without importing PyRoKi/JAXLS.",
    )
    return parser.parse_args()


def add_controlled_object_frames(config: dict, base_dir: Path, manifest: dict) -> None:
    """Draw the saved demo/current object frames on the registration overlays."""
    registration_dir = manifest.get("controlled_object_pose_estimation", {}).get("artifacts")
    if registration_dir is None:
        return

    import cv2
    import numpy as np
    from PIL import Image

    from motion_planning.full_pipeline import load_camera_bundle, merge_mappings
    from motion_planning.target_pose_registration import _project_world_points, transform_points

    registration_dir = Path(registration_dir)
    summary = load_json(registration_dir / "controlled_object_pose_summary.json")
    axis_points = np.vstack((np.zeros(3), 0.05 * np.eye(3)))
    for prefix in ("demo", "current"):
        camera_config = merge_mappings(config.get("camera", {}), config.get(f"{prefix}_camera"))
        camera = load_camera_bundle(camera_config, base_dir)
        pose = np.asarray(summary[f"T_world_target_{prefix}"], dtype=np.float64)
        pixels, valid = _project_world_points(
            transform_points(pose, axis_points),
            camera.intrinsic_matrix,
            camera.transform_world_camera,
        )
        valid &= np.isfinite(pixels).all(axis=1)
        path = registration_dir / f"{prefix}_mask_overlay.png"
        with Image.open(path) as image:
            overlay = np.asarray(image.convert("RGB")).copy()
        if valid[0]:
            origin = tuple(np.rint(pixels[0]).astype(int))
            for index, (label, color) in enumerate(
                zip("XYZ", ((255, 0, 0), (0, 255, 0), (0, 0, 255))), start=1,
            ):
                if not valid[index]:
                    continue
                endpoint = tuple(np.rint(pixels[index]).astype(int))
                cv2.arrowedLine(overlay, origin, endpoint, (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.15)
                cv2.arrowedLine(overlay, origin, endpoint, color, 3, cv2.LINE_AA, tipLength=0.15)
                text_origin = (endpoint[0] + 5, endpoint[1] - 5)
                cv2.putText(overlay, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(overlay, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            cv2.circle(overlay, origin, 4, (255, 255, 255), -1, cv2.LINE_AA)
        caption = f"{prefix.title()} | object axes: 5 cm"
        cv2.putText(overlay, caption, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, caption, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        Image.fromarray(overlay).save(path)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"ERROR: configuration file does not exist: {config_path}", file=sys.stderr)
        return 2

    config = load_json(config_path)
    try:
        manifest = run_pipeline(
            config,
            config_path=config_path,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            check_only=args.check_only,
            skip_optimization=args.skip_optimization,
        )
    except Exception as error:
        input_error = isinstance(error, PipelineInputError)
        error_report = {
            "status": "input_error" if input_error else "failed",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        if input_error:
            error_report["missing"] = list(error.missing)
        else:
            error_report["traceback"] = traceback.format_exc()

        report_path = args.output_dir.resolve() / "trajectory" / "pipeline_error.json"
        save_json(report_path, error_report)
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        if error_report.get("missing"):
            print("Missing/required items:", file=sys.stderr)
            for item in error_report["missing"]:
                print(f"  - {item}", file=sys.stderr)
        print(f"Error report: {report_path}", file=sys.stderr)
        return 2 if input_error else 1

    try:
        add_controlled_object_frames(config, config_path.parent, manifest)
    except Exception as error:
        print(f"WARNING: controlled-object frame visualization failed: {error}", file=sys.stderr)

    print("Pipeline status:", manifest["status"])
    print("Run folder:", Path(manifest["output_dir"]).resolve())
    for warning in manifest.get("warnings", []):
        print("WARNING:", warning)
    for name, path in manifest.get("artifacts", {}).items():
        if name.startswith("execution_"):
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
