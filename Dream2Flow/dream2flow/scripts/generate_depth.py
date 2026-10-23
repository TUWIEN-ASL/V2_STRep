"""Generate full-video depth in a standalone process before flow tracking."""

import argparse
from pathlib import Path

import torch

from dream2flow.flow.depth.spatracker_depth_extractor import (
    SpaTrackerDepthExtractor,
    SpaTrackerDepthExtractorConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--video-path", type=Path, default=Path("rgb.mp4"))
    parser.add_argument("--output-path", type=Path, default=Path("depth_frames.pt"))
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    video_path = scene_dir / args.video_path
    output_path = scene_dir / args.output_path

    extractor = SpaTrackerDepthExtractor(
        SpaTrackerDepthExtractorConfig(), device=args.device
    )
    extractor.tracking_model.S_wind = args.window
    extractor.tracking_model.overlap = args.overlap

    with torch.inference_mode():
        depth_frames = extractor.extract_depth_from_file(
            scene_dir=str(scene_dir), video_path=str(video_path)
        )

    torch.save(depth_frames.detach().cpu(), output_path)
    print(f"[Depth] Saved {tuple(depth_frames.shape)} to {output_path}")


if __name__ == "__main__":
    main()
