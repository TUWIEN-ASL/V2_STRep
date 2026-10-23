# SpaTracker dependency patch

`spatracker_memory.patch` captures the existing local modifications to:

- `models/SpaTrackV2/models/tracker3D/TrackRefiner.py`
- `models/SpaTrackV2/models/vggt4track/models/aggregator.py`

It preserves the current files, including their formatting and checks. The memory
changes use feature-extraction chunks of 8 frames and retain the VGGT layer outputs
needed by the prediction heads. This file does not modify an installed dependency
until the patch is explicitly applied.

The base repository revisions and dependency locations are recorded in
`../dependencies.json`. Paths in that file are relative to the Dream2Flow folder.
Apply the patch once to a fresh SpaTracker checkout at the recorded commit:

```bash
git -C deps/SpaTrackerV2 apply ../../patches/spatracker_memory.patch
```

The current local SpaTracker checkout already contains these changes. Do not
apply the patch again to this checkout.

The recorded CoTracker commit identifies the copied local checkout. Runtime still
uses the existing Torch Hub loader; pinning that loader remains installation work.
Model revisions and the working Python environment are not recorded here yet.

Dream2Flow's original Git metadata is preserved locally in `.upstream_git/` and
ignored by the parent repository. The original upstream commit is recorded in
`../dependencies.json`; the modified Dream2Flow source belongs to the parent project.
