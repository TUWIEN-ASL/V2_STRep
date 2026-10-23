from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from .se3 import (
    as_transform,
    interpolate_pose,
    invert_transform,
    make_transform,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ObjectPoseDiagnostics:
    reference_frame: int
    reference_track_ids: NDArray[np.int64]
    pca_eigenvalues_ascending: FloatArray
    reference_z_dot_world_up: float
    valid_correspondence_count: NDArray[np.int64]
    inlier_count: NDArray[np.int64]
    fit_rms_m: FloatArray
    failed_frame_indices: Tuple[int, ...]
    temporal_median_window: int
    max_translation_step_m: float
    max_rotation_step_deg: float
    limited_frame_indices: Tuple[int, ...]
    raw_translation_step_m: FloatArray
    raw_rotation_step_deg: FloatArray
    final_translation_step_m: FloatArray
    final_rotation_step_deg: FloatArray
    stable_track_ids: NDArray[np.int64]
    track_valid_ratio: FloatArray
    track_median_residual_m: FloatArray
    track_weight: FloatArray
    temporal_refinement_enabled: bool
    temporal_refinement_success: bool
    temporal_refinement_cost: float
    prelimit_translation_step_m: FloatArray
    prelimit_rotation_step_deg: FloatArray
    effective_max_translation_step_m: float
    effective_max_rotation_step_deg: float

    @property
    def minor_axis_gap_ratio(self) -> float:
        """
        Relative separation between the two smallest PCA eigenvalues.

        Values near zero mean that PCA x/y are poorly distinguished. In this
        case, rotation about the longest PCA axis should normally remain free.
        """
        first, second, _ = self.pca_eigenvalues_ascending
        denominator = max(abs(second), 1e-12)
        return float(abs(second - first) / denominator)


@dataclass(frozen=True)
class ObjectPoseTrajectory:
    """
    Canonical object-pose trajectory.

    transforms[t] is T_world_object at video frame t.

    The object frame is defined once at the reference frame from PCA axes with
    explicit gravity/camera semantics, then transported rigidly through the
    video using tracked point correspondences.
    """

    transforms: FloatArray
    diagnostics: ObjectPoseDiagnostics


def _to_numpy(value: Any) -> np.ndarray:
    """Convert Torch/TensorDict-compatible values to NumPy."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _read_field(container: Any, key: str) -> Any:
    """
    Read a named field from dictionaries, TensorDicts, or TensorClass objects.

    Some TensorClass versions interpret container["position"] as tensor
    indexing and raise ValueError. Therefore attribute access, get(), and
    wrapped TensorDict access are attempted before generic string indexing.
    """
    if container is None:
        raise KeyError(f"Cannot read field {key!r} from None.")

    # TensorClass fields are commonly exposed as attributes.
    try:
        value = getattr(container, key)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    else:
        return value

    # Standard Python mappings.
    if isinstance(container, Mapping):
        try:
            return container[key]
        except (KeyError, TypeError, ValueError):
            pass

    # TensorDict commonly exposes get().
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            value = getter(key)
        except (KeyError, TypeError, AttributeError, ValueError):
            pass
        else:
            if value is not None:
                return value

    # TensorClass and some serialized TensorDict implementations wrap their
    # real storage in one of these attributes.
    for attribute_name in ("tensordict", "_tensordict"):
        try:
            internal = getattr(container, attribute_name)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue

        if internal is container:
            continue

        if isinstance(internal, Mapping):
            try:
                return internal[key]
            except (KeyError, TypeError, ValueError):
                pass

        internal_getter = getattr(internal, "get", None)
        if callable(internal_getter):
            try:
                value = internal_getter(key)
            except (KeyError, TypeError, AttributeError, ValueError):
                pass
            else:
                if value is not None:
                    return value

        try:
            return internal[key]
        except (
            KeyError,
            TypeError,
            AttributeError,
            ValueError,
            IndexError,
        ):
            pass

    # Final fallback for custom dict-like objects.
    try:
        return container[key]
    except (
        KeyError,
        TypeError,
        AttributeError,
        ValueError,
        IndexError,
    ) as error:
        raise KeyError(
            f"Could not find field {key!r} in "
            f"{type(container).__name__}."
        ) from error


def extract_object_flow_arrays(
    payload: Any,
) -> Tuple[FloatArray, BoolArray]:
    """
    Extract ObjectFlow positions and validity.

    Supported common layouts include:

        payload["flow"].tensordict["position"]
        payload["flow"]["position"]
        payload["position"]

    Returns:
        positions_world: shape (T, N, 3), float64
        valid_mask:      shape (T, N), bool
    """
    flow = (
        payload["flow"]
        if isinstance(payload, Mapping) and "flow" in payload
        else payload
    )

    candidates = [flow]

    try:
        flow_tensordict = getattr(flow, "tensordict")
    except (AttributeError, KeyError, TypeError, ValueError):
        flow_tensordict = None

    if flow_tensordict is not None:
        candidates.insert(0, flow_tensordict)

    last_error: Optional[Exception] = None

    for candidate in candidates:
        try:
            positions = _to_numpy(
                _read_field(candidate, "position")
            ).astype(
                np.float64,
                copy=False,
            )
            valid_mask = _to_numpy(
                _read_field(candidate, "valid_mask")
            ).astype(
                bool,
                copy=False,
            )
            break
        except (
            KeyError,
            AttributeError,
            TypeError,
            ValueError,
        ) as error:
            last_error = error
    else:
        raise KeyError(
            "Could not extract 'position' and 'valid_mask' "
            "from ObjectFlow."
        ) from last_error

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            "positions must have shape (T, N, 3), "
            f"got {positions.shape}."
        )

    if valid_mask.shape != positions.shape[:2]:
        raise ValueError(
            "valid_mask must have shape "
            f"{positions.shape[:2]}, got {valid_mask.shape}."
        )

    valid_mask = (
        valid_mask
        & np.all(np.isfinite(positions), axis=-1)
    )

    return positions, valid_mask


def load_object_flow(
    path: str | Path,
    *,
    return_initial_pixel_coords: bool = False,
) -> (
    Tuple[FloatArray, BoolArray]
    | tuple[FloatArray, BoolArray, FloatArray]
):
    """
    Load a Dream2Flow ObjectFlow .pt file.

    By default this preserves the historical two-value return.  When
    ``return_initial_pixel_coords=True``, also return the ObjectFlowResult
    initial query pixels with shape (N, 2), ordered exactly like the tracks.
    """
    file_path = Path(path)

    try:
        payload = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "torch.load could not import a class stored in the "
            "ObjectFlow file. Run this in the Dream2Flow environment "
            "and ensure the Dream2Flow repository is on PYTHONPATH."
        ) from error

    positions, valid_mask = extract_object_flow_arrays(payload)
    if not return_initial_pixel_coords:
        return positions, valid_mask

    initial_pixel_coords = _to_numpy(
        _read_field(payload, "initial_pixel_coords")
    ).astype(np.float64, copy=False)
    if (
        initial_pixel_coords.ndim != 2
        or initial_pixel_coords.shape != (positions.shape[1], 2)
    ):
        raise ValueError(
            "initial_pixel_coords must have shape "
            f"({positions.shape[1]}, 2), got {initial_pixel_coords.shape}."
        )
    return positions, valid_mask, initial_pixel_coords


def normalize(
    vector: ArrayLike,
    *,
    name: str = "vector",
) -> FloatArray:
    """Validate and normalize a 3D vector."""
    array = np.asarray(vector, dtype=np.float64)

    if array.shape != (3,):
        raise ValueError(
            f"{name} must have shape (3,), got {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    norm = float(np.linalg.norm(array))
    if norm < 1e-12:
        raise ValueError(
            f"{name} has near-zero norm."
        )

    return array / norm


def project_axis_to_plane(
    axis: ArrayLike,
    plane_normal: ArrayLike,
    *,
    eps: float = 1e-10,
) -> Optional[FloatArray]:
    """
    Project a 3D axis onto the plane perpendicular to plane_normal.

    Returns null-equivalent None when the projection is too small to define a
    reliable direction.
    """
    axis_array = normalize(axis, name="axis")
    normal_array = normalize(
        plane_normal,
        name="plane_normal",
    )

    projected = (
        axis_array
        - float(np.dot(axis_array, normal_array))
        * normal_array
    )

    norm = float(np.linalg.norm(projected))
    if norm < eps:
        return None

    return projected / norm


def choose_pca_x_sign(
    x_axis: ArrayLike,
    z_axis: ArrayLike,
    *,
    world_x: Sequence[float] = (1.0, 0.0, 0.0),
    world_y: Sequence[float] = (0.0, 1.0, 0.0),
) -> FloatArray:
    """
    Resolve the remaining PCA x-axis sign deterministically.

    After fixing the PCA z-axis sign, these two frames are still equivalent:

        [ x,  y, z]
        [-x, -y, z]

    This function projects world +x and world +y onto the plane perpendicular
    to object z. It chooses the projected world axis most strongly aligned
    with the raw PCA x-axis, then flips PCA x so the alignment is positive.

    This is more stable than always enforcing dot(x_object, world_x) > 0,
    because world x may be almost parallel to object z or nearly orthogonal
    to PCA x.
    """
    x = normalize(x_axis, name="x_axis")
    z = normalize(z_axis, name="z_axis")

    reference_candidates = []

    for name, world_axis in (
        ("world_x", world_x),
        ("world_y", world_y),
    ):
        projected = project_axis_to_plane(
            world_axis,
            z,
        )

        if projected is None:
            continue

        alignment = abs(float(np.dot(x, projected)))
        reference_candidates.append(
            (alignment, name, projected)
        )

    if not reference_candidates:
        raise ValueError(
            "Could not construct a stable world reference "
            "for the PCA x-axis sign."
        )

    _, _, selected_reference = max(
        reference_candidates,
        key=lambda item: item[0],
    )

    if float(np.dot(x, selected_reference)) < 0.0:
        x = -x

    return x


def compute_pca_object_pose(
    points_world: ArrayLike,
    *,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
    camera_position_world: Optional[ArrayLike] = None,
    world_x: Sequence[float] = (1.0, 0.0, 0.0),
    world_y: Sequence[float] = (0.0, 1.0, 0.0),
) -> Tuple[FloatArray, FloatArray]:
    """Compute one gravity/camera-semantic PCA object pose.

    PCA supplies three mutually orthogonal geometric directions, but their
    eigenvalue order is not used as the semantic axis order. Instead:

    * object +z is the PCA axis most aligned with world up, with positive sign
    selected toward world up;
    * object +x is the remaining PCA axis most aligned with the horizontal
    direction from the object centroid toward the camera, but with positive
    sign selected away from the camera;
    * object +y = object +z cross object +x, followed by re-orthogonalization
    of object +x as object +y cross object +z.

    When no camera position is supplied, a projected world-axis direction is
    used as a deterministic compatibility fallback. The frame must be computed
    once at the initial/reference pose and transported rigidly; it must not be
    recomputed independently at every video frame.

    Returns:
        T_world_object: shape (4, 4)
        eigenvalues: ascending shape (3,)
    """
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "points_world must have shape (N, 3), "
            f"got {points.shape}."
        )
    if points.shape[0] < 3:
        raise ValueError("At least three points are required for PCA.")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_world contains non-finite values.")

    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / points.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    up = normalize(world_up, name="world_up")
    up_alignment = np.abs(eigenvectors.T @ up)
    z_index = int(np.argmax(up_alignment))
    z_axis = eigenvectors[:, z_index]
    if float(np.dot(z_axis, up)) < 0.0:
        z_axis = -z_axis
    z_axis = normalize(z_axis, name="z_axis")

    remaining_indices = [index for index in range(3) if index != z_index]

    camera_reference: Optional[FloatArray] = None
    if camera_position_world is not None:
        camera_position = np.asarray(camera_position_world, dtype=np.float64)
        if camera_position.shape != (3,) or not np.all(np.isfinite(camera_position)):
            raise ValueError(
                "camera_position_world must be a finite vector with shape (3,)."
            )
        camera_reference = project_axis_to_plane(
            camera_position - centroid,
            z_axis,
        )

    if camera_reference is None:
        fallback_candidates: list[FloatArray] = []
        for candidate in (world_x, world_y):
            projected = project_axis_to_plane(candidate, z_axis)
            if projected is not None:
                fallback_candidates.append(projected)
        if not fallback_candidates:
            raise ValueError(
                "Could not construct a camera/world gauge direction perpendicular "
                "to object z."
            )
        camera_reference = max(
            fallback_candidates,
            key=lambda direction: max(
                abs(float(np.dot(eigenvectors[:, index], direction)))
                for index in remaining_indices
            ),
        )

    x_index = max(
        remaining_indices,
        key=lambda index: abs(
            float(np.dot(eigenvectors[:, index], camera_reference))
        ),
    )

    x_axis = eigenvectors[:, x_index]

    # +x points AWAY from the camera.
    if float(np.dot(x_axis, camera_reference)) > 0.0:
        x_axis = -x_axis

    x_axis = normalize(x_axis, name="x_axis")

    # +y = +z cross +x
    y_axis = normalize(
        np.cross(z_axis, x_axis),
        name="y_axis",
    )

    # Re-orthogonalize x.
    x_axis = normalize(
        np.cross(y_axis, z_axis),
        name="x_axis",
    )


    
    rotation_world_object = np.column_stack((x_axis, y_axis, z_axis))

    if float(np.linalg.det(rotation_world_object)) < 0.0:
        raise RuntimeError("Constructed object frame is not right-handed.")

    return make_transform(rotation_world_object, centroid), eigenvalues

def points_world_to_object(
    transform_world_object: ArrayLike,
    points_world: ArrayLike,
) -> FloatArray:
    """
    Convert row-wise 3D points from world coordinates to object coordinates.
    """
    transform_object_world = invert_transform(
        transform_world_object
    )

    points = np.asarray(
        points_world,
        dtype=np.float64,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "points_world must have shape (N, 3), "
            f"got {points.shape}."
        )

    rotation = transform_object_world[:3, :3]
    translation = transform_object_world[:3, 3]

    return (
        points
        @ rotation.T
        + translation
    )


def transform_points(
    transform_target_source: ArrayLike,
    points_source: ArrayLike,
) -> FloatArray:
    """
    Transform row-wise points with T_target_source.
    """
    transform = as_transform(
        transform_target_source,
        copy=False,
    )

    points = np.asarray(
        points_source,
        dtype=np.float64,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "points_source must have shape (N, 3), "
            f"got {points.shape}."
        )

    return (
        points
        @ transform[:3, :3].T
        + transform[:3, 3]
    )


def fit_rigid_transform_kabsch(
    points_object: ArrayLike,
    points_world: ArrayLike,
) -> FloatArray:
    """
    Fit T_world_object from corresponding 3D points.

    Solves:

        p_world ~= R_world_object @ p_object
                   + t_world_object
    """
    source = np.asarray(
        points_object,
        dtype=np.float64,
    )

    target = np.asarray(
        points_world,
        dtype=np.float64,
    )

    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != 3
    ):
        raise ValueError(
            "Kabsch inputs must have equal shape (N, 3), "
            f"got {source.shape} and {target.shape}."
        )

    if source.shape[0] < 3:
        raise ValueError(
            "At least three correspondences are required."
        )

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)

    source_zero = source - source_center
    target_zero = target - target_center

    if np.linalg.matrix_rank(source_zero) < 2:
        raise ValueError(
            "Object-frame correspondences are degenerate."
        )

    if np.linalg.matrix_rank(target_zero) < 2:
        raise ValueError(
            "World-frame correspondences are degenerate."
        )

    covariance = source_zero.T @ target_zero

    left, _, right_transpose = np.linalg.svd(
        covariance
    )

    rotation = right_transpose.T @ left.T

    # Prevent reflection.
    if float(np.linalg.det(rotation)) < 0.0:
        right_transpose[-1, :] *= -1.0
        rotation = right_transpose.T @ left.T

    translation = (
        target_center
        - rotation @ source_center
    )

    return make_transform(
        rotation,
        translation,
    )


def fit_rigid_transform_ransac(
    points_object: ArrayLike,
    points_world: ArrayLike,
    *,
    inlier_threshold_m: float = 0.01,
    iterations: int = 256,
    random_seed: int = 0,
) -> Tuple[FloatArray, BoolArray, FloatArray]:
    """
    Robustly fit T_world_object using three-point RANSAC followed by Kabsch
    refinement.

    Returns:
        transform_world_object
        inlier_mask
        residuals_m
    """
    source = np.asarray(
        points_object,
        dtype=np.float64,
    )

    target = np.asarray(
        points_world,
        dtype=np.float64,
    )

    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != 3
    ):
        raise ValueError(
            "RANSAC inputs must have equal shape (N, 3), "
            f"got {source.shape} and {target.shape}."
        )

    count = source.shape[0]

    if count < 3:
        raise ValueError(
            "At least three correspondences are required."
        )

    if count == 3:
        transform = fit_rigid_transform_kabsch(
            source,
            target,
        )

        residuals = np.linalg.norm(
            transform_points(
                transform,
                source,
            )
            - target,
            axis=1,
        )

        return (
            transform,
            np.ones(count, dtype=bool),
            residuals,
        )

    rng = np.random.default_rng(
        random_seed
    )

    all_indices = np.arange(count)

    best_inliers: Optional[BoolArray] = None
    best_count = -1
    best_mean_error = np.inf

    for _ in range(iterations):
        sample_indices = rng.choice(
            all_indices,
            size=3,
            replace=False,
        )

        try:
            transform = fit_rigid_transform_kabsch(
                source[sample_indices],
                target[sample_indices],
            )
        except ValueError:
            continue

        residuals = np.linalg.norm(
            transform_points(
                transform,
                source,
            )
            - target,
            axis=1,
        )

        inliers = (
            residuals
            <= inlier_threshold_m
        )

        inlier_count = int(
            inliers.sum()
        )

        if inlier_count < 3:
            continue

        mean_error = float(
            residuals[inliers].mean()
        )

        if (
            inlier_count > best_count
            or (
                inlier_count == best_count
                and mean_error < best_mean_error
            )
        ):
            best_inliers = inliers
            best_count = inlier_count
            best_mean_error = mean_error

    if best_inliers is None:
        best_inliers = np.ones(
            count,
            dtype=bool,
        )

    transform = fit_rigid_transform_kabsch(
        source[best_inliers],
        target[best_inliers],
    )

    residuals = np.linalg.norm(
        transform_points(
            transform,
            source,
        )
        - target,
        axis=1,
    )

    refined_inliers = (
        residuals
        <= inlier_threshold_m
    )

    if int(refined_inliers.sum()) >= 3:
        transform = fit_rigid_transform_kabsch(
            source[refined_inliers],
            target[refined_inliers],
        )

        residuals = np.linalg.norm(
            transform_points(
                transform,
                source,
            )
            - target,
            axis=1,
        )

        best_inliers = refined_inliers

    return (
        transform,
        best_inliers,
        residuals,
    )


@dataclass(frozen=True)
class ObjectRegistrationDiagnostics:
    """Diagnostics for mesh-free demo-to-current point-cloud registration."""

    converged: bool
    iterations: int
    correspondence_count: int
    initial_rms_m: float
    final_rms_m: float
    selected_yaw_hypothesis_deg: float


def _rotation_about_local_z(angle_rad: float) -> FloatArray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rotation_delta_rad(rotation_a: FloatArray, rotation_b: FloatArray) -> float:
    relative = np.asarray(rotation_a).T @ np.asarray(rotation_b)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def register_object_points_trimmed_icp(
    points_object_demo: ArrayLike,
    points_world_current: ArrayLike,
    initial_transform_world_object: ArrayLike,
    *,
    max_iterations: int = 40,
    max_correspondence_distance_m: float = 0.04,
    trim_fraction: float = 0.7,
    minimum_correspondences: int = 30,
    yaw_hypotheses_deg: Sequence[float] = (0.0, 90.0, -90.0, 180.0),
    convergence_translation_m: float = 1e-5,
    convergence_rotation_rad: float = 1e-4,
) -> Tuple[FloatArray, ObjectRegistrationDiagnostics]:
    """Register a mesh-free demo cloud to the current RGB-D cloud.

    The demo points are expressed in the canonical object frame. A camera-gauge
    PCA pose provides the initial world pose. Several rotations about local
    object z are tried because this degree of freedom can be weakly observable.
    Each hypothesis is refined by nearest-neighbour trimmed ICP with Kabsch
    updates. The lowest final RMS solution is returned.

    This routine is intended as a lightweight classical-CV initialization. It
    does not require a CAD mesh and safely falls back at the caller when the
    fit is poor.
    """
    from scipy.spatial import cKDTree

    source = np.asarray(points_object_demo, dtype=np.float64)
    target = np.asarray(points_world_current, dtype=np.float64)
    initial = as_transform(initial_transform_world_object)

    if source.ndim != 2 or source.shape[1] != 3 or source.shape[0] < 3:
        raise ValueError("points_object_demo must have shape (N,3), N>=3.")
    if target.ndim != 2 or target.shape[1] != 3 or target.shape[0] < 3:
        raise ValueError("points_world_current must have shape (M,3), M>=3.")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("Registration point clouds contain non-finite values.")
    if not 0.0 < trim_fraction <= 1.0:
        raise ValueError("trim_fraction must lie in (0,1].")
    if max_correspondence_distance_m <= 0.0:
        raise ValueError("max_correspondence_distance_m must be positive.")

    # Deterministically cap cost for dense RGB-D clouds.
    if source.shape[0] > 4000:
        source = source[np.linspace(0, source.shape[0] - 1, 4000).astype(int)]
    if target.shape[0] > 8000:
        target = target[np.linspace(0, target.shape[0] - 1, 8000).astype(int)]

    tree = cKDTree(target)
    best: Optional[Tuple[float, FloatArray, ObjectRegistrationDiagnostics]] = None

    for yaw_deg in yaw_hypotheses_deg:
        candidate = initial.copy()
        candidate[:3, :3] = (
            initial[:3, :3]
            @ _rotation_about_local_z(np.deg2rad(float(yaw_deg)))
        )

        initial_transformed = transform_points(candidate, source)
        initial_distances, _ = tree.query(initial_transformed, k=1)
        initial_finite = initial_distances[np.isfinite(initial_distances)]
        initial_rms = float(
            np.sqrt(np.mean(initial_finite**2))
            if initial_finite.size
            else np.inf
        )

        converged = False
        used_count = 0
        completed_iterations = 0

        for iteration in range(max_iterations):
            transformed = transform_points(candidate, source)
            distances, indices = tree.query(transformed, k=1)
            eligible = np.flatnonzero(
                np.isfinite(distances)
                & (distances <= max_correspondence_distance_m)
            )
            if eligible.size < minimum_correspondences:
                break

            order = eligible[np.argsort(distances[eligible])]
            keep_count = max(
                minimum_correspondences,
                int(np.floor(trim_fraction * order.size)),
            )
            selected = order[: min(keep_count, order.size)]
            used_count = int(selected.size)

            updated = fit_rigid_transform_kabsch(
                source[selected],
                target[indices[selected]],
            )
            translation_delta = float(
                np.linalg.norm(updated[:3, 3] - candidate[:3, 3])
            )
            rotation_delta = _rotation_delta_rad(
                candidate[:3, :3],
                updated[:3, :3],
            )
            candidate = updated
            completed_iterations = iteration + 1

            if (
                translation_delta <= convergence_translation_m
                and rotation_delta <= convergence_rotation_rad
            ):
                converged = True
                break

        transformed = transform_points(candidate, source)
        distances, _ = tree.query(transformed, k=1)
        valid = distances[
            np.isfinite(distances)
            & (distances <= max_correspondence_distance_m)
        ]
        final_rms = float(
            np.sqrt(np.mean(valid**2))
            if valid.size >= minimum_correspondences
            else np.inf
        )
        diagnostics = ObjectRegistrationDiagnostics(
            converged=converged,
            iterations=completed_iterations,
            correspondence_count=int(valid.size),
            initial_rms_m=initial_rms,
            final_rms_m=final_rms,
            selected_yaw_hypothesis_deg=float(yaw_deg),
        )
        score = final_rms
        if best is None or score < best[0]:
            best = (score, candidate, diagnostics)

    if best is None or not np.isfinite(best[0]):
        raise ValueError("Mesh-free object registration failed for all hypotheses.")
    return best[1], best[2]


def _choose_reference_frame(
    valid_mask: BoolArray,
    requested_frame: Optional[int],
) -> int:
    """
    Choose the PCA reference frame.

    When not specified, use frame 0 whenever it has at least three valid
    tracks. This keeps the canonical object frame tied to the demonstrated
    initial scene. Fall back to the earliest frame with the largest valid count
    only when frame 0 is unusable.
    """
    frame_count = valid_mask.shape[0]

    if requested_frame is not None:
        if not 0 <= requested_frame < frame_count:
            raise ValueError(
                f"reference_frame={requested_frame} is outside "
                f"[0, {frame_count - 1}]."
            )

        if int(valid_mask[requested_frame].sum()) < 3:
            raise ValueError(
                f"reference_frame={requested_frame} has fewer "
                "than three valid points."
            )

        return requested_frame

    if int(valid_mask[0].sum()) >= 3:
        return 0

    return int(np.argmax(valid_mask.sum(axis=1)))


def _interpolate_failed_transforms(
    transforms: FloatArray,
    valid_pose_mask: BoolArray,
) -> FloatArray:
    """
    Fill failed frames while preserving the original video indexing.
    """
    valid_indices = np.flatnonzero(
        valid_pose_mask
    )

    if valid_indices.size == 0:
        raise ValueError(
            "No object pose could be estimated."
        )

    result = transforms.copy()

    first_valid = int(
        valid_indices[0]
    )

    last_valid = int(
        valid_indices[-1]
    )

    # Copy the nearest valid pose for missing leading/trailing frames.
    result[:first_valid] = result[first_valid]
    result[last_valid + 1 :] = result[last_valid]

    # Interpolate internal gaps.
    for left_raw, right_raw in zip(
        valid_indices[:-1],
        valid_indices[1:],
    ):
        left = int(left_raw)
        right = int(right_raw)

        if right - left <= 1:
            continue

        for frame in range(
            left + 1,
            right,
        ):
            fraction = (
                (frame - left)
                / (right - left)
            )

            result[frame] = interpolate_pose(
                result[left],
                result[right],
                float(fraction),
            )

    return result


def _temporal_median_filter_positions(
    positions: FloatArray,
    valid_mask: BoolArray,
    *,
    window: int,
) -> FloatArray:
    """Denoise valid 3D tracks without inventing missing observations."""
    width = int(window)
    if width <= 1:
        return np.asarray(positions, dtype=np.float64).copy()
    if width % 2 == 0:
        raise ValueError("temporal_median_window must be odd.")

    source = np.asarray(positions, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    result = source.copy()
    radius = width // 2
    frame_count, track_count = valid.shape

    for frame in range(frame_count):
        left = max(0, frame - radius)
        right = min(frame_count, frame + radius + 1)
        for track in range(track_count):
            # Keep the original invalid-mask semantics.  We smooth a measured
            # point only; we do not create a new track sample where none existed.
            if not valid[frame, track]:
                continue
            local_valid = valid[left:right, track]
            if not np.any(local_valid):
                continue
            values = source[left:right, track][local_valid]
            result[frame, track] = np.median(values, axis=0)

    return result


def _pose_step_sizes(
    transforms: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return adjacent translation (m) and geodesic rotation (deg) steps."""
    poses = np.asarray(transforms, dtype=np.float64)
    count = poses.shape[0]
    translation = np.zeros(count, dtype=np.float64)
    rotation_deg = np.zeros(count, dtype=np.float64)

    for frame in range(1, count):
        previous = poses[frame - 1]
        current = poses[frame]
        translation[frame] = float(
            np.linalg.norm(current[:3, 3] - previous[:3, 3])
        )
        relative_rotation = previous[:3, :3].T @ current[:3, :3]
        cosine = float((np.trace(relative_rotation) - 1.0) / 2.0)
        angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
        rotation_deg[frame] = float(np.degrees(angle))

    return translation, rotation_deg


def _limit_pose_steps_from_anchor(
    transforms: FloatArray,
    *,
    anchor_index: int,
    max_translation_step_m: float,
    max_rotation_step_deg: float,
) -> tuple[FloatArray, Tuple[int, ...]]:
    """Clamp adjacent SE(3) changes while preserving one exact anchor pose.

    The reference frame remains unchanged.  Frames after it are limited
    forward; frames before it are limited backward.  This guarantees every
    adjacent output step respects both configured bounds.
    """
    result = np.asarray(transforms, dtype=np.float64).copy()
    count = result.shape[0]
    anchor = int(anchor_index)
    if not 0 <= anchor < count:
        raise ValueError("anchor_index is outside the trajectory.")

    translation_limit = float(max_translation_step_m)
    rotation_limit_deg = float(max_rotation_step_deg)
    if translation_limit <= 0.0 or not np.isfinite(translation_limit):
        raise ValueError("max_translation_step_m must be finite and positive.")
    if rotation_limit_deg <= 0.0 or not np.isfinite(rotation_limit_deg):
        raise ValueError("max_rotation_step_deg must be finite and positive.")
    rotation_limit_rad = float(np.deg2rad(rotation_limit_deg))
    limited: list[int] = []

    def limited_pose(neighbor: FloatArray, target: FloatArray) -> tuple[FloatArray, bool]:
        delta_translation = target[:3, 3] - neighbor[:3, 3]
        translation_step = float(np.linalg.norm(delta_translation))
        relative_rotation = neighbor[:3, :3].T @ target[:3, :3]
        cosine = float((np.trace(relative_rotation) - 1.0) / 2.0)
        rotation_step = float(np.arccos(np.clip(cosine, -1.0, 1.0)))

        translation_fraction = 1.0
        rotation_fraction = 1.0
        if translation_step > translation_limit:
            translation_fraction = translation_limit / translation_step
        if rotation_step > rotation_limit_rad:
            rotation_fraction = rotation_limit_rad / rotation_step

        if translation_fraction >= 1.0 and rotation_fraction >= 1.0:
            return target, False

        # Clamp translation and rotation independently.  A noisy angular jump
        # must not unnecessarily slow down otherwise valid translational motion.
        limited = interpolate_pose(
            neighbor,
            target,
            float(rotation_fraction),
        )
        limited[:3, 3] = (
            neighbor[:3, 3]
            + float(translation_fraction) * delta_translation
        )
        return limited, True

    # Preserve the canonical/reference pose exactly and propagate outward.
    for frame in range(anchor + 1, count):
        result[frame], changed = limited_pose(result[frame - 1], result[frame])
        if changed:
            limited.append(frame)

    for frame in range(anchor - 1, -1, -1):
        result[frame], changed = limited_pose(result[frame + 1], result[frame])
        if changed:
            limited.append(frame)

    return result, tuple(sorted(set(limited)))



def fit_rigid_transform_weighted_kabsch(
    points_object: ArrayLike,
    points_world: ArrayLike,
    weights: ArrayLike,
) -> FloatArray:
    """Weighted Kabsch fit for corresponding 3D points."""
    source = np.asarray(points_object, dtype=np.float64)
    target = np.asarray(points_world, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            "Weighted Kabsch inputs must have equal shape (N, 3), got "
            f"{source.shape} and {target.shape}."
        )
    if len(source) < 3 or weight.shape != (len(source),):
        raise ValueError("Weighted Kabsch requires at least 3 points and N weights.")
    if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("Weighted Kabsch weights must be finite and non-negative.")

    total = float(np.sum(weight))
    if total <= 1e-12:
        raise ValueError("Weighted Kabsch received near-zero total weight.")

    normalized = weight / total
    source_center = np.sum(source * normalized[:, None], axis=0)
    target_center = np.sum(target * normalized[:, None], axis=0)
    source_zero = source - source_center
    target_zero = target - target_center

    weighted_source = source_zero * np.sqrt(normalized[:, None])
    weighted_target = target_zero * np.sqrt(normalized[:, None])
    if np.linalg.matrix_rank(weighted_source) < 2:
        raise ValueError("Weighted object-frame correspondences are degenerate.")
    if np.linalg.matrix_rank(weighted_target) < 2:
        raise ValueError("Weighted world-frame correspondences are degenerate.")

    covariance = (source_zero * normalized[:, None]).T @ target_zero
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if float(np.linalg.det(rotation)) < 0.0:
        right_transpose[-1, :] *= -1.0
        rotation = right_transpose.T @ left.T

    translation = target_center - rotation @ source_center
    return make_transform(rotation, translation)


def _fit_rigid_transform_weighted_irls(
    points_object: FloatArray,
    points_world: FloatArray,
    base_weights: FloatArray,
    *,
    initial_transform: FloatArray,
    huber_delta_m: float,
    iterations: int = 4,
) -> tuple[FloatArray, FloatArray]:
    """Refine one pose with globally stable track weights + local robust IRLS."""
    source = np.asarray(points_object, dtype=np.float64)
    target = np.asarray(points_world, dtype=np.float64)
    base = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    transform = as_transform(initial_transform).copy()
    delta = max(float(huber_delta_m), 1e-6)

    for _ in range(max(1, int(iterations))):
        residual = np.linalg.norm(transform_points(transform, source) - target, axis=1)
        robust = np.ones_like(residual)
        large = residual > delta
        robust[large] = delta / np.maximum(residual[large], 1e-12)
        weight = base * robust
        if int(np.count_nonzero(weight > 1e-6)) < 3:
            break
        try:
            transform = fit_rigid_transform_weighted_kabsch(source, target, weight)
        except (ValueError, np.linalg.LinAlgError):
            break

    residual = np.linalg.norm(transform_points(transform, source) - target, axis=1)
    return transform, residual


def _compute_global_track_reliability(
    transforms: FloatArray,
    *,
    reference_track_ids: NDArray[np.int64],
    reference_points_object: FloatArray,
    positions_world: FloatArray,
    valid_mask: BoolArray,
    minimum_valid_ratio: float,
    mad_scale: float,
    minimum_stable_tracks: int,
) -> tuple[NDArray[np.int64], FloatArray, FloatArray, FloatArray]:
    """Estimate task-independent per-track reliability over the whole video."""
    poses = np.asarray(transforms, dtype=np.float64)
    track_ids = np.asarray(reference_track_ids, dtype=np.int64)
    frame_count = poses.shape[0]
    track_count = len(track_ids)

    residual_matrix = np.full((frame_count, track_count), np.nan, dtype=np.float64)
    for frame in range(frame_count):
        valid_local = valid_mask[frame, track_ids]
        if not np.any(valid_local):
            continue
        predicted = transform_points(poses[frame], reference_points_object[valid_local])
        observed = positions_world[frame, track_ids[valid_local]]
        residual_matrix[frame, valid_local] = np.linalg.norm(predicted - observed, axis=1)

    track_valid_ratio = np.mean(np.isfinite(residual_matrix), axis=0)
    track_median_residual = np.full(track_count, np.inf, dtype=np.float64)
    for local_index in range(track_count):
        values = residual_matrix[:, local_index]
        values = values[np.isfinite(values)]
        if len(values) > 0:
            track_median_residual[local_index] = float(np.median(values))

    finite = np.isfinite(track_median_residual)
    candidate = finite & (track_valid_ratio >= float(minimum_valid_ratio))
    candidate_values = track_median_residual[candidate]

    if len(candidate_values) > 0:
        center = float(np.median(candidate_values))
        mad = float(np.median(np.abs(candidate_values - center)))
        robust_sigma = 1.4826 * mad
        if robust_sigma < 1e-6:
            residual_threshold = max(center * 2.0, center + 0.002)
        else:
            residual_threshold = center + float(mad_scale) * robust_sigma
        stable_local = candidate & (track_median_residual <= residual_threshold)
    else:
        stable_local = np.zeros(track_count, dtype=bool)

    # Never let the global filter collapse the geometry to <3 correspondences.
    requested_minimum = min(track_count, max(3, int(minimum_stable_tracks)))
    if int(stable_local.sum()) < requested_minimum:
        eligible = np.flatnonzero(finite)
        if len(eligible) > 0:
            order = eligible[np.argsort(track_median_residual[eligible])]
            stable_local[order[:requested_minimum]] = True

    stable_indices = np.flatnonzero(stable_local)
    if len(stable_indices) < 3:
        # Final fail-safe: keep every reference track rather than breaking pose fitting.
        stable_indices = np.arange(track_count, dtype=np.int64)
        stable_local[:] = True

    # Soft reliability weights. Visibility matters, but geometric consistency matters more.
    finite_stable_residual = track_median_residual[stable_local & finite]
    residual_scale = (
        float(np.median(finite_stable_residual))
        if len(finite_stable_residual) > 0
        else 0.005
    )
    residual_scale = max(residual_scale, 1e-4)
    weights = np.zeros(track_count, dtype=np.float64)
    normalized_residual = track_median_residual / residual_scale
    weights[finite] = (
        track_valid_ratio[finite]
        / (1.0 + normalized_residual[finite] ** 2)
    )
    weights[~stable_local] = 0.0
    if float(np.max(weights)) > 0.0:
        weights /= float(np.max(weights))
    weights[stable_local] = np.maximum(weights[stable_local], 0.05)

    return (
        track_ids[stable_indices].astype(np.int64),
        track_valid_ratio,
        track_median_residual,
        weights,
    )


def _refit_with_stable_tracks(
    initial_transforms: FloatArray,
    *,
    reference_index: int,
    reference_track_ids: NDArray[np.int64],
    reference_points_object: FloatArray,
    stable_track_ids: NDArray[np.int64],
    track_weights_local: FloatArray,
    positions_world: FloatArray,
    valid_mask: BoolArray,
    huber_delta_m: float,
    irls_iterations: int,
) -> tuple[FloatArray, BoolArray, NDArray[np.int64], FloatArray]:
    """Second-pass pose fit using a temporally consistent set of reliable tracks."""
    result = np.asarray(initial_transforms, dtype=np.float64).copy()
    frame_count = len(result)
    success = np.zeros(frame_count, dtype=bool)
    counts = np.zeros(frame_count, dtype=np.int64)
    rms = np.full(frame_count, np.nan, dtype=np.float64)

    # Map global track id -> local canonical-point index / reliability weight.
    local_of_track = {int(track): i for i, track in enumerate(reference_track_ids)}
    selected_local = np.asarray(
        [local_of_track[int(track)] for track in stable_track_ids], dtype=np.int64
    )
    selected_object = reference_points_object[selected_local]
    selected_weights = track_weights_local[selected_local]

    for frame in range(frame_count):
        observed_valid = valid_mask[frame, stable_track_ids]
        count = int(observed_valid.sum())
        counts[frame] = count
        if count < 3:
            continue
        if frame == int(reference_index):
            success[frame] = True
            rms[frame] = 0.0
            continue

        source = selected_object[observed_valid]
        target = positions_world[frame, stable_track_ids[observed_valid]]
        weights = selected_weights[observed_valid]
        try:
            transform, residual = _fit_rigid_transform_weighted_irls(
                source,
                target,
                weights,
                initial_transform=result[frame],
                huber_delta_m=huber_delta_m,
                iterations=irls_iterations,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        result[frame] = transform
        success[frame] = True
        rms[frame] = float(np.sqrt(np.mean(residual ** 2)))

    # Preserve the canonical anchor exactly.
    result[reference_index] = initial_transforms[reference_index]
    success[reference_index] = True
    rms[reference_index] = 0.0
    return result, success, counts, rms


def _rotation_increment_world(rotation_a: FloatArray, rotation_b: FloatArray) -> FloatArray:
    """World-frame rotation vector taking R_a to R_b."""
    local = Rotation.from_matrix(rotation_a.T @ rotation_b).as_rotvec()
    return rotation_a @ local


def _solve_regularized_sequence(
    values: FloatArray,
    observation_weight: FloatArray,
    *,
    observation_sigma: float,
    smoothness_sigma: float,
    difference_order: int,
    anchor_index: Optional[int] = None,
) -> FloatArray:
    """Small dense regularized least-squares solve for T x D sequences."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("values must have shape (T, D).")
    count = len(y)
    if count <= 2:
        return y.copy()

    weight = np.asarray(observation_weight, dtype=np.float64).reshape(-1)
    if weight.shape != (count,):
        raise ValueError("observation_weight must have shape (T,).")
    weight = np.clip(weight, 0.01, 1.0)

    obs_sigma = max(float(observation_sigma), 1e-8)
    smooth_sigma = max(float(smoothness_sigma), 1e-8)
    W = np.diag(weight / (obs_sigma * obs_sigma))

    if difference_order == 1:
        D = np.zeros((count - 1, count), dtype=np.float64)
        rows = np.arange(count - 1)
        D[rows, rows] = -1.0
        D[rows, rows + 1] = 1.0
    elif difference_order == 2:
        D = np.zeros((count - 2, count), dtype=np.float64)
        rows = np.arange(count - 2)
        D[rows, rows] = 1.0
        D[rows, rows + 1] = -2.0
        D[rows, rows + 2] = 1.0
    else:
        raise ValueError("difference_order must be 1 or 2.")

    A = W + (D.T @ D) / (smooth_sigma * smooth_sigma)
    B = W @ y

    if anchor_index is None:
        return np.linalg.solve(A, B)

    # Exact equality constraint x[anchor] = y[anchor] via a KKT system.
    anchor = int(anchor_index)
    if not 0 <= anchor < count:
        raise ValueError("anchor_index is outside the sequence.")
    c = np.zeros((count, 1), dtype=np.float64)
    c[anchor, 0] = 1.0
    K = np.block(
        [
            [A, c],
            [c.T, np.zeros((1, 1), dtype=np.float64)],
        ]
    )
    rhs = np.vstack([B, y[anchor][None]])
    solution = np.linalg.solve(K, rhs)[:count]
    solution[anchor] = y[anchor]
    return solution


def _temporally_refine_pose_trajectory(
    initial_transforms: FloatArray,
    *,
    anchor_index: int,
    frame_confidence: FloatArray,
    translation_observation_sigma_m: float,
    translation_acceleration_sigma_m: float,
    rotation_observation_sigma_deg: float,
    rotation_acceleration_sigma_deg: float,
) -> tuple[FloatArray, bool, float]:
    """Task-independent smoothing that preserves large downsampled motion.

    Translation is fit to the robust per-frame poses while penalizing only
    *second differences* (linear acceleration). Rotation is handled in the
    same spirit by smoothing world-frame incremental rotation vectors and
    penalizing only changes of angular velocity. Therefore a large, nearly
    constant per-frame translation/rotation receives almost no smoothness cost.
    """
    base = np.asarray(initial_transforms, dtype=np.float64)
    frame_count = len(base)
    if frame_count < 3:
        return base.copy(), True, 0.0

    confidence = np.asarray(frame_confidence, dtype=np.float64).reshape(-1)
    if confidence.shape != (frame_count,):
        raise ValueError("frame_confidence must have shape (T,).")
    confidence = np.clip(confidence, 0.05, 1.0)

    try:
        # Position: measurement fidelity + acceleration regularization.
        raw_position = base[:, :3, 3]
        smooth_position = _solve_regularized_sequence(
            raw_position,
            confidence,
            observation_sigma=translation_observation_sigma_m,
            smoothness_sigma=translation_acceleration_sigma_m,
            difference_order=2,
            anchor_index=anchor_index,
        )

        # Rotation: represent each adjacent motion as a world-frame rotvec.
        raw_increment = np.stack(
            [
                _rotation_increment_world(base[i, :3, :3], base[i + 1, :3, :3])
                for i in range(frame_count - 1)
            ],
            axis=0,
        )
        increment_confidence = np.minimum(confidence[:-1], confidence[1:])
        smooth_increment = _solve_regularized_sequence(
            raw_increment,
            increment_confidence,
            observation_sigma=float(np.deg2rad(rotation_observation_sigma_deg)),
            smoothness_sigma=float(np.deg2rad(rotation_acceleration_sigma_deg)),
            difference_order=1,
            anchor_index=None,
        )

        result = base.copy()
        result[:, :3, 3] = smooth_position
        anchor = int(anchor_index)
        result[anchor] = base[anchor]

        # Integrate the smoothed world-frame increments outward from the exact anchor.
        for frame in range(anchor, frame_count - 1):
            delta_rotation = Rotation.from_rotvec(smooth_increment[frame]).as_matrix()
            result[frame + 1, :3, :3] = delta_rotation @ result[frame, :3, :3]

        for frame in range(anchor - 1, -1, -1):
            delta_rotation = Rotation.from_rotvec(-smooth_increment[frame]).as_matrix()
            result[frame, :3, :3] = delta_rotation @ result[frame + 1, :3, :3]

        result[anchor] = base[anchor]

        # A compact diagnostic cost in normalized units.
        position_error = (result[:, :3, 3] - raw_position) / max(
            float(translation_observation_sigma_m), 1e-8
        )
        position_acceleration = (
            result[2:, :3, 3]
            - 2.0 * result[1:-1, :3, 3]
            + result[:-2, :3, 3]
        ) / max(float(translation_acceleration_sigma_m), 1e-8)
        rotation_error = (smooth_increment - raw_increment) / max(
            float(np.deg2rad(rotation_observation_sigma_deg)), 1e-8
        )
        rotation_acceleration = np.diff(smooth_increment, axis=0) / max(
            float(np.deg2rad(rotation_acceleration_sigma_deg)), 1e-8
        )
        cost = 0.5 * float(
            np.sum(confidence[:, None] * position_error ** 2)
            + np.sum(position_acceleration ** 2)
            + np.sum(increment_confidence[:, None] * rotation_error ** 2)
            + np.sum(rotation_acceleration ** 2)
        )
        return result, True, cost
    except (ValueError, np.linalg.LinAlgError):
        return base.copy(), False, float("nan")



def _adaptive_step_limits(
    translation_step_m: FloatArray,
    rotation_step_deg: FloatArray,
    *,
    minimum_translation_limit_m: float,
    minimum_rotation_limit_deg: float,
    mad_scale: float = 6.0,
    percentile: float = 90.0,
) -> tuple[float, float]:
    """Raise safety limits when downsampled motion is consistently large.

    The configured values are treated as *minimum safety floors*. Robust motion
    statistics can increase them, but never decrease them.
    """
    def robust_limit(values: FloatArray, floor: float) -> float:
        values = np.asarray(values, dtype=np.float64)[1:]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return float(floor)
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        robust = center + float(mad_scale) * 1.4826 * mad
        quantile = float(np.percentile(values, float(percentile)))
        return max(float(floor), robust, quantile)

    return (
        robust_limit(translation_step_m, minimum_translation_limit_m),
        robust_limit(rotation_step_deg, minimum_rotation_limit_deg),
    )


def estimate_object_pose_trajectory(
    positions_world: ArrayLike,
    valid_mask: ArrayLike,
    *,
    reference_frame: Optional[int] = None,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
    camera_position_world: Optional[ArrayLike] = None,
    world_x: Sequence[float] = (1.0, 0.0, 0.0),
    world_y: Sequence[float] = (0.0, 1.0, 0.0),
    inlier_threshold_m: float = 0.01,
    ransac_iterations: int = 256,
    random_seed: int = 0,
    temporal_median_window: int = 5,
    max_translation_step_m: float = 0.01,
    max_rotation_step_deg: float = 5.0,
    global_track_filter: bool = True,
    track_min_valid_ratio: float = 0.60,
    track_mad_scale: float = 3.0,
    minimum_stable_tracks: int = 6,
    stable_track_huber_delta_m: float = 0.006,
    stable_track_irls_iterations: int = 4,
    temporal_refinement: bool = True,
    temporal_translation_observation_sigma_m: float = 0.006,
    temporal_translation_acceleration_sigma_m: float = 0.030,
    temporal_rotation_observation_sigma_deg: float = 5.0,
    temporal_rotation_acceleration_sigma_deg: float = 20.0,
    adaptive_step_limit: bool = True,
    adaptive_step_limit_mad_scale: float = 6.0,
    adaptive_step_limit_percentile: float = 90.0,
    reference_points_world_override: Optional[ArrayLike] = None,
) -> ObjectPoseTrajectory:
    """Estimate a robust, temporally consistent canonical object pose trajectory.

    Pipeline:
        1. Define the canonical object frame once at the reference frame (PCA).
           If ``reference_points_world_override`` is supplied, use that real
           geometry for the canonical frame while estimating motion only from
           predicted track correspondences.
        2. Median-filter valid 3D tracks only.
        3. First-pass independent RANSAC/Kabsch pose fits.
        4. Estimate global per-track reliability over the complete video.
        5. Refit each pose with the same stable tracks using weighted robust IRLS.
        6. Smooth translation acceleration and angular-velocity changes only.
        7. Apply an adaptive hard step limiter as a final safety check.

    The temporal prior is deliberately task-independent. It does not know about
    target geometry or phase constraints. It also does NOT penalize large
    constant per-frame motion, which is important for temporally downsampled
    videos. The configured max step values are treated as safety floors when
    adaptive_step_limit=True, so consistent high-speed motion can raise them.
    """
    positions = np.asarray(positions_world, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            "positions_world must have shape (T, N, 3), "
            f"got {positions.shape}."
        )
    if valid.shape != positions.shape[:2]:
        raise ValueError(
            "valid_mask must have shape "
            f"{positions.shape[:2]}, got {valid.shape}."
        )

    valid = valid & np.all(np.isfinite(positions), axis=-1)

    temporal_median_window = int(temporal_median_window)
    if temporal_median_window < 1 or temporal_median_window % 2 == 0:
        raise ValueError("temporal_median_window must be a positive odd integer.")

    max_translation_step_m = float(max_translation_step_m)
    max_rotation_step_deg = float(max_rotation_step_deg)
    if max_translation_step_m <= 0.0 or not np.isfinite(max_translation_step_m):
        raise ValueError("max_translation_step_m must be finite and positive.")
    if max_rotation_step_deg <= 0.0 or not np.isfinite(max_rotation_step_deg):
        raise ValueError("max_rotation_step_deg must be finite and positive.")

    if not (0.0 < float(track_min_valid_ratio) <= 1.0):
        raise ValueError("track_min_valid_ratio must be in (0, 1].")
    if float(track_mad_scale) <= 0.0:
        raise ValueError("track_mad_scale must be positive.")
    if int(minimum_stable_tracks) < 3:
        raise ValueError("minimum_stable_tracks must be at least 3.")
    if float(stable_track_huber_delta_m) <= 0.0:
        raise ValueError("stable_track_huber_delta_m must be positive.")
    if int(stable_track_irls_iterations) < 1:
        raise ValueError("stable_track_irls_iterations must be positive.")
    if float(temporal_translation_observation_sigma_m) <= 0.0:
        raise ValueError("temporal_translation_observation_sigma_m must be positive.")
    if float(temporal_translation_acceleration_sigma_m) <= 0.0:
        raise ValueError("temporal_translation_acceleration_sigma_m must be positive.")
    if float(temporal_rotation_observation_sigma_deg) <= 0.0:
        raise ValueError("temporal_rotation_observation_sigma_deg must be positive.")
    if float(temporal_rotation_acceleration_sigma_deg) <= 0.0:
        raise ValueError("temporal_rotation_acceleration_sigma_deg must be positive.")

    frame_count = positions.shape[0]
    reference = _choose_reference_frame(valid, reference_frame)

    # Optional metric anchor from the real initial RGB-D frame.  The override
    # must correspond track-for-track with positions[:, N, :].  We still fit
    # motion from predicted frame-0 points to predicted later points; only the
    # canonical geometry/pose comes from the real depth.
    real_reference_points: Optional[FloatArray]
    if reference_points_world_override is None:
        real_reference_points = None
        reference_track_ids = np.flatnonzero(valid[reference])
    else:
        if reference != 0:
            raise ValueError(
                "reference_points_world_override anchors the initial frame, "
                "so the selected reference frame must be 0."
            )
        real_reference_points = np.asarray(
            reference_points_world_override,
            dtype=np.float64,
        )
        if real_reference_points.shape != (positions.shape[1], 3):
            raise ValueError(
                "reference_points_world_override must have shape "
                f"({positions.shape[1]}, 3), got {real_reference_points.shape}."
            )
        real_reference_valid = np.all(
            np.isfinite(real_reference_points),
            axis=-1,
        )
        reference_track_ids = np.flatnonzero(
            valid[reference] & real_reference_valid
        )
        if len(reference_track_ids) < 3:
            raise ValueError(
                "Fewer than three frame-0 tracks have both predicted and real "
                "3D positions."
            )

    positions_for_fit = _temporal_median_filter_positions(
        positions,
        valid,
        window=temporal_median_window,
    )
    # Never move the predicted motion anchor itself.
    positions_for_fit[reference] = positions[reference]

    if real_reference_points is None:
        reference_points_world = positions[reference, reference_track_ids]
    else:
        reference_points_world = real_reference_points[reference_track_ids]

    transform_world_object_reference, eigenvalues = compute_pca_object_pose(
        reference_points_world,
        world_up=world_up,
        camera_position_world=camera_position_world,
        world_x=world_x,
        world_y=world_y,
    )

    if real_reference_points is None:
        # Historical mode: fit the canonical object coordinates directly into
        # each predicted world-frame point cloud.
        fit_reference_points = points_world_to_object(
            transform_world_object_reference,
            reference_points_world,
        )
        fit_anchor_transform = transform_world_object_reference
        fit_is_relative_motion = False
    else:
        # Anchored mode: fit ONLY predicted relative motion P_pred(0)->P_pred(t).
        # The real frame-0 cloud is used solely to define T_world_object(0).
        fit_reference_points = positions[reference, reference_track_ids].copy()
        fit_anchor_transform = np.eye(4, dtype=np.float64)
        fit_is_relative_motion = True

    # ------------------------------------------------------------------
    # Pass 1: original independent RANSAC measurement fits.
    # ------------------------------------------------------------------
    first_pass = np.full((frame_count, 4, 4), np.nan, dtype=np.float64)
    valid_pose_mask = np.zeros(frame_count, dtype=bool)
    correspondence_counts = np.zeros(frame_count, dtype=np.int64)
    inlier_counts = np.zeros(frame_count, dtype=np.int64)
    fit_rms = np.full(frame_count, np.nan, dtype=np.float64)
    failed_frames: list[int] = []

    for frame in range(frame_count):
        common_mask = valid[frame, reference_track_ids]
        source_object = fit_reference_points[common_mask]
        target_world = positions_for_fit[
            frame,
            reference_track_ids[common_mask],
        ]
        correspondence_counts[frame] = source_object.shape[0]

        if source_object.shape[0] < 3:
            failed_frames.append(frame)
            continue

        if frame == reference:
            first_pass[frame] = fit_anchor_transform
            valid_pose_mask[frame] = True
            inlier_counts[frame] = int(source_object.shape[0])
            fit_rms[frame] = 0.0
            continue

        try:
            transform, inliers, residuals = fit_rigid_transform_ransac(
                source_object,
                target_world,
                inlier_threshold_m=inlier_threshold_m,
                iterations=ransac_iterations,
                random_seed=random_seed + frame,
            )
        except (ValueError, np.linalg.LinAlgError):
            failed_frames.append(frame)
            continue

        first_pass[frame] = transform
        valid_pose_mask[frame] = True
        inlier_counts[frame] = int(inliers.sum())
        selected_residuals = residuals[inliers] if int(inliers.sum()) > 0 else residuals
        fit_rms[frame] = float(np.sqrt(np.mean(selected_residuals ** 2)))

    first_pass = _interpolate_failed_transforms(first_pass, valid_pose_mask)
    if fit_is_relative_motion:
        first_pass_absolute = np.einsum(
            "tij,jk->tik",
            first_pass,
            transform_world_object_reference,
        )
    else:
        first_pass_absolute = first_pass
    raw_translation_step_m, raw_rotation_step_deg = _pose_step_sizes(
        first_pass_absolute
    )

    # ------------------------------------------------------------------
    # Global track reliability: same physical tracks dominate all frames.
    # ------------------------------------------------------------------
    (
        stable_track_ids,
        track_valid_ratio,
        track_median_residual_m,
        track_weight,
    ) = _compute_global_track_reliability(
        first_pass,
        reference_track_ids=reference_track_ids,
        reference_points_object=fit_reference_points,
        positions_world=positions_for_fit,
        valid_mask=valid,
        minimum_valid_ratio=track_min_valid_ratio,
        mad_scale=track_mad_scale,
        minimum_stable_tracks=minimum_stable_tracks,
    )

    if not global_track_filter:
        stable_track_ids = reference_track_ids.astype(np.int64).copy()
        track_weight = np.ones(len(reference_track_ids), dtype=np.float64)

    # ------------------------------------------------------------------
    # Pass 2: weighted robust refit using globally stable tracks.
    # ------------------------------------------------------------------
    stable_refit, stable_success, stable_counts, stable_rms = _refit_with_stable_tracks(
        first_pass,
        reference_index=reference,
        reference_track_ids=reference_track_ids,
        reference_points_object=fit_reference_points,
        stable_track_ids=stable_track_ids,
        track_weights_local=track_weight,
        positions_world=positions_for_fit,
        valid_mask=valid,
        huber_delta_m=stable_track_huber_delta_m,
        irls_iterations=stable_track_irls_iterations,
    )

    # If a second-pass frame could not be fit, retain the first-pass pose.
    stable_refit[~stable_success] = first_pass[~stable_success]
    stable_refit[reference] = fit_anchor_transform

    # Frame confidence affects only how strongly the temporal refinement trusts
    # a per-frame measurement; it never creates a semantic/task constraint.
    if int(np.max(stable_counts)) > 0:
        frame_confidence = stable_counts.astype(np.float64) / float(np.max(stable_counts))
    else:
        frame_confidence = np.ones(frame_count, dtype=np.float64)

    finite_rms = stable_rms[np.isfinite(stable_rms) & (stable_rms > 0.0)]
    if len(finite_rms) > 0:
        rms_scale = max(float(np.median(finite_rms)), 1e-4)
        quality = np.ones(frame_count, dtype=np.float64)
        valid_rms = np.isfinite(stable_rms)
        quality[valid_rms] = 1.0 / (1.0 + (stable_rms[valid_rms] / rms_scale) ** 2)
        frame_confidence *= quality
    frame_confidence = np.clip(frame_confidence, 0.05, 1.0)
    frame_confidence[reference] = 1.0

    # Convert the robust relative-motion sequence into the real metric object
    # pose sequence exactly once.  From here onward all smoothing/step limits
    # operate on the actual T_world_object trajectory.
    if fit_is_relative_motion:
        stable_pose_sequence = np.einsum(
            "tij,jk->tik",
            stable_refit,
            transform_world_object_reference,
        )
    else:
        stable_pose_sequence = stable_refit
    stable_pose_sequence[reference] = transform_world_object_reference

    # ------------------------------------------------------------------
    # Temporal refinement: acceleration only, not motion magnitude.
    # ------------------------------------------------------------------
    temporal_refinement_success = False
    temporal_refinement_cost = float("nan")
    if temporal_refinement:
        refined, temporal_refinement_success, temporal_refinement_cost = (
            _temporally_refine_pose_trajectory(
                stable_pose_sequence,
                anchor_index=reference,
                frame_confidence=frame_confidence,
                translation_observation_sigma_m=(
                    temporal_translation_observation_sigma_m
                ),
                translation_acceleration_sigma_m=(
                    temporal_translation_acceleration_sigma_m
                ),
                rotation_observation_sigma_deg=(
                    temporal_rotation_observation_sigma_deg
                ),
                rotation_acceleration_sigma_deg=(
                    temporal_rotation_acceleration_sigma_deg
                ),
            )
        )
        if not temporal_refinement_success:
            refined = stable_pose_sequence.copy()
    else:
        refined = stable_pose_sequence.copy()

    refined[reference] = transform_world_object_reference
    prelimit_translation_step_m, prelimit_rotation_step_deg = _pose_step_sizes(refined)

    # ------------------------------------------------------------------
    # Final safety limiter. With adaptive mode, configured values are floors.
    # ------------------------------------------------------------------
    if adaptive_step_limit:
        (
            effective_max_translation_step_m,
            effective_max_rotation_step_deg,
        ) = _adaptive_step_limits(
            prelimit_translation_step_m,
            prelimit_rotation_step_deg,
            minimum_translation_limit_m=max_translation_step_m,
            minimum_rotation_limit_deg=max_rotation_step_deg,
            mad_scale=adaptive_step_limit_mad_scale,
            percentile=adaptive_step_limit_percentile,
        )
    else:
        effective_max_translation_step_m = max_translation_step_m
        effective_max_rotation_step_deg = max_rotation_step_deg

    transforms, limited_frame_indices = _limit_pose_steps_from_anchor(
        refined,
        anchor_index=reference,
        max_translation_step_m=effective_max_translation_step_m,
        max_rotation_step_deg=effective_max_rotation_step_deg,
    )
    transforms[reference] = transform_world_object_reference
    final_translation_step_m, final_rotation_step_deg = _pose_step_sizes(transforms)

    world_up_array = normalize(world_up, name="world_up")
    reference_z_dot = float(
        np.dot(transform_world_object_reference[:3, 2], world_up_array)
    )

    diagnostics = ObjectPoseDiagnostics(
        reference_frame=reference,
        reference_track_ids=reference_track_ids.astype(np.int64),
        pca_eigenvalues_ascending=eigenvalues.astype(np.float64),
        reference_z_dot_world_up=reference_z_dot,
        valid_correspondence_count=correspondence_counts,
        inlier_count=inlier_counts,
        fit_rms_m=fit_rms,
        failed_frame_indices=tuple(failed_frames),
        temporal_median_window=temporal_median_window,
        max_translation_step_m=max_translation_step_m,
        max_rotation_step_deg=max_rotation_step_deg,
        limited_frame_indices=limited_frame_indices,
        raw_translation_step_m=raw_translation_step_m,
        raw_rotation_step_deg=raw_rotation_step_deg,
        final_translation_step_m=final_translation_step_m,
        final_rotation_step_deg=final_rotation_step_deg,
        stable_track_ids=stable_track_ids.astype(np.int64),
        track_valid_ratio=track_valid_ratio.astype(np.float64),
        track_median_residual_m=track_median_residual_m.astype(np.float64),
        track_weight=track_weight.astype(np.float64),
        temporal_refinement_enabled=bool(temporal_refinement),
        temporal_refinement_success=bool(temporal_refinement_success),
        temporal_refinement_cost=float(temporal_refinement_cost),
        prelimit_translation_step_m=prelimit_translation_step_m,
        prelimit_rotation_step_deg=prelimit_rotation_step_deg,
        effective_max_translation_step_m=float(effective_max_translation_step_m),
        effective_max_rotation_step_deg=float(effective_max_rotation_step_deg),
    )

    return ObjectPoseTrajectory(
        transforms=transforms,
        diagnostics=diagnostics,
    )

