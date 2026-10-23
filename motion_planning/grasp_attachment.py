from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .se3 import (
    as_transform,
    compose_transform,
    make_transform,
    relative_transform,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GraspAttachment:
    """
    Rigid attachment between the canonical object frame O and control/grasp
    frame G.

    G has:
        origin      = selected grasp point
        orientation = canonical object PCA orientation

    Therefore R_object_grasp is identity, while p_object_grasp stores the
    selected grasp point in object coordinates.
    """

    transform_object_grasp: FloatArray

    @property
    def point_object(self) -> FloatArray:
        return self.transform_object_grasp[:3, 3].copy()


def build_grasp_attachment(
    transform_world_object_at_grasp: ArrayLike,
    grasp_point_world: ArrayLike,
) -> GraspAttachment:
    """
    Build constant T_object_grasp from an initial object pose and grasp point.

    The grasp-frame orientation equals the canonical object orientation.
    """
    transform_world_object = as_transform(
        transform_world_object_at_grasp,
        copy=False,
    )
    point_world = np.asarray(grasp_point_world, dtype=np.float64)
    if point_world.shape != (3,):
        raise ValueError(
            f"grasp_point_world must have shape (3,), got {point_world.shape}."
        )
    if not np.all(np.isfinite(point_world)):
        raise ValueError("grasp_point_world contains non-finite values.")

    # Same orientation as O, translated to the chosen grasp point.
    transform_world_grasp = make_transform(
        transform_world_object[:3, :3],
        point_world,
    )
    transform_object_grasp = relative_transform(
        transform_world_object,
        transform_world_grasp,
    )

    return GraspAttachment(
        transform_object_grasp=transform_object_grasp,
    )


def object_trajectory_to_grasp_trajectory(
    transforms_world_object: ArrayLike,
    attachment: GraspAttachment,
) -> FloatArray:
    """
    Convert T_world_object[t] to T_world_grasp[t].

    The same rigid T_object_grasp is used for every frame.
    """
    trajectory = np.asarray(transforms_world_object, dtype=np.float64)
    if trajectory.ndim != 3 or trajectory.shape[1:] != (4, 4):
        raise ValueError(
            "transforms_world_object must have shape (T, 4, 4), "
            f"got {trajectory.shape}."
        )

    transform_object_grasp = as_transform(
        attachment.transform_object_grasp,
        copy=False,
    )

    result = []
    for frame, transform_world_object in enumerate(trajectory):
        try:
            result.append(
                compose_transform(
                    transform_world_object,
                    transform_object_grasp,
                )
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid object transform at frame {frame}: {error}"
            ) from error

    return np.stack(result, axis=0)
