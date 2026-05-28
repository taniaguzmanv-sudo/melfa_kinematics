"""Plotting helpers for MELFA kinematics examples."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_frames_3d(frames, ax=None, scale=50, labels=None, show=True):
    """Plot one or more 3D coordinate frames.

    Parameters
    ----------
    frames : sequence of array-like or mapping
        Homogeneous transforms with shape ``(4, 4)``. A single transform may be
        passed directly. When a mapping is supplied, its keys are used as labels
        unless ``labels`` is provided.
    ax : matplotlib 3D axes, optional
        Existing axes to draw on. When omitted, a new figure and 3D axes are
        created.
    scale : float, optional
        Length used for each coordinate axis.
    labels : sequence of str, optional
        Frame labels. Defaults to the mapping keys or ``F0``, ``F1``, ...
    show : bool, optional
        Whether to call ``plt.show()`` before returning.

    Returns
    -------
    matplotlib.axes.Axes
        The 3D axes containing the plotted frames.
    """
    if isinstance(frames, dict):
        frame_items = list(frames.items())
        if labels is None:
            labels = [str(name) for name, _ in frame_items]
        frames = [frame for _, frame in frame_items]
    else:
        frames = list(frames) if _is_sequence_of_frames(frames) else [frames]

    if labels is None:
        labels = [f"F{i}" for i in range(len(frames))]

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    origins = []
    axis_points = []

    for frame, label in zip(frames, labels):
        transform = np.asarray(frame, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("Each frame must be a 4x4 homogeneous transform")

        origin = transform[:3, 3]
        rotation = transform[:3, :3]
        origins.append(origin)
        axis_points.extend(origin + scale * rotation[:, i] for i in range(3))

        ax.scatter(origin[0], origin[1], origin[2], color="k", s=20)
        ax.text(origin[0], origin[1], origin[2], label)

        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            rotation[0, 0],
            rotation[1, 0],
            rotation[2, 0],
            length=scale,
            color="r",
            normalize=True,
        )
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            rotation[0, 1],
            rotation[1, 1],
            rotation[2, 1],
            length=scale,
            color="g",
            normalize=True,
        )
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            rotation[0, 2],
            rotation[1, 2],
            rotation[2, 2],
            length=scale,
            color="b",
            normalize=True,
        )

    if origins:
        _set_axes_equal(ax, np.vstack([origins, axis_points]))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if show:
        plt.show()

    return ax


def _is_sequence_of_frames(frames):
    array = np.asarray(frames)
    return array.ndim == 3 and array.shape[-2:] == (4, 4)


def _set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2
    radius = np.max(maxs - mins) / 2

    if radius == 0:
        radius = 1

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
