"""Target velocity read from a terrain velocity map at the walker's position."""

from pathlib import Path

import numpy as np


class TargetVelocityMap:
    """The speed of a `myoassist_terrains` velocity map, interpolated at a world (x, y).

    The map's samples lie on a rectilinear grid: every tile carries the same n x n sample offsets
    and the tiles are aligned in rows and columns, so the sample x and y coordinates each form one
    sorted axis. The speed is interpolated bilinearly between the four surrounding samples, so the
    target changes continuously as the walker moves; the nearest sample would step every
    tile_size / samples_per_tile metres. Beyond the outermost samples the edge value holds.

    Only the speed is used. The env observes and rewards a scalar forward speed along pelvis_tx
    and has no heading term, so the map's direction has nothing to act on.
    """

    def __init__(self, terrain, params: dict):
        from myoassist_terrains.config import config_from_dict, load_config
        from myoassist_terrains.velocity_map import generate_velocity_map

        assert terrain is not None, "a velocity map is sampled over the terrain, so env_params.terrain must be set"
        config = config_from_dict(terrain) if isinstance(terrain, dict) else load_config(Path(terrain))
        samples = generate_velocity_map(config, **params)

        self.xs = np.unique([s.position[0] for s in samples])
        self.ys = np.unique([s.position[1] for s in samples])
        assert len(self.xs) * len(self.ys) == len(samples), (
            f"{len(samples)} velocity-map samples do not form a {len(self.ys)} x {len(self.xs)} grid"
        )
        self.speed = np.full((len(self.ys), len(self.xs)), np.nan)
        for s in samples:
            self.speed[np.searchsorted(self.ys, s.position[1]), np.searchsorted(self.xs, s.position[0])] = s.speed
        assert not np.isnan(self.speed).any(), "a grid point of the velocity map has no sample"

    def speed_at(self, x: float, y: float) -> float:
        ix0, ix1, tx = _bracket(self.xs, x)
        iy0, iy1, ty = _bracket(self.ys, y)
        s = self.speed
        return float((1 - ty) * ((1 - tx) * s[iy0, ix0] + tx * s[iy0, ix1]) + ty * ((1 - tx) * s[iy1, ix0] + tx * s[iy1, ix1]))


def _bracket(axis: np.ndarray, v: float) -> tuple[int, int, float]:
    """The two samples around `v` on a sorted axis and `v`'s fraction between them, held at the ends."""
    if len(axis) == 1:
        return 0, 0, 0.0
    i1 = int(np.clip(np.searchsorted(axis, v), 1, len(axis) - 1))
    t = (v - axis[i1 - 1]) / (axis[i1] - axis[i1 - 1])
    return i1 - 1, i1, float(np.clip(t, 0.0, 1.0))
