"""Training starts and steps on each terrain form.

Covers the two terrain config shapes (a uniform whole-ground heightfield and a tile grid) and the
tile types whose geometry is not just a height offset -- stairs, a trench, and scattered supports --
because those are what terrain injection and model seating can get wrong. A failure here means the
composed model no longer loads or no longer holds the model up, not that a policy learns badly.

The full nine-tile matrix is not run here: the omitted tiles differ from the covered ones only in
their heightfield contents, so they cost runtime without adding a failure mode. Extend TERRAINS if
a tile ever needs its own case.
"""

from __future__ import annotations

import json

import pytest

from tools.tests.conftest import REPO_ROOT, build_env, shipped_configs, take_a_few_ppo_steps

TUTORIAL = next(p for p in shipped_configs() if "Tutorial_L1" in p.name)
STRIDE = next(p for p in shipped_configs() if "STRIDE_L2" in p.name)


def _tile(tile_type: str) -> dict:
    return {
        "terrain_name": f"tile_{tile_type}",
        "grid": {"rows": 1, "cols": 1, "tile_size": [8.0, 8.0]},
        "border": {"width": 0.5, "match_mode": "min"},
        "tiles": [{"row": 0, "col": 0, "type": tile_type, "params": {}}],
    }


# (id, terrain config or None for the flat default, training config)
TERRAINS = [
    ("flat_default", None, TUTORIAL),
    ("uniform_slope", {"terrain": "slope", "terrain_name": "uniform_slope", "deg": 8.0}, TUTORIAL),
    ("uniform_random", {"terrain": "random", "terrain_name": "uniform_random", "seed": 3}, TUTORIAL),
    ("tile_rough", _tile("rough"), TUTORIAL),
    ("tile_stairs", _tile("stairs"), TUTORIAL),
    ("tile_gap", _tile("gap"), TUTORIAL),
    ("tile_stepping_stones", _tile("stepping_stones"), TUTORIAL),
    # A tiled course on an L2 multibody device with a thick sole: terrain seating and the
    # reference height correction both act on the model's standing height, so this is the case
    # where they can disagree.
    ("mixed_course_stride", None, STRIDE),
]


@pytest.mark.parametrize("case_id,terrain,config_path", TERRAINS, ids=[t[0] for t in TERRAINS])
def test_training_starts_on_terrain(case_id, terrain, config_path, tmp_path):
    if case_id == "mixed_course_stride":
        example = json.loads((REPO_ROOT / "docs/examples/env_tiled_course.json").read_text())
        terrain = example["terrain"]

    terrain_path = None
    if terrain is not None:
        terrain_path = tmp_path / "terrain.json"
        terrain_path.write_text(json.dumps(terrain))

    config, env = build_env(config_path, terrain=terrain_path)
    try:
        take_a_few_ppo_steps(config, env)
    finally:
        env.close()
