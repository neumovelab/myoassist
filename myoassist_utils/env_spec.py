"""Shared environment spec: the ``{msk, device, terrain}`` triple.

The single front-door schema that both the CO (``ctrl_optim``) and RL
(``rl_train``) pipelines use to define a composed model.  Author it once (a JSON
file or a dict) and hand it to either pipeline, which resolves it to a loadable
MJCF via :func:`myoassist_utils.compose.compose_env_model`.

Fields are raw assist_sim registry keys -- run ``python -m assist_sim list`` to
see the valid set:

``msk``
    Human MSK key, e.g. ``"myolegs22"`` (2D lineage) or ``"myolegs26"`` (3D).
``device``
    Assistive device key, e.g. ``"DephyExoBoot_L1"``, ``"Humotech_L1"``, or
    ``"Tutorial_L1"`` (a passive device).
``terrain``
    One of:
      * ``None``       -- a flat default ground plane,
      * a path (str)   -- a ``myoassist_terrains`` JSON config file,
      * an inline dict -- a terrains config (grid form, or uniform e.g.
        ``{"terrain": "slope", "deg": 8}``).

Example JSON::

    {"msk": "myolegs22", "device": "Humotech_L1", "terrain": {"terrain": "slope", "deg": 5}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

# A terrain is unset (flat default), a path to a terrains JSON, or an inline
# terrains config dict.
TerrainSpec = Optional[Union[str, dict]]


def slope_deg_from_terrain(terrain: TerrainSpec) -> float:
    """The incline angle in degrees if ``terrain`` is a uniform ``slope``, else 0.

    Terrain is the single source of the course grade; downstream consumers (the
    eval follow-camera, cost function, display) derive the angle from it rather
    than a separate flag.  Accepts a dict, an inline JSON string, or a path.
    """
    if terrain is None:
        return 0.0
    cfg: Optional[dict] = None
    if isinstance(terrain, dict):
        cfg = terrain
    else:
        s = str(terrain).strip()
        if s.startswith("{"):
            cfg = json.loads(s)
        else:
            from myoassist_terrains.config import load_config

            loaded = load_config(Path(terrain))
            return float(getattr(loaded, "deg", 0.0)) if getattr(loaded, "terrain", None) == "slope" else 0.0
    if isinstance(cfg, dict) and cfg.get("terrain") == "slope":
        return float(cfg.get("deg", 0.0))
    return 0.0


@dataclass
class EnvSpec:
    """A composed-environment definition shared by the CO and RL pipelines."""

    msk: str
    device: str
    terrain: TerrainSpec = None

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: dict) -> "EnvSpec":
        """Build an :class:`EnvSpec` from a plain dict (e.g. a parsed JSON block)."""
        missing = [k for k in ("msk", "device") if not raw.get(k)]
        if missing:
            raise ValueError(f"EnvSpec requires non-empty {missing}; got keys {sorted(raw)}.")
        return cls(msk=str(raw["msk"]), device=str(raw["device"]), terrain=raw.get("terrain"))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EnvSpec":
        """Load an :class:`EnvSpec` from a JSON file."""
        with Path(path).open(encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict:
        """Serialize to a plain dict (omits ``terrain`` when unset)."""
        out: dict = {"msk": self.msk, "device": self.device}
        if self.terrain is not None:
            out["terrain"] = self.terrain
        return out

    # -- validation -----------------------------------------------------------
    def validate(self) -> "EnvSpec":
        """Validate ``msk``/``device`` against the assist_sim registry and the
        ``terrain`` against the terrains loader.

        Returns ``self`` so calls can chain (``EnvSpec(...).validate().compose()``).
        Raises ``ValueError`` listing the valid options when a key is unknown or
        the msk/device pair is incompatible.
        """
        from assist_sim.registry import get_available_combinations

        combos = get_available_combinations()
        if self.msk not in combos:
            available = sorted(combos) or "(none -- is myo_sim installed?)"
            raise ValueError(f"Unknown or unavailable MSK {self.msk!r}. Available MSK keys: {available}.")
        if self.device not in combos[self.msk]:
            raise ValueError(
                f"Device {self.device!r} is not compatible with MSK {self.msk!r}. "
                f"Compatible devices: {sorted(combos[self.msk])}."
            )
        self._validate_terrain()
        return self

    def _validate_terrain(self) -> None:
        if self.terrain is None:
            return
        if isinstance(self.terrain, dict):
            from myoassist_terrains.config import config_from_dict

            config_from_dict(self.terrain)  # raises on a malformed terrain config
            return
        if not Path(self.terrain).exists():
            raise ValueError(f"terrain config file not found: {self.terrain!r}")

    # -- build ----------------------------------------------------------------
    def compose(
        self,
        export_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        """Compose this spec into a loadable MJCF string.

        When ``export_path`` is given, the merged model is also written there as
        a standalone, ``from_xml_path``-loadable file.

        ``cache_dir`` opts in to the merged-model cache; leaving it unset still picks up
        ``MYOASSIST_CACHE_DIR`` from the environment, which is the zero-plumbing way to
        enable it for a whole RL or CO run.  See
        :func:`myoassist_utils.compose.compose_env_model`.
        """
        from myoassist_utils.compose import compose_env_model

        return compose_env_model(
            self.msk,
            self.device,
            terrain=self.terrain,
            export_path=export_path,
            cache_dir=cache_dir,
        )
