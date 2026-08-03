"""Shared model compose + load pipeline for MyoAssist (see REPO_ALIGNMENT.md A10).

``compose_env_model(msk_key, device_key, terrain=None, export_path=None) -> str``

Composes a human musculoskeletal model + assistive device (via ``assist_sim``)
and merges a terrain scene (via ``myoassist_terrains``) into a single, loadable
MJCF, returned as an XML *string*.  Both the RL and CO frameworks call this so
they share one model/env build path.

The env consumes the returned string directly through myosuite's ``SimScene``
(``"<mujoco"`` -> ``MjModel.from_xml_string``), so there is no temp file on the
run path and myosuite is not modified.  All asset paths (model meshes, terrain
heightfields / textures) are absolutized so the string loads regardless of CWD.

Reference: the proven figure merge in ``compile_check/fig/build_fig.py`` (helpers
``inject_terrain`` / ``absolutize_files`` / ``seat_model`` / ``lowest_geom_z``);
the figure-only bits (slide root, white scene, velocity arrows) are dropped here.
"""
from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import mujoco as mj

from assist_sim import load_combined
from myoassist_terrains import build_terrain
from myoassist_terrains.config import _config_from_dict, load_config

# Seat the lowest collidable geom this far *below* the terrain surface so the
# model opens in light contact (MuJoCo needs penetration, not just touching, to
# register a contact).  ~5 mm matches the validated merge prototype.
_CONTACT_SEAT_DEPTH = 0.005


# ---------------------------------------------------------------------------
# XML helpers (adapted from compile_check/fig/build_fig.py)
# ---------------------------------------------------------------------------
def _find_or_make(root: ET.Element, tag: str, index: int = 0) -> ET.Element:
    el = root.find(tag)
    if el is None:
        el = ET.Element(tag)
        root.insert(index, el)
    return el


def _absolutize_files(root: ET.Element, base_dir: Union[str, Path]) -> None:
    """Rewrite every relative ``file=`` attr to absolute so an XML *string* loads
    regardless of the current working directory."""
    base_dir = Path(base_dir)
    for el in root.iter():
        f = el.get("file")
        if f and not Path(f).is_absolute():
            el.set("file", str((base_dir / f).resolve()).replace("\\", "/"))


def _inject_terrain(root: ET.Element, terrain_root: ET.Element) -> None:
    """Merge a terrain spec's ``<asset>`` + worldbody geoms into the model scene.

    Each injected terrain geom gets ``contype``/``conaffinity`` stamped to ``1``.
    CRITICAL: ``MjSpec.to_xml`` drops those attrs when they match the terrain
    spec's own default; without re-stamping, the injected geoms inherit the
    *consuming* model's default geom class (non-colliding for the composed
    model), and the model falls straight through the terrain.
    """
    model_asset = _find_or_make(root, "asset", 0)
    t_asset = terrain_root.find("asset")
    if t_asset is not None:
        for child in list(t_asset):
            model_asset.append(child)
    model_wb = root.find("worldbody")
    t_wb = terrain_root.find("worldbody")
    if t_wb is not None:
        for child in list(t_wb):
            if child.tag == "geom":
                child.set("contype", "1")
                child.set("conaffinity", "1")
            model_wb.append(child)


def _seat_model(root: ET.Element, dz: float) -> None:
    """Shift every worldbody-direct ``<body>`` (the model's root parts) by ``dz``
    in z so the model opens with its feet on the terrain surface.  Terrain geoms
    are direct worldbody *geoms*, not bodies, so they are left in place."""
    if abs(dz) < 1e-9:
        return
    for body in root.find("worldbody").findall("body"):
        pos = [float(v) for v in (body.get("pos") or "0 0 0").split()]
        pos[2] += dz
        body.set("pos", " ".join(f"{v:.6f}" for v in pos))


def _seat_dz_by_collision(
    merged_xml: str,
    terrain_geom_names: set,
    penetration: float = _CONTACT_SEAT_DEPTH,
) -> float:
    """Vertical shift (m) that seats the model on the terrain in light contact.

    Compiles the merged (unseated) model, gives the terrain geoms a large contact
    margin, and reads the smallest signed distance between any terrain geom and
    any model geom at the opening pose (the "stand" keyframe when present).  This
    uses MuJoCo's real collision geometry, so it is exact for mesh device geoms
    (unlike an axis-aligned-bounding-box estimate) and it accounts for the
    terrain surface height directly under the model.

    Returns ``dz`` such that after shifting, the closest model geom sits
    ``penetration`` below the surface.  Returns ``0.0`` if no terrain-model pair
    comes within margin (model too far from the terrain to measure).
    """
    m = mj.MjModel.from_xml_string(merged_xml)
    d = mj.MjData(m)
    terr = [i for i in range(m.ngeom) if (m.geom(i).name or "") in terrain_geom_names]
    if not terr:
        return 0.0
    m.geom_margin[terr] = 50.0  # detect near-contacts well outside touching range
    if m.nkey > 0:
        mj.mj_resetDataKeyframe(m, d, 0)  # keyframe 0 == "stand"
    else:
        d.qpos[:] = m.qpos0
    mj.mj_forward(m, d)
    terrset = set(terr)
    gaps = [d.contact[c].dist for c in range(d.ncon)
            if (d.contact[c].geom1 in terrset) ^ (d.contact[c].geom2 in terrset)]
    if not gaps:
        return 0.0
    # new_gap = min(gaps) + dz ; want new_gap == -penetration.
    return -penetration - min(gaps)


def _inject_lighting(root: ET.Element) -> None:
    """Add a headlight + a directional light so the composed model renders in a
    viewer / eval.  Physics does not need this; only add what is missing."""
    visual = _find_or_make(root, "visual", 1)
    if visual.find("headlight") is None:
        ET.SubElement(visual, "headlight", {
            "ambient": "0.4 0.4 0.4",
            "diffuse": "0.6 0.6 0.6",
            "specular": "0.1 0.1 0.1",
        })
    wb = root.find("worldbody")
    if not wb.findall("light"):
        ET.SubElement(wb, "light", {
            "name": "compose_key",
            "directional": "true",
            "castshadow": "true",
            "pos": "3 -3 4",
            "dir": "-0.55 0.45 -0.7",
            "diffuse": "0.6 0.6 0.6",
            "specular": "0.2 0.2 0.2",
        })


def _flat_default_config():
    """A 1x1 flat tile at z=0 -- the default ground when no terrain is given
    (the assist_sim model-only export carries no surface)."""
    return _config_from_dict({
        "terrain_name": "flat_default",
        "grid": {"rows": 1, "cols": 1, "tile_size": [12.0, 12.0]},
        "border": {"width": 0.0},
        "tiles": [{"row": 0, "col": 0, "type": "flat", "params": {"height": 0.0}}],
    })


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def compose_env_model(
    msk_key: str,
    device_key: str,
    terrain: Optional[Union[str, Path]] = None,
    export_path: Optional[Union[str, Path]] = None,
) -> str:
    """Compose ``msk_key`` + ``device_key`` + a terrain into one loadable MJCF.

    Parameters
    ----------
    msk_key : str
        A human MSK registry key (e.g. ``"myolegs22"``, ``"myolegs26"``).
    device_key : str
        An assist_sim device key (e.g. ``"DephyExoBoot_L1"``).
    terrain : str | Path | None
        Path to a ``myoassist_terrains`` JSON config, or ``None`` for a flat
        default ground (a single 1x1 flat tile at z=0).
    export_path : str | Path | None
        If given, the merged model is also written to this path as a
        standalone, ``from_xml_path``-loadable file.  Any terrain assets
        (heightfields / textures) are written to a sibling ``*_assets`` dir.

    Returns
    -------
    str
        The merged model as an MJCF XML string (env input via
        ``MjModel.from_xml_string``).
    """
    # 1) compose human + device -> reload-safe, model-only XML (scene stripped).
    model_dir = Path(tempfile.mkdtemp(prefix="myoassist_compose_"))
    model_xml_path = model_dir / "model_only.xml"
    load_combined(msk_key, device_key, export_xml=str(model_xml_path))
    model_root = ET.parse(model_xml_path).getroot()
    # meshes are written relative to the export dir -> resolve to the (absolute,
    # persistent) install location so the temp export dir can be removed.
    _absolutize_files(model_root, model_xml_path.parent)

    # 2) build the terrain (flat default, or from a terrains JSON config).
    cfg = _flat_default_config() if terrain is None else load_config(Path(terrain))
    # Terrain assets (hfield PNGs / textures) must outlive this call.  When
    # exporting, keep them beside the export; otherwise use a persistent temp
    # dir (the flat default writes no assets, so this is a no-op for RL runs).
    if export_path is not None:
        export_path = Path(export_path)
        assets_dir = export_path.resolve().parent / f"{export_path.stem}_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
    else:
        assets_dir = Path(tempfile.mkdtemp(prefix="myoassist_terrain_"))
    tspec = build_terrain(cfg, output_dir=assets_dir)
    tspec.compile()
    terrain_root = ET.fromstring(tspec.to_xml())
    _absolutize_files(terrain_root, assets_dir)
    terrain_geom_names = {g.get("name")
                          for g in terrain_root.find("worldbody").iter("geom")
                          if g.get("name")}

    # 3) merge terrain (colliding) + add default lighting.
    _inject_terrain(model_root, terrain_root)
    _inject_lighting(model_root)

    # 4) seat the model so its feet rest on the terrain surface (light contact),
    #    measured from the real collision geometry at the opening pose.
    dz = _seat_dz_by_collision(ET.tostring(model_root, encoding="unicode"),
                               terrain_geom_names)
    _seat_model(model_root, dz)

    # 5) serialize.  Model meshes are now absolute install paths, so the temp
    #    export file/dir can go.
    xml_str = ET.tostring(model_root, encoding="unicode")
    model_xml_path.unlink(missing_ok=True)
    for scratch in (model_dir, assets_dir):
        try:
            scratch.rmdir()  # only removes it if empty (e.g. no terrain assets)
        except OSError:
            pass

    if export_path is not None:
        ET.indent(model_root, space="  ")
        export_path.write_text(ET.tostring(model_root, encoding="unicode"), encoding="utf-8")

    return xml_str
