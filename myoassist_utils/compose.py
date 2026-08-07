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

import atexit
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import mujoco as mj

from assist_sim import load_combined
from myoassist_terrains import build_terrain
from myoassist_terrains.config import config_from_dict, load_config

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
    gaps = [d.contact[c].dist for c in range(d.ncon) if (d.contact[c].geom1 in terrset) ^ (d.contact[c].geom2 in terrset)]
    if not gaps:
        return 0.0
    # new_gap = min(gaps) + dz ; want new_gap == -penetration.
    return -penetration - min(gaps)


def _inject_render_defaults(root: ET.Element) -> None:
    """Add the render-only settings a composed model needs in a viewer / eval:
    a headlight, a directional light, and an offscreen framebuffer big enough for
    the eval's replay resolution.  Physics needs none of this; only add what is
    missing.

    The framebuffer matters because MuJoCo defaults ``offwidth``/``offheight`` to
    640x480 and *raises* rather than downscaling when asked for a larger frame, so
    without this the eval's 1920x1080 replay dies with "Image width 1920 >
    framebuffer width 640" -- and, since the analyzer runs in a worker process,
    takes the whole training run with it.  The per-model XMLs this pipeline replaced
    set it themselves (`models/22muscle_2D/myoLeg22_2D_TUTORIAL.xml` on `dev`:
    ``<global offwidth="1920" offheight="1080"/>``); composing from myo_sim +
    assist_sim specs does not carry it over.
    """
    visual = _find_or_make(root, "visual", 1)
    if visual.find("global") is None:
        ET.SubElement(visual, "global", {"offwidth": "1920", "offheight": "1080"})
    if visual.find("headlight") is None:
        ET.SubElement(
            visual,
            "headlight",
            {
                "ambient": "0.4 0.4 0.4",
                "diffuse": "0.6 0.6 0.6",
                "specular": "0.1 0.1 0.1",
            },
        )
    wb = root.find("worldbody")
    if not wb.findall("light"):
        ET.SubElement(
            wb,
            "light",
            {
                "name": "compose_key",
                "directional": "true",
                "castshadow": "true",
                "pos": "3 -3 4",
                "dir": "-0.55 0.45 -0.7",
                "diffuse": "0.6 0.6 0.6",
                "specular": "0.2 0.2 0.2",
            },
        )


def _inject_terrain_haze(root: ET.Element, terrain_root: ET.Element) -> None:
    """Propagate the terrain's horizon haze (fog) color onto the model's
    ``<visual>`` so an infinite ground plane fades into its own color rather
    than MuJoCo's default white.  No-op if the terrain declares no haze."""
    t_vis = terrain_root.find("visual")
    t_rgba = t_vis.find("rgba") if t_vis is not None else None
    if t_rgba is None or t_rgba.get("haze") is None:
        return
    visual = _find_or_make(root, "visual", 1)
    rgba = visual.find("rgba")
    if rgba is None:
        rgba = ET.SubElement(visual, "rgba")
    rgba.set("haze", t_rgba.get("haze"))


def _flat_default_config():
    """The default ground when no terrain is given: a single infinite flat plane
    at z=0 (the assist_sim model-only export carries no surface).  This mirrors
    the old myoassist ground -- an effectively-infinite plane, one cheap geom --
    rather than a finite box tile, so a walking model never runs off the edge."""
    return config_from_dict({"terrain": "flat"})


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def compose_env_model(
    msk_key: str,
    device_key: str,
    terrain: Optional[Union[str, Path, dict]] = None,
    export_path: Optional[Union[str, Path]] = None,
) -> str:
    """Compose ``msk_key`` + ``device_key`` + a terrain into one loadable MJCF.

    Parameters
    ----------
    msk_key : str
        A human MSK registry key (e.g. ``"myolegs22"``, ``"myolegs26"``).
    device_key : str
        An assist_sim device key (e.g. ``"DephyExoBoot_L1"``).
    terrain : str | Path | dict | None
        A ``myoassist_terrains`` config: a path to a JSON file, or an inline
        config dict (grid form, or uniform e.g. ``{"terrain": "slope", "deg": 8}``).
        ``None`` gives a flat default ground plane.
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
    # The immediate rmdir below only clears empty scratch dirs; register a
    # process-exit cleanup so any that still hold files (e.g. a terrain assets
    # dir the returned model references) are removed rather than leaked.
    atexit.register(shutil.rmtree, str(model_dir), ignore_errors=True)
    model_xml_path = model_dir / "model_only.xml"
    load_combined(msk_key, device_key, export_xml=str(model_xml_path))
    model_root = ET.parse(model_xml_path).getroot()
    # meshes are written relative to the export dir -> resolve to the (absolute,
    # persistent) install location so the temp export dir can be removed.
    _absolutize_files(model_root, model_xml_path.parent)

    # 2) build the terrain (flat default, a terrains JSON path, or an inline config dict).
    if terrain is None:
        cfg = _flat_default_config()
    elif isinstance(terrain, dict):
        cfg = config_from_dict(terrain)
    else:
        cfg = load_config(Path(terrain))
    # Terrain assets (hfield PNGs / textures) must outlive this call.  When
    # exporting, keep them beside the export; otherwise use a persistent temp
    # dir (the flat default writes no assets, so this is a no-op for RL runs).
    if export_path is not None:
        export_path = Path(export_path)
        assets_dir = export_path.resolve().parent / f"{export_path.stem}_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
    else:
        assets_dir = Path(tempfile.mkdtemp(prefix="myoassist_terrain_"))
        # Temp terrain-asset dir must outlive this call (the returned model
        # references its files), so clean it at process exit, not immediately.
        atexit.register(shutil.rmtree, str(assets_dir), ignore_errors=True)
    tspec = build_terrain(cfg, output_dir=assets_dir)
    tspec.compile()
    terrain_root = ET.fromstring(tspec.to_xml())
    _absolutize_files(terrain_root, assets_dir)
    terrain_geom_names = {g.get("name") for g in terrain_root.find("worldbody").iter("geom") if g.get("name")}

    # 3) merge terrain (colliding) + add default lighting + horizon haze.
    _inject_terrain(model_root, terrain_root)
    _inject_render_defaults(model_root)
    _inject_terrain_haze(model_root, terrain_root)

    # 4) seat the model so its feet rest on the terrain surface (light contact),
    #    measured from the real collision geometry at the opening pose.
    dz = _seat_dz_by_collision(ET.tostring(model_root, encoding="unicode"), terrain_geom_names)
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
