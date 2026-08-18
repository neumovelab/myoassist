"""Shared model compose + load pipeline for MyoAssist.

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
import hashlib
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

import mujoco as mj
import numpy as np

from assist_sim import load_combined
from myoassist_terrains import build_terrain
from myoassist_terrains.config import config_from_dict, load_config
from myoassist_terrains.surface import TerrainSurface

# Seat the lowest collidable geom this far *below* the terrain surface so the
# model opens in light contact (MuJoCo needs penetration, not just touching, to
# register a contact).  ~5 mm matches the validated merge prototype.
_CONTACT_SEAT_DEPTH = 0.005

# Opt every caller into the compose cache without threading a ``cache_dir`` through each
# config.  An explicit ``cache_dir=`` argument still wins over the variable.
CACHE_DIR_ENV = "MYOASSIST_CACHE_DIR"


def _terrain_fingerprint(terrain) -> str:
    """A stable string for the terrain argument, for the cache key.

    A dict is serialised with sorted keys; a path contributes its resolved location *and*
    mtime, so editing a terrain JSON invalidates the entry.
    """
    if terrain is None:
        return "flat-default"
    if isinstance(terrain, dict):
        return json.dumps(terrain, sort_keys=True)
    path = Path(terrain).resolve()
    stamp = path.stat().st_mtime_ns if path.exists() else "missing"
    return f"{path}@{stamp}"


def _compose_cache_key(msk_key: str, device_key: str, terrain, planar_root: bool) -> str:
    """Hash everything that changes the merged model.

    Three source tokens, not just the arguments, because every one of these repositories can
    change the output while the arguments stay identical:

    - **assist_sim**, which builds the human + device half.  Without this, the cache would
      serve a model from an older combine pipeline while assist_sim's own cache correctly
      rebuilt.
    - **myoassist_terrains**, which builds the ground.  Editing the terrain package changes
      the geometry for an unchanged terrain *config*, so the config fingerprint alone is not
      enough.
    - **this module**, where the terrain merge, the render defaults and the seating live.

    ``_package_token`` is version + newest source mtime, so an editable checkout invalidates
    on every edit rather than only on a release bump.
    """
    from assist_sim.loading import _package_token

    import assist_sim
    import myoassist_terrains

    parts = [
        msk_key,
        device_key,
        _terrain_fingerprint(terrain),
        f"planar={bool(planar_root)}",
        f"assist_sim@{_package_token(assist_sim)}",
        f"terrains@{_package_token(myoassist_terrains)}",
        f"compose@{Path(__file__).resolve().stat().st_mtime_ns}",
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


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


# How far below the model's lowest point the terrain is probed for a footprint,
# so a foot resting on a stepping stone or an obstacle is seated on top of it
# rather than inside it. Roughly a foot half-length.
_FOOTPRINT_RADIUS = 0.12
# Only geometry within this much of the model's lowest point can be what rests on
# the ground; probing the whole body wastes time and lets a hand or a backpack
# decide the seating.
_SEATING_BAND = 0.40


# At most this many contact candidates per geom, and only those within
# _CONTACT_PATCH of that geom's own lowest point. A sole rests on its contact
# patch, not on its whole mesh, and querying every vertex of every mesh made
# seating cost seconds instead of milliseconds.
_MAX_POINTS_PER_GEOM = 64
_CONTACT_PATCH = 0.02
# How many of the deepest candidates get the (more expensive) footprint query.
_REFINE_COUNT = 24


def _model_ground_candidates(model: mj.MjModel, data: mj.MjData, terrain_ids: set) -> np.ndarray:
    """World points on the model that could touch the ground, as (N, 3).

    Mesh geoms contribute the vertices in their own lowest `_CONTACT_PATCH`, so a
    tilted or contoured sole is measured from its real surface rather than a
    bounding box, without paying for the whole mesh. Primitives contribute their
    lowest point. Only the lowest `_SEATING_BAND` of the model is considered, so a
    hand or a backpack cannot decide the seating.
    """
    points: list[np.ndarray] = []
    for i in range(model.ngeom):
        if i in terrain_ids:
            continue
        if model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[i]
            first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[first : first + count]
            world = data.geom_xpos[i] + local @ data.geom_xmat[i].reshape(3, 3).T
            patch = world[world[:, 2] <= world[:, 2].min() + _CONTACT_PATCH]
            if len(patch) > _MAX_POINTS_PER_GEOM:
                patch = patch[np.argsort(patch[:, 2])[:_MAX_POINTS_PER_GEOM]]
            points.append(patch)
        else:
            low = data.geom_xpos[i].copy()
            low[2] -= float(np.max(model.geom_size[i]))
            points.append(low.reshape(1, 3))
    if not points:
        return np.empty((0, 3))
    stacked = np.vstack(points)
    return stacked[stacked[:, 2] <= stacked[:, 2].min() + _SEATING_BAND]


def _seat_dz_by_terrain(
    merged_xml: str,
    terrain_config,
    terrain_geom_names: set,
    penetration: float = _CONTACT_SEAT_DEPTH,
) -> float:
    """Vertical shift (m) that seats the model on the terrain in light contact.

    Asks ``myoassist_terrains`` how high the ground is under each candidate contact
    point and takes the tightest clearance, so after shifting the closest point
    sits ``penetration`` below the surface.

    Two passes. A cheap point query over every candidate finds the region that
    matters, then the deepest ``_REFINE_COUNT`` are re-measured with a footprint
    query -- a foot has extent, so the point between two stepping stones is not
    where it rests. Doing the footprint query for every candidate is 20x the cost
    for the same answer.

    This replaces a collision probe that gave every terrain geom a 50 m contact
    margin and read ``min(contact.dist)``. At that margin MuJoCo's mesh-versus-box
    narrowphase stops returning a physical separation -- the same query would
    report a tibia 1.3 m clear while the thorax read 0.4 m penetrated -- so the
    minimum landed on the wrong pair. Composed models came out buried 1.6-2.6 m,
    or 24 m in the air, silently. Only plane-based terrain escaped it, because
    plane-versus-mesh distance stays exact at any margin.

    The terrain package knows its own surface analytically, so no margin, no
    narrowphase and no MuJoCo version sensitivity are involved.
    """
    model = mj.MjModel.from_xml_string(merged_xml)
    data = mj.MjData(model)
    terrain_ids = {i for i in range(model.ngeom) if (model.geom(i).name or "") in terrain_geom_names}
    if not terrain_ids:
        return 0.0
    if model.nkey > 0:
        mj.mj_resetDataKeyframe(model, data, 0)  # keyframe 0 == "stand"
    else:
        data.qpos[:] = model.qpos0
    mj.mj_forward(model, data)

    candidates = _model_ground_candidates(model, data, terrain_ids)
    if candidates.size == 0:
        return 0.0

    surface = TerrainSurface(terrain_config)
    point_clearance = np.array([float(pz) - surface.height_at(float(px), float(py)) for px, py, pz in candidates])
    deepest = np.argsort(point_clearance)[:_REFINE_COUNT]
    clearance = min(
        float(candidates[i][2]) - surface.max_height_in(float(candidates[i][0]), float(candidates[i][1]), _FOOTPRINT_RADIUS)
        for i in deepest
    )
    # new_clearance = clearance + dz ; want new_clearance == -penetration.
    return -penetration - clearance


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
    planar_root: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
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
    planar_root : bool
        CO-only.  Re-orient a freejoint 3D-lineage leg MSK (``myolegs26``,
        ``myolegs``) to the ``myolegs22`` frame and swap the freejoint for the
        named pelvis DOF joints, so the reflex controller can drive it.  No-op on
        the planar ``myolegs22``.  Leave False for RL (keeps the freejoint base).
    cache_dir : str | Path | None
        Opt in to caching the *merged* MJCF, keyed on everything that changes it
        (see :func:`_compose_cache_key`).  Also read from the ``MYOASSIST_CACHE_DIR``
        environment variable, which this argument overrides.

        This is worth doing because the model is composed far more often than once
        per run: ``func_Walk_FitCost`` builds a ``myoLeg_reflex`` per CMA candidate,
        so a CO run at the shipped ``--popsize 32 --maxiter 1000`` composes tens of
        thousands of times, and an RL run composes once per ``SubprocVecEnv`` worker.

        Measured on ``myolegs22`` (best of five): composing costs 0.65-0.83 s per
        call against 0.04-0.07 s for a cache hit, which is parity with the static
        model files MyoAssist 0.1 shipped (0.037-0.068 s for the same three
        devices).  Almost all of the remaining cost is the ``from_xml_string`` the
        env performs either way; the cache read itself is ~0.2 ms.

    Returns
    -------
    str
        The merged model as an MJCF XML string (env input via
        ``MjModel.from_xml_string``).
    """
    cache_dir = cache_dir if cache_dir is not None else os.environ.get(CACHE_DIR_ENV) or None
    if cache_dir is not None:
        return _compose_env_model_cached(msk_key, device_key, terrain, export_path, planar_root, Path(cache_dir))
    return _compose_env_model(msk_key, device_key, terrain, export_path, planar_root)


def _compose_env_model_cached(
    msk_key: str,
    device_key: str,
    terrain,
    export_path,
    planar_root: bool,
    cache_dir: Path,
) -> str:
    """Serve the merged MJCF from *cache_dir*, composing only on a miss.

    An entry is a single file plus, when the terrain writes assets, shared files in
    ``cache_dir/terrain_assets``. Model mesh paths are already absolute inside the
    installed ``assist_sim``. Uniform terrains (flat / slope / random / sinusoidal)
    bake their geometry into the spec and write nothing; a *tiled* terrain with a
    ``rough`` tile writes a heightmap PNG, and that file has to outlive the process
    that composed it or a later cache hit returns a model pointing at a deleted temp
    dir -- which failed hard with ``Error opening file``. So the assets live beside
    the cache, not in a per-process temp dir. Their names are content-addressed by
    the terrain package, so distinct terrains never collide there and identical
    heightmaps are stored once.

    Writes go to a per-writer ``.partial`` name published with ``os.replace``, because the
    case this cache exists for is N processes starting at once against a cold cache -- an RL
    launch, or several CO runs sharing a directory.  Last writer wins and they all produce
    the same bytes.  An unreadable entry is treated as a miss and discarded, since a cache
    should never be the reason a run fails.
    """
    key = _compose_cache_key(msk_key, device_key, terrain, planar_root)
    entry = cache_dir / f"{key}.xml"

    if entry.exists():
        try:
            xml_str = entry.read_text(encoding="utf-8")
        except OSError:
            xml_str = ""
        if xml_str.lstrip().startswith("<mujoco"):
            if export_path is not None:
                Path(export_path).write_text(xml_str, encoding="utf-8")
            return xml_str
        entry.unlink(missing_ok=True)

    xml_str = _compose_env_model(
        msk_key, device_key, terrain, export_path, planar_root, assets_dir=cache_dir / "terrain_assets"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    staged = cache_dir / f"{key}.{os.getpid()}.{uuid4().hex[:8]}.partial"
    staged.write_text(xml_str, encoding="utf-8")
    os.replace(staged, entry)
    return xml_str


def _compose_env_model(
    msk_key: str,
    device_key: str,
    terrain: Optional[Union[str, Path, dict]] = None,
    export_path: Optional[Union[str, Path]] = None,
    planar_root: bool = False,
    assets_dir: Optional[Path] = None,
) -> str:
    """The uncached compose.  See :func:`compose_env_model` for the parameters.

    ``assets_dir`` overrides where terrain assets are written and, when given,
    marks that directory as persistent (no exit-time cleanup). The cache passes
    its own so entries stay loadable in later processes.
    """
    # 1) compose human + device -> reload-safe, model-only XML (scene stripped).
    model_dir = Path(tempfile.mkdtemp(prefix="myoassist_compose_"))
    # The immediate rmdir below only clears empty scratch dirs; register a
    # process-exit cleanup so any that still hold files (e.g. a terrain assets
    # dir the returned model references) are removed rather than leaked.
    atexit.register(shutil.rmtree, str(model_dir), ignore_errors=True)
    model_xml_path = model_dir / "model_only.xml"
    load_combined(msk_key, device_key, export_xml=str(model_xml_path), planar_root=planar_root)
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
    if assets_dir is not None:
        # Caller-owned and persistent (the compose cache): the files have to
        # survive this process for a later cache hit to load.
        assets_dir = Path(assets_dir)
        assets_dir.mkdir(parents=True, exist_ok=True)
    elif export_path is not None:
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
    dz = _seat_dz_by_terrain(ET.tostring(model_root, encoding="unicode"), cfg, terrain_geom_names)
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
