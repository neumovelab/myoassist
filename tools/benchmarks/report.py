"""Print every benchmark result in one directory as Markdown tables.

Reads the JSON files that sim.py, train.py, build.py and co.py write (each carries a
`benchmark` field) and prints, for whatever is present: the system, single-core simulation speed,
activation sensitivity, the device comparison, concurrent simulation, the replay validation,
training throughput, build time and CO throughput.

Usage:
    python tools/benchmarks/report.py tools/benchmarks/results/<run_dir>
"""

import json
import pathlib
import sys

import common

LABEL = {msk: str(n) for msk, n in common.MUSCLES.items()}


def pm(s, fmt="{:,.0f}"):
    if s is None or s["mean"] is None:
        return ""
    return fmt.format(s["mean"]) + ("" if s["sd"] is None else " ± " + fmt.format(s["sd"]))


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def by_order(keys, field):
    order = {m: i for i, m in enumerate(common.MSKS)}
    return sorted(keys, key=lambda k: order[field(k)])


def sim(docs):
    single = [d for d in docs if d["settings"]["concurrent"] == 1]
    for d in single:
        dev = d["settings"]["device"]
        print(f"## Simulation, 1 pinned core, device {dev}\n")
        rows = []
        for k in by_order(d["summary"], lambda k: d["summary"][k]["msk"]):
            s = d["summary"][k]
            rows.append(
                [
                    LABEL[s["msk"]],
                    f"{s['activation']:g}",
                    pm(s["physics_steps_per_s"]),
                    pm(s["rtf"], "{:.2f}"),
                    pm(s["wall_per_sim_s"], "{:.4f}"),
                    s["mean_ncon"],
                    s["mean_nefc"],
                ]
            )
        table(["muscles", "activation", "physics steps/s", "RTF", "wall s per sim s", "contacts", "constraint rows"], rows)

    with_dev = [d for d in single if d["settings"]["device"] != "none"]
    bare = [d for d in single if d["settings"]["device"] == "none"]
    if with_dev and bare:
        dev, none = with_dev[0]["summary"], bare[0]["summary"]
        print(f"## Device comparison, 1 pinned core ({with_dev[0]['settings']['device']} vs none)\n")
        rows = []
        for k in by_order([k for k in dev if k in none], lambda k: dev[k]["msk"]):
            a, b = dev[k]["physics_steps_per_s"]["mean"], none[k]["physics_steps_per_s"]["mean"]
            rows.append(
                [
                    LABEL[dev[k]["msk"]],
                    pm(dev[k]["physics_steps_per_s"]),
                    pm(none[k]["physics_steps_per_s"]),
                    f"{100 * (a / b - 1):+.1f}%",
                    f"{dev[k]['mean_ncon']} / {none[k]['mean_ncon']}",
                    f"{dev[k]['mean_nefc']} / {none[k]['mean_nefc']}",
                ]
            )
        table(["muscles", "with device", "without", "change", "contacts (with / without)", "constraint rows"], rows)

    conc = sorted([d for d in docs if d["settings"]["concurrent"] > 1], key=lambda d: d["settings"]["concurrent"])
    if conc:
        print("## Simulation, N concurrent processes\n")
        print("Aggregate control steps/s (per-process physics steps/s in parentheses).\n")
        keys = by_order(conc[0]["summary"], lambda k: conc[0]["summary"][k]["msk"])
        rows = []
        for k in keys:
            cells = [LABEL[conc[0]["summary"][k]["msk"]]]
            for d in conc:
                s = d["summary"][k]
                cells.append(f"{s['aggregate_control_steps_per_s']:,.0f} ({s['physics_steps_per_s']['mean']:,.0f})")
            rows.append(cells)
        table(["muscles"] + [f"N = {d['settings']['concurrent']}" for d in conc], rows)


def validate(docs):
    for d in docs:
        print(f"## Replay vs closed-loop policy ({d['settings']['msk']}, same process, interleaved)\n")
        print(f"- closed-loop replay drift (max |qpos| error): {d['closed_loop_replay_max_qpos_drift']}")
        print(f"- RL step breakdown (ms): {d['rl_step_breakdown_ms']}\n")
        ref = d["summary"]["closed_loop_policy"]["us_per_step"]["mean"]
        rows = [
            [
                name,
                pm(s["us_per_step"], "{:.2f}"),
                f"{100 * (s['us_per_step']['mean'] / ref - 1):+.1f}%",
                s["mean_ncon"]["mean"],
            ]
            for name, s in d["summary"].items()
        ]
        table(["source", "us per physics step", "vs closed loop", "contacts"], rows)


def train(docs):
    for d in docs:
        st = d["settings"]
        title = "env stepping only (random actions)" if st["env_only"] else "PPO training"
        print(f"## {title}: init_policy={st['init_policy']}\n")
        rows = []
        for s in d["summary"].values():
            row = [LABEL.get(s["msk"], s["msk"]), s["num_envs"], s["threads"], pm(s["env_steps_per_s"], "{:,.1f}")]
            if not st["env_only"]:
                row += [
                    pm(s["collect_steps_per_s"], "{:,.1f}"),
                    pm(s["update_s_per_iter"], "{:.2f}"),
                    pm(s["epochs_per_iter"], "{:.1f}"),
                    pm(s["mean_episode_len"], "{:.2f}"),
                ]
            rows.append(row)
        header = ["muscles", "envs", "threads", "env steps/s"]
        if not st["env_only"]:
            header += ["collect steps/s", "update s/iter", "epochs/iter", "episode length"]
        table(header, rows)


def build(docs):
    for d in docs:
        print("## Model build time (s): compose + MuJoCo compile\n")
        rows = []
        for k in by_order(d["summary"], lambda k: d["summary"][k]["msk"]):
            s = d["summary"][k]
            rows.append(
                [
                    LABEL[s["msk"]],
                    s["device"],
                    s["build"],
                    s["cache"],
                    pm(s["cold_total_s"], "{:.3f}"),
                    pm(s["warm_total_s"], "{:.3f}"),
                    pm(s["warm_compose_s"], "{:.3f}"),
                    pm(s["warm_compile_s"], "{:.3f}"),
                ]
            )
        table(
            ["muscles", "device", "build", "cache", "first build", "repeated build", "repeated: compose", "repeated: compile"],
            rows,
        )


def co(docs):
    for d in docs:
        print("## Controller optimization (CMA-ES rollouts)\n")
        rows = []
        for k in by_order(d["summary"], lambda k: d["summary"][k]["msk"]):
            s = d["summary"][k]
            rows.append(
                [
                    LABEL[s["msk"]],
                    s["exo"],
                    s["procs"] or "serial",
                    s["cache"],
                    pm(s["rollouts_per_s"], "{:.1f}"),
                    pm(s["stepping_physics_steps_per_s"]),
                    pm(s["physics_steps_per_s"]),
                    pm(s["build_s_mean"], "{:.3f}"),
                    pm(s["sim_s_per_rollout_mean"], "{:.2f}"),
                    pm(s["frac_rejected_at_pose_check"], "{:.2f}"),
                ]
            )
        header = ["muscles", "exo", "processes", "cache", "rollouts/s", "stepping physics steps/s (1 process)"]
        header += ["physics steps/s (all)", "build s", "sim s per rollout", "rejected"]
        table(header, rows)


def main():
    d = pathlib.Path(sys.argv[1])
    docs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    docs = [doc for doc in docs if "benchmark" in doc]
    assert docs, f"no benchmark results in {d}"
    print("## System\n")
    for k, v in docs[0]["system"].items():
        print(f"- {k}: {v}")
    print()
    for name, fn in (("sim", sim), ("validate", validate), ("train", train), ("build", build), ("co", co)):
        group = [doc for doc in docs if doc["benchmark"] == name]
        if group:
            fn(group)


if __name__ == "__main__":
    main()
