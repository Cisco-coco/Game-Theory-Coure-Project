"""Reproducible local/reference CFR comparison. See THIRD_PARTY.md."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import cfr
from tabular_cfr import TabularCFR
from evaluation import evaluate
from experiment_io import save_json as write_json


FACTORIES = {
    "local_cfr": lambda game: TabularCFR(game),
    "local_cfr_plus": lambda game: TabularCFR(game, plus=True),
    "official_cfr": cfr.CFRSolver,
    "official_cfr_plus": cfr.CFRPlusSolver,
}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    game = pyspiel.load_game(args.game, {"players": 2})
    rows, policies, setup_times = [], {}, {}
    baseline = evaluate(game, policy.UniformRandomPolicy(game))
    for name, factory in FACTORIES.items():
        started = time.perf_counter()
        solver = factory(game)
        setup_times[name] = time.perf_counter() - started
        train_seconds = 0.
        for iteration in range(1, args.iterations + 1):
            started = time.perf_counter()
            solver.evaluate_and_update_policy()
            train_seconds += time.perf_counter() - started
            if iteration == 1 or iteration % args.eval_every == 0 or iteration == args.iterations:
                started = time.perf_counter()
                metrics = evaluate(game, solver.average_policy())
                rows.append(dict(algorithm=name, iteration=iteration, train_seconds=train_seconds,
                                 eval_seconds=time.perf_counter() - started,
                                 node_visits=getattr(solver, "node_visits", ""), **metrics))
                print(f"{args.game} {name} {iteration}/{args.iterations}: {metrics['exploitability']:.9f}", flush=True)
        average = solver.average_policy()
        policies[name] = average.action_probability_array.copy()
        write_json(args.output / f"{name}_policy.json", {
            "game": str(game), "state_lookup": average.state_lookup,
            "action_probabilities": policies[name].tolist()})
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    differences = {}
    for suffix in ("cfr", "cfr_plus"):
        local = [r for r in rows if r["algorithm"] == "local_" + suffix]
        official = [r for r in rows if r["algorithm"] == "official_" + suffix]
        differences[suffix] = {
            "final_average_policy_max_abs_error": float(np.max(np.abs(policies["local_" + suffix] - policies["official_" + suffix]))),
            "checkpoint_exploitability_max_abs_error": max(abs(a["exploitability"] - b["exploitability"]) for a, b in zip(local, official))}
    write_json(args.output / "reference_comparison.json", differences)
    root = Path(__file__).resolve().parent
    source_paths = [root / name for name in ("tabular_cfr.py", "compare_tabular.py", "evaluation.py", "experiment_io.py")] + [Path(cfr.__file__)]
    write_json(args.output / "metadata.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "game": str(game),
        "parameters": game.get_parameters(), "iterations": args.iterations, "eval_every": args.eval_every,
        "interpreter": sys.executable, "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "versions": {p: importlib.metadata.version(p) for p in ("open-spiel", "numpy", "matplotlib")},
        "solver_setup_seconds": setup_times, "random_baseline": baseline,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "algorithm_config": {"cfr": {"alternating": True, "regret_matching_plus": False, "linear_averaging": False},
                             "cfr_plus": {"alternating": True, "regret_matching_plus": True, "linear_averaging": True, "averaging_delay": 0}},
        "timing": "Sequential runs, update calls only; excludes initialization, evaluation and I/O; one run, no timing confidence interval",
        "node_visits": "Local recursion entries including chance/terminal/pruned entries, cumulative; reference not instrumented",
        "seed": None, "seed_reason": "Deterministic full-tree solvers; no random sampling",
        "evaluation": "Exact NashConv/2 on average policies, chips per hand"})
    plot(rows, args.game, args.output, baseline["exploitability"])
    return differences


def plot(rows, game, output, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name in FACTORIES:
        selected = [r for r in rows if r["algorithm"] == name]
        color = "tab:orange" if name.endswith("plus") else "tab:blue"
        style = "--" if name.startswith("official") else "-"
        for ax, key in zip(axes, ("iteration", "train_seconds")):
            ax.plot([r[key] for r in selected], [r["exploitability"] for r in selected],
                    label=name, color=color, linestyle=style, linewidth=1.6)
    for ax, label in zip(axes, ("Iterations", "Training time (s), excluding evaluation")):
        ax.axhline(baseline, linestyle=":", color="gray", label="Uniform random")
        ax.set(xlabel=label, ylabel="Exploitability (chips / hand)", yscale="log")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{game}: local vs OpenSpiel (overlapping curves expected)")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=["kuhn_poker", "leduc_poker"], required=True)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.iterations, args.eval_every) < 1:
        parser.error("iterations and eval-every must be positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("choose an empty output directory")
    print(json.dumps(run(args), indent=2))
