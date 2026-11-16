"""Train external-sampling MCCFR with optional matched-budget tabular baselines."""
import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pyspiel
from open_spiel.python.algorithms import external_sampling_mccfr as reference
from external_mccfr import ExternalSamplingMCCFR
from evaluation import evaluate
from experiment_io import save_json, save_csv


def summarize(rows, baseline, output, game_name):
    aggregates = []
    for iteration in sorted({r["iteration"] for r in rows}):
        group = [r for r in rows if r["iteration"] == iteration]
        aggregates.append({"iteration": iteration, "seeds": len(group),
            **{f"{metric}_{stat}": float(fn([r[metric] for r in group]))
               for metric in ("exploitability", "train_seconds", "node_visits")
               for stat, fn in (("mean", np.mean), ("std", lambda x: np.std(x, ddof=1)))}})
    save_csv(output / "aggregate.csv", aggregates)
    base = []
    if baseline:
        with (baseline / "metrics.csv").open(encoding="utf-8") as f:
            base = [{k: (float(v) if k != "algorithm" and v else v) for k, v in r.items()}
                    for r in csv.DictReader(f) if r["algorithm"].startswith("local_")]
        info = json.loads((baseline / "metadata.json").read_text(encoding="utf-8"))
        if info["game"] != f"{game_name}(players=2)" or info["versions"]["open-spiel"] != md.version("open-spiel"):
            raise ValueError("Baseline game/OpenSpiel version does not match")
        # Discrete, hold-last-checkpoint comparison. No interpolation or extrapolation.
        budget_rows = []
        for metric in ("train_seconds", "node_visits"):
            groups = {name: [r for r in base if r["algorithm"] == name] for name in ("local_cfr", "local_cfr_plus")}
            groups.update({f"mccfr_seed_{seed}": [r for r in rows if r["seed"] == seed] for seed in sorted({r["seed"] for r in rows})})
            budget = min(g[-1][metric] for g in groups.values())
            for name, group in groups.items():
                eligible = [r for r in group if r[metric] <= budget]
                if eligible:
                    selected = eligible[-1]
                    budget_rows.append({"budget_metric": metric, "budget": budget, "algorithm": name,
                        "checkpoint_actual_cost": selected[metric], "iteration": selected["iteration"],
                        "exploitability": selected["exploitability"]})
        if budget_rows:
            save_csv(output / "matched_budget.csv", budget_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, key, label in zip(axes, ("iteration", "train_seconds", "node_visits"),
                             ("Iterations (different work per iteration)", "Training seconds", "Training node visits")):
        for seed in sorted({r["seed"] for r in rows}):
            group = [r for r in rows if r["seed"] == seed]
            ax.plot([r[key] for r in group], [r["exploitability"] for r in group], color="tab:green", alpha=.17)
        x = [r["iteration"] if key == "iteration" else r[key + "_mean"] for r in aggregates]
        y = np.array([r["exploitability_mean"] for r in aggregates])
        sd = np.array([r["exploitability_std"] for r in aggregates])
        ax.plot(x, y, color="tab:green", label=f"MCCFR mean ({len({r['seed'] for r in rows})} seeds)")
        ax.fill_between(x, np.maximum(y-sd, 1e-10), y+sd, color="tab:green", alpha=.15, label="+/- 1 sample SD")
        for name, color in (("local_cfr", "tab:blue"), ("local_cfr_plus", "tab:orange")):
            group = [r for r in base if r["algorithm"] == name]
            if group:
                ax.plot([r[key] for r in group], [r["exploitability"] for r in group], color=color, label=name)
        ax.set(xlabel=label, ylabel="Exploitability (chips / hand)", xscale="log", yscale="log")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{game_name}: external-sampling MCCFR (SD band aligned by iteration)")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    game = pyspiel.load_game(args.game, {"players": 2})
    checkpoints = {1, 10, 100, args.iterations} | set(range(args.eval_every, args.iterations + 1, args.eval_every))
    rows, setup = [], {}
    for seed in args.seeds:
        start = time.perf_counter()
        solver = ExternalSamplingMCCFR(game, seed)
        setup[seed] = time.perf_counter() - start
        train_seconds = 0.
        for iteration in range(1, args.iterations + 1):
            start = time.perf_counter()
            solver.evaluate_and_update_policy()
            train_seconds += time.perf_counter() - start
            if iteration in checkpoints:
                start = time.perf_counter()
                metrics = evaluate(game, solver.average_policy())
                rows.append({"seed": seed, "iteration": iteration, "train_seconds": train_seconds,
                             "eval_seconds": time.perf_counter()-start, **solver.counters(), **metrics})
        save_csv(args.output / f"seed-{seed}.csv", [r for r in rows if r["seed"] == seed])
        save_json(args.output / f"seed-{seed}-policy.json", {"game": str(game), "seed": seed,
            "unvisited_information_sets": "uniform legal actions", "nodes": [
                {"player": key[0], "information_state": key[1], "actions": node[0],
                 "probabilities": (node[2]/node[2].sum()).tolist()} for key, node in solver.nodes.items()]})
        print(f"{args.game} seed={seed}: exploitability={rows[-1]['exploitability']:.7f}, train={train_seconds:.2f}s, nodes={solver.node_visits}", flush=True)
    save_csv(args.output / "metrics.csv", rows)
    root = Path(__file__).resolve().parent
    files = [root / n for n in ("external_mccfr.py", "train_mccfr.py", "evaluation.py", "experiment_io.py")] + [Path(reference.__file__)]
    save_json(args.output / "metadata.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable, "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "game": str(game), "parameters": game.get_parameters(),
        "iterations": args.iterations, "seeds": args.seeds, "checkpoints": sorted(c for c in checkpoints if c <= args.iterations),
        "versions": {p: md.version(p) for p in ("open-spiel", "numpy", "matplotlib")},
        "solver_setup_seconds": setup, "baseline": str(args.baseline.resolve()) if args.baseline else None,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "averaging": "two-player SIMPLE; opponent-node strategy accumulation; initial pseudocount 1e-6",
        "rng": "independent numpy RandomState per solver",
        "timing": "update calls only; excludes setup, exact evaluation, snapshot copying, I/O and plots",
        "node_visits": "all recursive entries including chance and terminal, training only",
        "uncertainty": "sample SD over seeds, not confidence interval; band aligned by iteration; time/node x values are seed means",
        "matched_budget": "last recorded checkpoint at or below minimum final budget across all runs, no interpolation",
        "evaluation": "exact NashConv/2 on average policies"})
    summarize(rows, args.baseline, args.output, args.game)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("kuhn_poker", "leduc_poker"), required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 99, 123, 2026])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.iterations, args.eval_every) < 1 or len(set(args.seeds)) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive iterations/eval-every and at least two distinct seeds required")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("choose an empty output directory")
    run(args)
