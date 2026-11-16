"""Train/evaluate project Deep CFR; checkpoints are trusted local artifacts."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata as md
from pathlib import Path
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
import numpy as np
from deep_cfr import Config, DeepCFR
from evaluation import evaluate
from experiment_io import save_csv, save_json


def run_seed(args, seed):
    if args.resume:
        solver, state = DeepCFR.load(args.resume, device=args.device)
        if not isinstance(state, dict) or state.get("strategy_schedule") != "every_iteration":
            raise ValueError("Checkpoint does not use the current every-iteration strategy training schedule")
        seed = solver.config.seed
        rows = state["rows"]
        eval_every = state["eval_every"]
        train_seconds = state["train_seconds"]
        average_seconds = state["average_seconds"]
    else:
        config = Config(game=args.game, seed=seed, hidden=args.hidden, traversals=args.traversals,
            capacity=args.capacity, batch_size=args.batch_size, advantage_steps=args.advantage_steps,
            strategy_steps=args.strategy_steps, learning_rate=args.learning_rate, device=args.device)
        solver = DeepCFR(config)
        rows, train_seconds, average_seconds = [], 0., 0.
        eval_every = args.eval_every
    if args.iterations <= solver.iteration:
        raise ValueError("Target iterations must exceed checkpoint iteration")
    output = args.output / f"seed-{seed}"
    output.mkdir(parents=True)
    start_iteration = solver.iteration
    for iteration in range(start_iteration + 1, args.iterations + 1):
        stats = solver.step()
        train_seconds += stats["traversal_seconds"] + stats["advantage_seconds"]
        # Strategy fitting is part of training, independent of evaluation cadence.
        average = solver.train_average()
        train_seconds += average["seconds"]
        average_seconds += average["seconds"]
        if iteration == 1 or iteration % eval_every == 0 or iteration == args.iterations:
            start = time.perf_counter()
            table = solver.average_policy()  # validates all rows before exact evaluation
            metrics = evaluate(solver.game, table)
            eval_seconds = time.perf_counter()-start
            rows.append({"iteration": iteration, "train_seconds": train_seconds,
                "average_training_seconds": average_seconds, "eval_seconds": eval_seconds,
                "node_visits": solver.node_visits, "gradient_steps": solver.gradient_steps,
                "strategy_gradient_steps": solver.strategy_gradient_steps,
                "advantage_loss_p0": stats["players"][0]["last_loss"],
                "advantage_loss_p1": stats["players"][1]["last_loss"],
                "strategy_loss": average["last_loss"], "strategy_samples_seen": solver.strategy_memory.seen,
                "strategy_buffer_size": len(solver.strategy_memory), **metrics})
            save_csv(output / "metrics.csv", rows)
            save_json(output / "training.json", solver.history)
            solver.save(output / "checkpoint.pt", {"rows": rows, "eval_every": eval_every,
                "train_seconds": train_seconds, "average_seconds": average_seconds,
                "strategy_schedule": "every_iteration"})
            if args.keep_checkpoints:
                shutil.copy2(output / "checkpoint.pt", output / f"iteration-{iteration:06d}.pt")
            print(f"{solver.config.game} seed={seed} iter={iteration}: exploitability={metrics['exploitability']:.6f}, train={train_seconds:.2f}s, strategy_loss={average['last_loss']:.6f}", flush=True)
    save_json(output / "average_policy.json", {"game": str(solver.game), "state_lookup": table.state_lookup,
        "action_probabilities": table.action_probability_array.tolist()})
    # Verify the produced checkpoint and exported strategy before declaring success.
    restored, _ = DeepCFR.load(output / "checkpoint.pt", device=solver.config.device)
    np.testing.assert_array_equal(table.action_probability_array, restored.average_policy().action_probability_array)
    root = Path(__file__).resolve().parent
    save_json(output / "metadata.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(solver.config), "target_iterations": args.iterations, "eval_every": eval_every,
        "interpreter": sys.executable, "python": platform.python_version(), "platform": platform.platform(),
        "versions": {p: md.version(p) for p in ("open-spiel", "torch", "numpy")},
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("deep_cfr.py", "reservoir.py", "train_deep_cfr.py", "evaluation.py", "experiment_io.py")},
        "resume_from": str(args.resume) if args.resume else None, "start_iteration": start_iteration,
        "checkpoint_roundtrip_exact": True,
        "timing": "traversal + advantage reset/training + all average network training; excludes initialization, exact evaluation, checkpoint I/O and plotting",
        "rng": "Owned traversal/init/batch/reservoir streams; isolated torch initialization; no stochastic torch layers",
        "averaging": "Strategy MLP warm-start training every iteration; losses weighted t/current_iteration",
        "inference_cache": "Only visited information sets; cleared at each player pass and after network updates",
        "checkpoint_scope": "Between completed iterations; includes networks, Adam, buffers/seen, owned RNG and logs"})
    return seed, rows


def plot_and_summary(output, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for seed, rows in runs:
        for ax, key in zip(axes, ("iteration", "train_seconds")):
            ax.plot([r[key] for r in rows], [r["exploitability"] for r in rows], marker=".", label=f"seed {seed}")
    for ax, label in zip(axes, ("Outer iterations", "Training seconds (including strategy MLP)")):
        ax.set(xlabel=label, ylabel="Exploitability (chips / hand)", yscale="log")
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.savefig(output / "convergence.png", dpi=160)
    plt.close(fig)
    values = [rows[-1]["exploitability"] for _, rows in runs]
    save_json(output / "summary.json", {"seeds": [seed for seed, _ in runs], "final_exploitabilities": values,
        "mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)) if len(values)>1 else None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("kuhn_poker", "leduc_poker"), default="kuhn_poker")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 99, 123, 2026])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--traversals", type=int, default=200)
    parser.add_argument("--capacity", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--advantage-steps", type=int, default=300)
    parser.add_argument("--strategy-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", type=Path, help="Trusted local checkpoint; uses stored config and evaluation schedule")
    parser.add_argument("--keep-checkpoints", action="store_true", help="Keep numbered evaluation snapshots as well as the latest checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.eval_every < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive iteration counts and unique seeds required")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Choose an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    runs = [run_seed(args, seed) for seed in ([args.seeds[0]] if args.resume else args.seeds)]
    plot_and_summary(args.output, runs)
