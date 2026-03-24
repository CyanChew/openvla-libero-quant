"""Run large-scale OpenVLA quantization sweeps on LIBERO.

This script launches many `run_libero_eval.py` jobs with different module-level
quantization settings and saves each run's JSON metrics to a dedicated file.
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_list(raw, cast=str):
    values = []
    for item in raw.split(","):
        value = item.strip()
        if value:
            values.append(cast(value))
    return values


def select_unique_configs(base_experiments, max_unique_configs, strategy, sampling_seed):
    if max_unique_configs <= 0 or max_unique_configs >= len(base_experiments):
        return list(base_experiments)

    if strategy == "sequential":
        return base_experiments[:max_unique_configs]

    if strategy != "stratified":
        raise ValueError(
            f"Unsupported unique sampling strategy: {strategy}. "
            "Use one of: sequential, stratified"
        )

    rng = random.Random(int(sampling_seed))

    # Stratify by quant mode and llm target to avoid front-truncation bias.
    buckets = defaultdict(list)
    for exp in base_experiments:
        key = (exp["quant_mode"], exp["llm_target"])
        buckets[key].append(exp)

    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)
    for key in bucket_keys:
        rng.shuffle(buckets[key])

    selected = []
    active_keys = list(bucket_keys)
    while len(selected) < max_unique_configs and len(active_keys) > 0:
        progressed = False
        for key in list(active_keys):
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop())
                progressed = True
                if len(selected) >= max_unique_configs:
                    break
            if len(bucket) == 0:
                active_keys.remove(key)

        if not progressed:
            break

    # Fallback fill (should rarely trigger), still randomized by seed.
    if len(selected) < max_unique_configs:
        remaining = []
        for key in bucket_keys:
            remaining.extend(buckets[key])
        rng.shuffle(remaining)
        needed = max_unique_configs - len(selected)
        selected.extend(remaining[:needed])

    return selected


def build_experiment_grid(args):
    quant_modes = parse_list(args.quant_modes, str)
    llm_targets = parse_list(args.llm_targets, str)
    llm_ratios = parse_list(args.llm_ratios, float)
    llm_layer_selections = parse_list(args.llm_layer_selections, str)
    vision_targets = parse_list(args.vision_targets, str)
    action_targets = parse_list(args.action_targets, str)
    seeds = parse_list(args.seeds, int)

    if len(seeds) == 0:
        raise ValueError("At least one seed is required. Please provide --seeds.")

    repeats = int(args.repeat_per_config)
    if repeats <= 0:
        raise ValueError("--repeat_per_config must be >= 1")

    combinations = itertools.product(
        quant_modes,
        llm_targets,
        llm_ratios,
        llm_layer_selections,
        vision_targets,
        action_targets,
    )

    base_experiments = []
    for quant_mode, llm_target, llm_ratio, llm_layer_selection, vision_target, action_target in combinations:
        ratio_tag = int(round(llm_ratio * 100))
        base_tag = (
            f"{quant_mode}_llm-{llm_target}_r{ratio_tag}_sel-{llm_layer_selection}"
            f"_vis-{vision_target}_act-{action_target}"
        )
        base_experiments.append(
            {
                "quant_mode": quant_mode,
                "llm_target": llm_target,
                "llm_ratio": llm_ratio,
                "llm_layer_selection": llm_layer_selection,
                "vision_target": vision_target,
                "action_target": action_target,
                "base_tag": base_tag,
            }
        )

    if args.max_unique_configs > 0:
        base_experiments = select_unique_configs(
            base_experiments=base_experiments,
            max_unique_configs=args.max_unique_configs,
            strategy=args.unique_sampling_strategy,
            sampling_seed=args.sampling_seed,
        )

    experiments = []
    for base_exp in base_experiments:
        for repeat_idx in range(repeats):
            seed = seeds[repeat_idx % len(seeds)]
            exp = dict(base_exp)
            exp["seed"] = seed
            exp["repeat_idx"] = repeat_idx
            exp["run_tag"] = f"{base_exp['base_tag']}_rep-{repeat_idx}_seed-{seed}"
            experiments.append(exp)

    if args.max_experiments > 0:
        experiments = experiments[: args.max_experiments]

    return experiments


def build_eval_command(args, exp, metrics_path):
    cmd = [
        sys.executable,
        "experiments/robot/libero/run_libero_eval.py",
        "--model_family",
        "openvla",
        "--pretrained_checkpoint",
        args.pretrained_checkpoint,
        "--task_suite_name",
        args.task_suite_name,
        "--center_crop",
        str(args.center_crop),
        "--num_trials_per_task",
        str(args.num_trials_per_task),
        "--max_tasks_per_run",
        str(args.max_tasks_per_run),
        "--save_rollout_videos",
        str(args.save_rollout_videos),
        "--run_id_note",
        exp["run_tag"],
        "--seed",
        str(exp["seed"]),
        "--llm_quant_target",
        exp["llm_target"],
        "--llm_quant_ratio",
        str(exp["llm_ratio"]),
        "--llm_layer_selection",
        exp["llm_layer_selection"],
        "--vision_quant_target",
        exp["vision_target"],
        "--action_quant_target",
        exp["action_target"],
        "--metrics_output_path",
        str(metrics_path),
    ]

    if exp["quant_mode"] == "int4":
        cmd.extend(["--load_in_4bit", "True"])
    elif exp["quant_mode"] == "int8":
        cmd.extend(["--load_in_8bit", "True"])

    return cmd


def append_jsonl(path, payload):
    with open(path, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_last_lines(path, num_lines=30):
    if not path.exists():
        return ""
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()
    return "".join(lines[-num_lines:])


def render_progress(done, total, success, failed, skipped, start_time):
    ratio = (done / total) if total > 0 else 0.0
    bar_len = 32
    filled = int(round(ratio * bar_len))
    bar = "#" * filled + "-" * (bar_len - filled)
    elapsed = max(0.0, time.time() - start_time)
    avg_per_task = (elapsed / done) if done > 0 else 0.0
    eta = avg_per_task * max(0, total - done)
    print(
        f"[PROG] |{bar}| {done}/{total} ({ratio * 100:.1f}%) "
        f"ok={success} fail={failed} skip={skipped} "
        f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
        flush=True,
    )


def write_progress_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def build_progress_payload(total, done, success, failed, skipped, start_time, active, gpu_ids):
    return {
        "total": total,
        "done": done,
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "pending": max(0, total - done),
        "elapsed_sec": time.time() - start_time,
        "workers": [
            {
                "gpu_id": gid,
                "active": active[gid]["exp"]["run_tag"] if active.get(gid) is not None else None,
            }
            for gid in gpu_ids
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run OpenVLA quantization sweep.")
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--num_trials_per_task", type=int, default=5)
    parser.add_argument("--max_tasks_per_run", type=int, default=0)
    parser.add_argument("--save_rollout_videos", type=str2bool, default=False)
    parser.add_argument("--center_crop", type=str2bool, default=True)

    parser.add_argument("--quant_modes", type=str, default="int4,int8")
    parser.add_argument("--llm_targets", type=str, default="ffn_only,all")
    parser.add_argument("--llm_ratios", type=str, default="0.25,0.5,0.75,1.0")
    parser.add_argument("--llm_layer_selections", type=str, default="prefix,suffix,uniform")
    parser.add_argument("--vision_targets", type=str, default="none,projector_only")
    parser.add_argument("--action_targets", type=str, default="none,all")
    parser.add_argument("--seeds", type=str, default="7,17,27")
    parser.add_argument("--repeat_per_config", type=int, default=1)
    parser.add_argument("--max_unique_configs", type=int, default=0)
    parser.add_argument(
        "--unique_sampling_strategy",
        type=str,
        default="stratified",
        choices=["sequential", "stratified"],
        help=(
            "How to select unique configs when --max_unique_configs > 0. "
            "'stratified' is balanced over (quant_mode, llm_target)."
        ),
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=42,
        help="Random seed used by unique config sampling (stratified).",
    )

    parser.add_argument("--max_experiments", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="experiments/robot/libero/sweep_results")
    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument("--poll_interval_sec", type=float, default=2.0)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--dry_run", type=str2bool, default=False)

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir = output_dir / "run_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    progress_path = output_dir / "progress.json"
    experiments = build_experiment_grid(args)
    gpu_ids = parse_list(args.gpu_ids, int) if args.gpu_ids.strip() else []
    if len(gpu_ids) == 0:
        gpu_ids = [None]

    print(f"[INFO] Prepared {len(experiments)} experiments")
    print(f"[INFO] GPU workers: {gpu_ids}")
    if args.dry_run:
        for idx, exp in enumerate(experiments[:20]):
            print(f"[{idx:03d}] {exp['run_tag']}")
        if len(experiments) > 20:
            print("[INFO] ... (showing first 20)")
        return

    start_time = time.time()
    total = len(experiments)
    done = 0
    success = 0
    failed = 0
    skipped = 0

    pending = []
    for idx, exp in enumerate(experiments):
        metrics_path = output_dir / f"{idx:04d}__{exp['run_tag']}.json"
        if args.resume and metrics_path.exists():
            skipped += 1
            done += 1
            record = {
                "index": idx,
                "run_tag": exp["run_tag"],
                "metrics_path": str(metrics_path),
                "status": "skipped",
            }
            append_jsonl(manifest_path, record)
        else:
            pending.append((idx, exp, metrics_path))

    active = {gpu_id: None for gpu_id in gpu_ids}
    next_pending_idx = 0

    render_progress(done, total, success, failed, skipped, start_time)
    write_progress_json(
        progress_path,
        build_progress_payload(total, done, success, failed, skipped, start_time, active, gpu_ids),
    )

    while True:
        for gpu_id in gpu_ids:
            worker_state = active[gpu_id]

            if worker_state is None and next_pending_idx < len(pending):
                idx, exp, metrics_path = pending[next_pending_idx]
                next_pending_idx += 1

                command = build_eval_command(args, exp, metrics_path)
                env = os.environ.copy()
                existing_pythonpath = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f".{os.pathsep}{existing_pythonpath}" if existing_pythonpath else "."
                if gpu_id is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                log_path = run_logs_dir / f"{idx:04d}__{exp['run_tag']}.log"
                log_f = open(log_path, "w")
                process = subprocess.Popen(
                    command,
                    cwd=str(repo_root),
                    env=env,
                    stdout=log_f,
                    stderr=log_f,
                    text=True,
                )

                active[gpu_id] = {
                    "index": idx,
                    "exp": exp,
                    "metrics_path": metrics_path,
                    "log_path": log_path,
                    "log_file": log_f,
                    "process": process,
                    "start_time": time.time(),
                }

                gpu_text = "cpu/default" if gpu_id is None else f"gpu={gpu_id}"
                print(f"[RUN ] ({idx + 1}/{total}) {exp['run_tag']} [{gpu_text}]", flush=True)
                write_progress_json(
                    progress_path,
                    build_progress_payload(total, done, success, failed, skipped, start_time, active, gpu_ids),
                )

            worker_state = active[gpu_id]
            if worker_state is None:
                continue

            returncode = worker_state["process"].poll()
            if returncode is None:
                continue

            worker_state["log_file"].close()
            idx = worker_state["index"]
            exp = worker_state["exp"]
            metrics_path = worker_state["metrics_path"]
            log_path = worker_state["log_path"]
            duration = time.time() - worker_state["start_time"]

            record = {
                "index": idx,
                "run_tag": exp["run_tag"],
                "metrics_path": str(metrics_path),
                "log_path": str(log_path),
                "gpu_id": gpu_id,
                "returncode": returncode,
                "duration_sec": duration,
                "status": "success" if returncode == 0 else "failed",
            }

            if returncode != 0:
                failed += 1
                record["log_tail"] = read_last_lines(log_path, num_lines=30)
                print(f"[FAIL] {exp['run_tag']} (returncode={returncode})", flush=True)
            else:
                success += 1
                print(f"[OK  ] {exp['run_tag']} ({duration/60:.1f}m)", flush=True)

            done += 1
            append_jsonl(manifest_path, record)
            render_progress(done, total, success, failed, skipped, start_time)
            write_progress_json(
                progress_path,
                build_progress_payload(total, done, success, failed, skipped, start_time, active, gpu_ids),
            )
            active[gpu_id] = None

        all_workers_idle = all(active[gpu_id] is None for gpu_id in gpu_ids)
        no_more_pending = next_pending_idx >= len(pending)
        if all_workers_idle and no_more_pending:
            break

        time.sleep(max(0.5, args.poll_interval_sec))

    print("[DONE] Sweep completed")


if __name__ == "__main__":
    main()
