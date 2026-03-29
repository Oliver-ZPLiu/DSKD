import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TASK_ALIASES = {
    "PiQA": "piqa",
    "ARC-C": "arc_challenge",
    "BoolQ": "boolq",
    "MMLU": "mmlu",
    "AGI-EN": "agieval_en",
    "AGI-ZH": "agieval_cn",
    "IFEval": "ifeval",
}

MC_TASKS = [
    TASK_ALIASES["PiQA"],
    TASK_ALIASES["ARC-C"],
    TASK_ALIASES["BoolQ"],
    TASK_ALIASES["MMLU"],
    TASK_ALIASES["AGI-EN"],
    TASK_ALIASES["AGI-ZH"],
]

IFEVAL_TASK = TASK_ALIASES["IFEval"]

METRIC_PREFERENCE = {
    "piqa": ["acc_norm", "acc"],
    "arc_challenge": ["acc_norm", "acc"],
    "boolq": ["acc"],
    "mmlu": ["acc"],
    "agieval_en": ["acc"],
    "agieval_cn": ["acc"],
    "ifeval": ["prompt_level_strict_acc"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lm-evaluation-harness on PiQA/ARC-C/BoolQ/MMLU/AGI-EN/AGI-ZH/IFEval."
    )
    parser.add_argument("--model-path", required=True, help="HF model path or local checkpoint.")
    parser.add_argument("--peft-path", default=None, help="Optional LoRA/PEFT adapter path.")
    parser.add_argument("--output-dir", default="outputs/lm_eval_harness", help="Directory for raw and summary results.")
    parser.add_argument("--device", default="cuda:0", help="Evaluation device, e.g. cuda:0 or cpu.")
    parser.add_argument("--dtype", default="bfloat16", help="dtype passed to lm-eval model_args.")
    parser.add_argument("--batch-size-mc", default="auto", help="Batch size for multiple-choice tasks.")
    parser.add_argument("--batch-size-ifeval", default="1", help="Batch size for IFEval (generation task).")
    parser.add_argument(
        "--extra-model-args",
        default="",
        help="Extra lm-eval model_args string, e.g. attn_implementation=flash_attention_2.",
    )
    parser.add_argument("--num-fewshot", type=int, default=None, help="Override num_fewshot for all tasks.")
    parser.add_argument("--limit", type=float, default=None, help="Optional lm-eval --limit for quick debugging.")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code in model_args.")
    parser.add_argument(
        "--launcher",
        choices=["python", "accelerate"],
        default="python",
        help="Command launcher. Use 'accelerate' for multi-GPU data parallel.",
    )
    parser.add_argument("--num-processes", type=int, default=1, help="Number of processes for accelerate launcher.")
    parser.add_argument("--main-process-port", type=int, default=29500, help="main_process_port for accelerate.")
    return parser.parse_args()


def build_model_args(args: argparse.Namespace) -> str:
    trust_remote_code = "False" if args.no_trust_remote_code else "True"
    parts = [
        f"pretrained={args.model_path}",
        f"dtype={args.dtype}",
        f"trust_remote_code={trust_remote_code}",
    ]
    if args.peft_path:
        parts.append(f"peft={args.peft_path}")
    if args.extra_model_args.strip():
        parts.append(args.extra_model_args.strip())
    return ",".join(parts)


def run_lm_eval(
    tasks: List[str],
    batch_size: str,
    output_path: Path,
    model_args: str,
    device: str,
    num_fewshot: Optional[int],
    limit: Optional[float],
    launcher: str,
    num_processes: int,
    main_process_port: int,
) -> None:
    if launcher == "accelerate":
        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            str(max(1, num_processes)),
            "--main_process_port",
            str(main_process_port),
            "-m",
            "lm_eval",
        ]
    else:
        cmd = [sys.executable, "-m", "lm_eval"]

    cmd.extend(
        [
            "--model",
            "hf",
            "--model_args",
            model_args,
            "--tasks",
            ",".join(tasks),
            "--device",
            device,
            "--batch_size",
            str(batch_size),
            "--output_path",
            str(output_path),
        ]
    )
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_latest_json(path: Path) -> Path:
    json_files = [p for p in path.rglob("*.json") if p.is_file()]
    if not json_files:
        raise FileNotFoundError(f"No JSON results found under: {path}")
    return max(json_files, key=lambda p: p.stat().st_mtime)


def load_results(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_metric(task_result: Dict, preferred_metrics: Iterable[str]) -> Tuple[Optional[float], Optional[str]]:
    for metric_name in preferred_metrics:
        for key, val in task_result.items():
            if key.endswith("_stderr"):
                continue
            key_base = key.split(",")[0]
            if key_base == metric_name and isinstance(val, (int, float)):
                return float(val), key
    for key, val in task_result.items():
        if key.endswith("_stderr"):
            continue
        if isinstance(val, (int, float)):
            return float(val), key
    return None, None


def build_summary(merged_results: Dict[str, Dict]) -> Dict:
    scores = {}
    metric_keys = {}

    for alias, task_name in TASK_ALIASES.items():
        if task_name not in merged_results:
            scores[alias] = None
            metric_keys[alias] = None
            continue
        score, key = pick_metric(merged_results[task_name], METRIC_PREFERENCE[task_name])
        scores[alias] = score
        metric_keys[alias] = key

    valid_scores = [x for x in scores.values() if isinstance(x, (int, float))]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    return {
        "scores": scores,
        "metric_keys": metric_keys,
        "avg": avg_score,
    }


def main() -> None:
    args = parse_args()
    model_args = build_model_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mc_out = output_dir / "run_mc"
    ifeval_out = output_dir / "run_ifeval"

    run_lm_eval(
        tasks=MC_TASKS,
        batch_size=str(args.batch_size_mc),
        output_path=mc_out,
        model_args=model_args,
        device=args.device,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        launcher=args.launcher,
        num_processes=args.num_processes,
        main_process_port=args.main_process_port,
    )
    run_lm_eval(
        tasks=[IFEVAL_TASK],
        batch_size=str(args.batch_size_ifeval),
        output_path=ifeval_out,
        model_args=model_args,
        device=args.device,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        launcher=args.launcher,
        num_processes=args.num_processes,
        main_process_port=args.main_process_port,
    )

    mc_json = find_latest_json(mc_out)
    ifeval_json = find_latest_json(ifeval_out)
    mc_data = load_results(mc_json)
    ifeval_data = load_results(ifeval_json)

    merged_results = {}
    merged_results.update(mc_data.get("results", {}))
    merged_results.update(ifeval_data.get("results", {}))

    summary = build_summary(merged_results)
    payload = {
        "task_aliases": TASK_ALIASES,
        "raw_result_files": {
            "mc": str(mc_json),
            "ifeval": str(ifeval_json),
        },
        "summary": summary,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n===== Summary =====")
    for alias in TASK_ALIASES:
        score = summary["scores"].get(alias)
        key = summary["metric_keys"].get(alias)
        if score is None:
            print(f"{alias:7s}: N/A")
        else:
            print(f"{alias:7s}: {score:.4f} ({key})")
    if summary["avg"] is None:
        print("Avg.   : N/A")
    else:
        print(f"Avg.   : {summary['avg']:.4f}")
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
