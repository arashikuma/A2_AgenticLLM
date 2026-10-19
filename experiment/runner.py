import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

from experiment.config import (
    ENVIRONMENTS,
    EVAL_KWARGS,
    LOG_DIR,
    MANIFEST_PATH,
    MODEL_CONFIGS,
    RESULT_DIR,
    SEEDS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_FIELDS = [
    "timestamp",
    "run_id",
    "agent",
    "model",
    "env",
    "seed",
    "status",
    "return_code",
    "progression",
    "metric_source",
    "summary_found",
    "log_path",
    "result_dir",
]

def project_path(path: Path) -> Path:
  
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path

def stop_process(process):
  
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

def append_manifest(row: dict) -> None:
  
    path = project_path(MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    old_rows = []
    old_fields = None

    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames
            old_rows = list(reader)

    if old_fields == MANIFEST_FIELDS:
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writerow(row)
    else:
       
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()

            for old_row in old_rows:
                migrated = {
                    field: old_row.get(field, "")
                    for field in MANIFEST_FIELDS
                }
                writer.writerow(migrated)

            writer.writerow(row)

def extract_progression(result_dir: Path, env: str):

    summary_files = set(result_dir.rglob("summary.json"))
    summary_files.update(result_dir.rglob("*_summary.json"))

  
    for path in sorted(summary_files):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        
        env_data = data.get("environments", {}).get(env)
        if isinstance(env_data, dict):
            score = env_data.get("progression_percentage")
            if score is not None:
                return float(score), str(path)

        
        if (
            path.name == f"{env}_summary.json"
            and data.get("progression_percentage") is not None
        ):
            return float(data["progression_percentage"]), str(path)

       
        if (
            path.name == "summary.json"
            and len(data.get("environments", {})) == 1
            and data.get("average_progress") is not None
        ):
            return float(data["average_progress"]), str(path)

    
    episode_files = [
        p for p in result_dir.rglob("*.json")
        if p.name != "summary.json"
        and not p.name.endswith("_summary.json")
    ]

   
    env_files = [
        p for p in episode_files
        if env in p.relative_to(result_dir).parts
    ]
    if env_files:
        episode_files = env_files

    scores = []
    for path in episode_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if "progression" in data:
                scores.append(float(data["progression"]))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue

    if scores:
        avg = mean(scores)

        # BALROG episode progression is between 0-1；
        if 0.0 <= avg <= 1.0:
            avg *= 100.0

        return avg, "episode_json_mean"

    return None, "not_found"


def is_already_success(agent_type, model_key, env, seed):
    manifest_file = project_path(MANIFEST_PATH)
    if not manifest_file.exists():
        return False
    with manifest_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("agent") == agent_type and 
                row.get("model") == model_key and 
                row.get("env") == env and 
                str(row.get("seed")) == str(seed) and 
                row.get("status") == "success"):
                return True
    return False

def execute_run(
    agent_type: str,
    model_key: str,
    env: str,
    seed: int,
) -> tuple[int, float | None]:

    if is_already_success(agent_type, model_key, env, seed):
        print(f"skip {agent_type}_{model_key}_{env}_seed{seed}")
        return 0, None
        

    cfg = MODEL_CONFIGS[model_key]

    safe_model = (
        model_key.replace(":", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    run_id = (
        f"{agent_type}_{safe_model}_{env}_"
        f"seed{seed}_{timestamp}"
    )

    result_root = project_path(RESULT_DIR)
    log_root = project_path(LOG_DIR)

    run_result_dir = (
        result_root
        / agent_type
        / safe_model
        / env
        / f"seed_{seed}_{timestamp}"
    ).resolve()

    run_log_dir = log_root / agent_type / safe_model / env
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_result_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_log_dir / f"{run_id}.log"

    cmd = [
        sys.executable,
        "-u",
        "eval.py",
        f"agent.type={agent_type}",
        f"envs.names={env}",
        f"agent.max_image_history={EVAL_KWARGS['max_image_history']}",
        f"agent.max_text_history={EVAL_KWARGS['max_text_history']}",
        f"eval.num_workers={EVAL_KWARGS['num_workers']}",
        f"envs.env_kwargs.seed={seed}",
        "eval.save_trajectories=true",
        f"eval.output_dir={run_result_dir.as_posix()}",
        f"eval.num_episodes.babyai={EVAL_KWARGS['babyai_episodes']}",
        f"eval.num_episodes.textworld={EVAL_KWARGS['textworld_episodes']}",
        f"client.client_name={cfg['client_name']}",
        f"client.model_id={cfg['model_id']}",
        (
            f"client.base_url={cfg['base_url']}"
            if cfg.get("base_url")
            else "client.base_url=null"
        ),
        (
            "client.generate_kwargs.temperature="
            f"{EVAL_KWARGS['temperature']}"
        ),
    ]

    print(
        f"\n[START] {run_id}\n"
        f"[RESULT] {run_result_dir}",
        flush=True,
    )

    process = None
    return_code = 1
    status = "runner_error"
    progression_score = None
    metric_source = "not_found"
    summary_found = False

    try:
        with log_path.open("w", encoding="utf-8") as f_log:
            f_log.write(f"Run ID: {run_id}\n")
            f_log.write(f"Project root: {PROJECT_ROOT}\n")
            f_log.write(f"Command: {' '.join(cmd)}\n\n")
            f_log.flush()

            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )

                if process.stdout is not None:
                    for line in process.stdout:
                
                        print(line, end="", flush=True)

                        # record
                        f_log.write(line)
                        f_log.flush()

                return_code = process.wait()
                status = "success" if return_code == 0 else "failed"

            except KeyboardInterrupt:
                status = "interrupted"
                return_code = 130
                stop_process(process)
                f_log.write("\n[INTERRUPTED] Run stopped by user.\n")
                f_log.flush()
                raise

            except Exception as exc:
                status = "runner_error"
                if process is not None and process.poll() is None:
                    stop_process(process)

                if process is not None and process.returncode is not None:
                    return_code = process.returncode
                else:
                    return_code = 1

                f_log.write(
                    f"\n[RUNNER EXCEPTION] "
                    f"{type(exc).__name__}: {exc}\n"
                )
                f_log.flush()

    except KeyboardInterrupt:
        raise

    finally:
        # record 
        summary_found = (
            run_result_dir.exists()
            and any(
                p.name == "summary.json"
                or p.name.endswith("_summary.json")
                for p in run_result_dir.rglob("*.json")
            )
        )

        # get score for return code =0
        if status == "success" and return_code == 0:
            try:
                progression_score, metric_source = extract_progression(
                    run_result_dir,
                    env,
                )
            except Exception as exc:
                metric_source = f"parse_error: {type(exc).__name__}"
                progression_score = None

        append_manifest({
            "timestamp": timestamp,
            "run_id": run_id,
            "agent": agent_type,
            "model": model_key,
            "env": env,
            "seed": seed,
            "status": status,
            "return_code": return_code,
            "progression": (
                progression_score
                if progression_score is not None
                else "N/A"
            ),
            "metric_source": metric_source,
            "summary_found": summary_found,
            "log_path": str(log_path),
            "result_dir": str(run_result_dir),
        })

    if status != "success":
        print(
            f"[RUN END] status={status}, "
            f"return_code={return_code}, log={log_path}",
            flush=True,
        )
    elif progression_score is None:
        print(
            f"[RUN END] Process exited normally, but no score was parsed. "
            f"Check results at: {run_result_dir}",
            flush=True,
        )
    else:
        print(
            f"[RUN END] {run_id} | "
            f"progression={progression_score:.2f}%",
            flush=True,
        )

    return return_code, progression_score

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BALROG")
    parser.add_argument(
        "--agent",
        type=str,
        default="naive",
        choices=["naive", "custom", "reflection"],
        help="Agent",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        help="key,like qwen2.5:7b or llama3.1:8b",
    )
    args = parser.parse_args()

    models_to_run = (
        list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    )

    print("\n" + "=" * 60)
    print(f"model={models_to_run}")
    print(f"game={ENVIRONMENTS}")
    print(f"seed={SEEDS}")


    for model_key in models_to_run:
        for env in ENVIRONMENTS:
            for seed in SEEDS:
                execute_run(
                    agent_type=args.agent,
                    model_key=model_key,
                    env=env,
                    seed=seed,
                )

    print("\n done")