import csv
from collections import defaultdict
from experiment.config import ENVIRONMENTS, MANIFEST_PATH, MODEL_CONFIGS


def generate_matrix(agent_filter: str = "naive"):
    if not MANIFEST_PATH.exists():
        print(f"{MANIFEST_PATH} not found , no data")
        return

    #  (model, env) -> [scores]
    matrix_data = defaultdict(list)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["agent"] != agent_filter or row["status"] != "success":
                continue
            try:
                score = float(row["progression"])
                matrix_data[(row["model"], row["env"])].append(score)
            except (ValueError, TypeError):
                continue

    # Markdown
    print("\n" + "=" * 75)
    print(f"### Performance Matrix (Progression %: Mean ± Std) - Agent: {agent_filter.upper()}")
    print("=" * 75)
    header = f"| {'LLM Model':<18} | " + " | ".join([f"{env.capitalize():<18}" for env in ENVIRONMENTS]) + " |"
    divider = f"|:{'-'*18}-|-" + "-|-".join(['-'*18 for _ in ENVIRONMENTS]) + "-:|"
    print(header)
    print(divider)

    for model_key in MODEL_CONFIGS.keys():
        row_str = f"| {model_key:<18} | "
        for env in ENVIRONMENTS:
            scores = matrix_data.get((model_key, env), [])
            if scores:
                mean = sum(scores) / len(scores)
                variance = sum((x - mean) ** 2 for x in scores) / len(scores)
                std = variance ** 0.5
                row_str += f"{mean:5.2f}% ± {std:5.2f}%   | "
            else:
                row_str += f"{'N/A':<18} | "
        print(row_str)
    print("=" * 75 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="naive")
    args = parser.parse_args()
    generate_matrix(args.agent)
