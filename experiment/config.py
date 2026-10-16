from pathlib import Path


SEEDS = [42, 100, 2026]
ENVIRONMENTS = ["babyai", "textworld"]

# dir
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
RESULT_DIR = BASE_DIR / "results"
MANIFEST_PATH = LOG_DIR / "run_manifest.csv"

# setup
MODEL_CONFIGS = {
    "qwen2.5:7b": {
        "client_name": "vllm",
        "model_id": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
    },
    "llama3.1:8b": {
        "client_name": "vllm",
        "model_id": "llama3.1:8b",
        "base_url": "http://localhost:11434/v1",
    },
}

# parameters
EVAL_KWARGS = {
    "max_image_history": 0,
    "max_text_history": 16,
    "num_workers": 1,
    "babyai_episodes": 10,
    "textworld_episodes": 10,
    "temperature": 0.0,
}
