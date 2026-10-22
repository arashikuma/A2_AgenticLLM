

# Assignment 2 


## Repository Structure

```text
A2/
├── README.md                  # Main documentation and experiment overview for Exercise 1
├── balrog/                    # original balrog folder
│   ├── agents/                # original agents folder
│   │   ├── custom.py          # Our modified Think-Plan-Action agents
│   │   ├── custom_task2.py    # Backup for our agents
│   │   ├── custom_balrog.py   # Backup for balrog custom.py agents
├── experiment/                # Modular evaluation and scheduling pipeline
│   ├── config.py              # Centralized configuration (SSOT for models, envs, seeds, and hyperparams)
│   ├── runner.py              # Execution engine handling subprocesses, streaming logs, and failure handling
│   └── analyze.py             # Statistical matrix generator parsing manifests to compute Mean ± Std
├── logs/                      # Log artifacts and experiment tracking
│   ├── naive/                 # Raw execution logs (.log) capturing full step-by-step model interactions
│   │   ├── qwen2.5_7b/        # Logs for Qwen-2.5-7B across BabyAI & TextWorld (Seeds: 42, 100, 2026)
│   │   └── llama3.1_8b/       # Logs for Llama-3.1-8B across BabyAI & TextWorld (Seeds: 42, 100, 2026)
│   ├── custom/                # Raw execution logs (.log) from custom.py (from BALROG) 
│   │   ├── qwen2.5_7b/        # Logs for Qwen-2.5-7B across BabyAI & TextWorld (Seeds: 42, 100, 2026)
│   │   └── llama3.1_8b/       # Logs for Llama-3.1-8B across BabyAI & TextWorld (Seeds: 42, 100, 2026)
│   ├── custom_v2/             # Raw execution logs (.log) from our modified custom.py 
│   │   ├── qwen2.5_7b/        # Logs for Qwen-2.5-7B across BabyAI & TextWorld (Seeds: 42, 100, 2026)
│   │   └── llama3.1_8b/       # Logs for Llama-3.1-8B across BabyAI & TextWorld (Seeds: 42, 100, 2026)
│   ├── cot/                   # Raw execution logs (.log) from cot.py (from BALROG)
│   │   ├── qwen2.5_7b/        # Logs for Qwen-2.5-7B across BabyAI & TextWorld (Seeds: 42, 100, 2026)
│   │   └── llama3.1_8b/       # Logs for Llama-3.1-8B across BabyAI & TextWorld (Seeds: 42, 100, 2026)
│   └── run_manifest.csv       # Unified manifest indexing all runs, return codes, and progression metrics
├── results/                   # results folder
│   ├── cot/                   # results for chain of thought
│   ├── custom/                # results for original BALROG custom
│   ├── custom_v2/             # results for our modified custom (Think-Plan-Action)
│   └── naive/                 # results for naive
│   └── naive/                 # results for self-reflection
└── naive_prompt_and_results(overall and breakdown).tar   #  naive models' prompts and overall results and subtasks results (for table 2 ＆ 3 ＆ 4)
└── reflection_results.tar     # the results for models with reflection (for table 6)

```

### Description of Key Files
* **`experiment/`**: Contains the decoupled automation pipeline designed to ensure reproducible benchmark runs across diverse models and seeds while enforcing strict variable control.
* **`logs/naive/`**: Stores complete trace logs for all runs in Exercise 1. Each `.log` file records standard outputs, including detailed observations, full prompts, raw model completions, and environment transitions for qualitative error analysis.
* **`logs/run_manifest.csv`**: A centralized structured ledger that tracks execution metadata, task statuses, and extracted progression scores across all seeds.

## 1. Hardware & Environment Setup

* **OS:** Linux (Ubuntu)
* **GPU:** 1x NVIDIA GeForce RTX 4060 (8GB VRAM)
* **Local Backend:** Ollama (served at `http://localhost:11434/v1`)
* **Models Tested:** `qwen2.5:7b` and `llama3.1:8b`

## 2. Code Structure & Pipeline Design
To ensure strict control of variables and guarantee reproducibility across multiple seeds, a custom pipeline was implemented in the `experiment/` directory:
* `experiment/config.py`: Acts as the Single Source of Truth (SSOT) for all experimental hyperparameters (Seeds: 42, 100, 2026; Environments: BabyAI, TextWorld).
* `experiment/runner.py`: The execution engine that handles subprocess creation, standard output redirection (to prevent quota limit issues), and result tracking.
* `experiment/analyze.py`: Automatically parses the `run_manifest.csv` to compute the Mean ± Std for the Performance Matrix.

## 3 Start the local model 

**Step 1: Ensure everything Stop**
```bash
pkill -u $USER -f "ollama serve"
tmux kill-session -t ollama
sleep 2
curl http://localhost:11434      # should fail with "Connection refused"
```

**Step 2: Start the server in a detached tmux session**
```bash
tmux new -d -s ollama "export PATH=/local/s3782999/ollama/bin:\$PATH; export HOME=/local/s3782999; export OLLAMA_MODELS=/local/s3782999/ollama/models; export OLLAMA_CONTEXT_LENGTH=8192; ollama serve"
```

**Step 3: Check if ollama running**
```bash
sleep 3
curl http://localhost:11434          # should print "Ollama is running"
```

**Step 4: Pull the model**
```bash
export PATH=/local/s3782999/ollama/bin:$PATH
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
```

## 4. How to Run the Exercise 1
The execution pipeline is fully automated. 

**Step 1: Ensure Local Model started**
Ensure Ollama is running in the background and the models are pulled:
```bash
ollama list
```
**Step 2: Execute the Benchmark**
```bash
uv run python -m experiment.runner --model qwen2.5:7b --agent naive
uv run python -m experiment.runner --model llama3.1:8b --agent naive
```
***Step 3: Generate the Performance Matrix**
```bash
uv run python -m experiment.analyze --agent naive
```

## 5. How to Run the Exercise 2
The execution pipeline is fully automated. 

**Step 1: Execute the Benchmark**
```bash
uv run python -m experiment.runner --model qwen2.5:7b --agent custom
uv run python -m experiment.runner --model llama3.1:8b --agent custom
```

If you want to add label 
```bash
uv run python -m experiment.runner --model qwen2.5:7b --agent custom --label custom_v2
uv run python -m experiment.runner --model llama3.1:8b --agent custom --label custom_v2
```

For running Chain of Thought
```bash
uv run python -m experiment.runner --model qwen2.5:7b --agent cot
uv run python -m experiment.runner --model llama3.1:8b --agent cot
```

***Step 2: Generate the Performance Matrix**
```bash
uv run python -m experiment.analyze --agent custom
uv run python -m experiment.analyze --agent custom_v2
uv run python -m experiment.analyze --agent cot
```

## 6. How to Run Exercise 3 
The execution workflow follows the same pipeline as Exercise 1.

**Step 1: Integrating Custom Agents (`__init__.py`)**
To ensure BALROG can recognize the newly implemented `ReflectionAgent`, it must be registered in the agent registry located at `balrog/agents/__init__.py`. Append the following lines to map the `--agent reflection` argument to your code:

Add to the begining
```bash
from experiment.reflection import ReflectionAgent
```
Add to the end
```bash
elif self.config.agent.type == "reflection":
            return ReflectionAgent(client_factory, prompt_builder)
```

**Step 2: Execute the Reflection Benchmark**
```bash
uv run python -m experiment.runner --model qwen2.5:7b --agent reflection
uv run python -m experiment.runner --model llama3.1:8b --agent reflection
```

**Step 3: Generate the Performance Matrix**
```bash
uv run python -m experiment.analyze --agent reflection
```