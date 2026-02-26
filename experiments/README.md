# Experiments

Controlled experiments to evaluate subsystem contributions.

## Structure

```
experiments/
├── runner.py              # Experiment runner + query sets + CLI
├── data/                  # Exported telemetry (JSONL/CSV)
├── results/               # Experiment result JSON files
└── README.md              # This file
```

## Available Experiments

| Experiment | What It Tests | Arms |
|---|---|---|
| `continuation_gate` | Does the topic continuation gate prevent false continuations? | with_gate vs without_gate |
| `behavior_engine` | Does behavioral adaptation improve response quality? | with_behavior vs without_behavior |
| `thread_clustering` | Does thread clustering improve topic coherence? | with_threading vs without_threading |
| `research_memory` | Does insight extraction improve cross-turn recall? | with_research vs without_research |
| `full_pipeline` | Full LMA pipeline vs minimal RAG baseline | full_pipeline vs minimal_rag |
| `baseline_rag` | Pure RAG baseline for comparison | standard_rag only |

## Running Experiments

```bash
# Start the backend first
cd backend && uvicorn main:app --port 8000

# Run an experiment
python experiments/runner.py full_pipeline
python experiments/runner.py behavior_engine --queries behavioral
python experiments/runner.py research_memory --queries multi_turn

# Custom output path
python experiments/runner.py continuation_gate --output experiments/results/my_test.json
```

## Query Sets

| Set | Messages | What It Tests |
|---|---|---|
| `multi_turn` | 8 | Topic shifts, continuations, cross-thread references |
| `repetition` | 4 | Repetition detection and handling |
| `greetings` | 6 | Greeting loop detection |
| `behavioral` | 11 | Frustration, testing, rapid-fire, exploratory patterns |
| `profile` | 4 | Profile extraction and recall |

## Telemetry API

The backend exposes telemetry endpoints:

```
GET  /telemetry          → Aggregate summary (gate rates, latencies, distributions)
GET  /telemetry/recent   → Last N raw records
POST /telemetry/export   → Export to experiments/data/ as JSONL or CSV
POST /telemetry/clear    → Reset telemetry buffer
```

## Analysis

After running experiments, analyze results with the Jupyter notebook:

```bash
jupyter notebook experiments/analysis.ipynb
```

Or load results programmatically:

```python
from experiments.runner import ExperimentRunner
results = ExperimentRunner.load_results("experiments/results/full_pipeline_1234.json")
```
