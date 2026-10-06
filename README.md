# Ayojna

**AI that plans where every byte should live.** Predictive, policy-aware storage tiering with a fault-tolerant supervisor.

Hackfest 2026 · Problem Statement 2A: Smartly Optimize Storage for Modern Workloads · Team CodeBlooded

## What it does
Ayojna watches storage telemetry, predicts which data will be hot, warm or cold, checks compliance rules, and moves each dataset to the cheapest tier that still meets its latency target. A central supervisor (with a hot-standby replica) runs every step with timeouts, retries and fallbacks, so one failing module never stops the system.

## Repo map
| Folder | What lives there | Owner |
|---|---|---|
| `ayojna/contracts/` | Shared data shapes every module uses | Both |
| `config/` | Tiers, prices, SLA classes, compliance tags, label rules | Both |
| `ayojna/ingest/` | Trace replay, MSR parser, extent-hourly table | Bhoomi |
| `ayojna/twin/` | Digital twin: tier cost + latency model, baselines | Bhoomi |
| `ayojna/supervisor/` | Central layer: primary, replica, Redis lease, checkpoints | Bhoomi |
| `ayojna/executor/` | Safe moves on MinIO tiers, rollback | Bhoomi |
| `ayojna/api/` | FastAPI backend | Bhoomi |
| `ayojna/models/` | Features, hotness, forecast, anomaly + fallbacks | Vaishnavi |
| `ayojna/policy/` | OPA compliance rules | Vaishnavi |
| `ayojna/planner/` | OR-Tools optimizer + RL bandit | Vaishnavi |
| `web/` | React dashboard | Vaishnavi |
| `ayojna/copilot/` | LLM copilot (stretch) | Vaishnavi |

## Setup (once per laptop)
```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Mac / Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Run the pipeline on synthetic data
```bash
python -m ayojna.ingest.synth --days 5 --out data/raw/synth
python -m ayojna.ingest.build --raw data/raw/synth --out data/lake/extent_hourly.parquet
python -m ayojna.models.build_features --inp data/lake/extent_hourly.parquet --out data/lake/features.parquet
```

## Datasets
- [MSR Cambridge traces (SNIA IOTTA)](https://iotta.snia.org/traces/block-io/388): train and test
- [Alibaba block traces](https://github.com/alibaba/block-traces): generalization check
- Put downloaded CSVs in `data/raw/msr/`. Data folders are gitignored.

## How we work
- `main` is protected. Work on a branch, open a pull request, the other person reviews.
- Branch names: `feat/<thing>`, `fix/<thing>`, `chore/<thing>`.
- Commit messages: `feat(ingest): ...`, `fix(models): ...`, `test(...)`, `docs(...)`, `chore(...)`.
- Never change `ayojna/contracts/` without telling your teammate.
