# Ayojna: AI that plans where every byte should live

**Hackfest 2026 · Problem Statement 2A · Team CodeBlooded** (Bhoomi B, Vaishnavi K, Joel B, Aditya P)

Ayojna watches block-storage traffic, predicts which data will be hot, warm or cold in the next
24 hours, and moves every 256 MB extent to the cheapest storage tier that still meets its latency
SLA and compliance rules. It moves data safely and survives its own failures. A copilot explains
every decision in plain English.

## How it works

```
MSR traces ──> extent_hourly ──> features ──> hotness model (ML, rule fallback)
                                                  │
              config (prices, SLA, compliance) ──>│
                                                  v
     policy guard (legal hold, residency, SLA floors) ──> cost optimizer ──> move plan
                                                                               │
  central supervisor (leader lease + fencing, checkpoints, levels L0-L3)       v
                                                   executor saga: copy -> verify -> switch
                                                                               │
                         dashboard + API + grounded copilot <── audit, catalog, scoreboard
```

| Module | What it does |
|---|---|
| `ayojna/ingest` | MSR Cambridge traces (synthetic or real, streamed from the SNIA .tar) to 256 MB extent x hour table |
| `ayojna/models` | features, hot/warm/cold labels, sklearn gradient boosting with reasons, rule baseline, promotion gate |
| `ayojna/twin` | digital twin: replays the trace hour by hour and prices storage, retrieval, moves, latency, SLA |
| `ayojna/policy` + `ayojna/planner` | compliance first, then cheapest allowed tier with hysteresis, capacity and a migration budget |
| `ayojna/supervisor` | leader lease with fencing token, timeouts, retries, fallbacks, checkpoints, crash resume |
| `ayojna/executor` | saga per move: fence check, idempotency key, copy, checksum verify, catalog switch, rollback |
| `ayojna/api` + `web/` | FastAPI and a single-page control room |
| `ayojna/copilot` | Q&A grounded in live facts; LLM optional (Gemini / Groq / Ollama), template fallback |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m pytest -q                # all tests should pass
python -m ayojna.demo --serve      # whole pipeline on a synthetic trace, then the dashboard
```

Open http://localhost:8000 (dashboard) and http://localhost:8000/docs (API).

### Real MSR Cambridge traces
1. Download "MSR Cambridge Traces" from SNIA IOTTA: https://iotta.snia.org/traces/block-io/388
2. Put the downloaded `.tar` file(s) in `data/raw/msr/`. You don't need to unzip them.
3. `python -m ayojna.ingest.real --list` lists the volumes found.
4. `python -m ayojna.demo --source real --serve`

The traces are under the SNIA trace license and are never committed (`data/` is git-ignored).

## Results

Scored on unseen hours 147-194; predictions: primary / ok (model 20261008-163726)

| Strategy | $/month | Saving vs all-hot | SLA met | Compliance | GB moved |
|---|---:|---:|---:|---:|---:|
| all_hot | 21.075 | 0.0% | 100.00% | 100.0% | 0.00 |
| age_rule | 9.534 | 54.8% | 100.00% | 96.2% | 68.25 |
| access_timer | 13.886 | 34.1% | 100.00% | 96.2% | 62.50 |
| lru | 8.921 | 57.7% | 100.00% | 96.2% | 129.00 |
| ayojna | 7.143 | 66.1% | 83.66% | 100.0% | 30.75 |


## Reliability demos

| Command | What you see |
|---|---|
| `python -m ayojna.supervisor.run --fail hotness` | L3: a critical step failed, nothing moves |
| `python -m ayojna.supervisor.run --recommend-only` | L2: the plan is shown, nothing moves |
| `python -m ayojna.supervisor.run --corrupt-move` | a bad copy is detected by checksum and rolled back |
| primary with `--crash-after-moves 40`, then a replica | the replica takes token 2, resumes, skips 40 finished moves |

## Copilot (optional LLM)
Copy `.env.example` to `.env` and add a **free** key: `GEMINI_API_KEY` (Google AI Studio) and/or
`GROQ_API_KEY`. Without one, answers come from templates. With several, they are tried in order.
Every LLM answer passes a grounding check: any number not found in the live facts is rejected.

## Datasets
- MSR Cambridge block I/O traces (SNIA IOTTA): https://iotta.snia.org/traces/block-io/388
- D. Narayanan, A. Donnelly, A. Rowstron, "Write Off-Loading: Practical Power Management for
  Enterprise Storage", USENIX FAST 2008.