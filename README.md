# Catan AI

Unofficial research project; not affiliated with Catan GmbH or its publishers. Original visual assets only.

A reproducible research stack for four-player base-game Catan: one authoritative Python rules engine,
bots behind one interface, seeded simulation, paired seat-rotated experiments with JSON receipts, and a
React table for solo play and private rooms. Target: Colonist ranked four-player base game.

| Doc | What it holds |
|---|---|
| [docs/CONTRACT.md](docs/CONTRACT.md) | Current state and claim limits |
| [docs/PLAN.md](docs/PLAN.md) | Plan |
| [docs/POLICIES.md](docs/POLICIES.md) | Served bots and what each measured |
| [docs/AI_POLICIES.md](docs/AI_POLICIES.md) | Every policy family and external AI, with benchmarks |
| [docs/atlas/](docs/atlas/) | Policy-by-comparison table (`python docs/atlas/build.py`) |
| [docs/results/LADDER.md](docs/results/LADDER.md) | Receipt index |
| [experiments/README.md](experiments/README.md) | Every recorded comparison and research notes |
| [docs/ENGINE.md](docs/ENGINE.md) | Rules coverage, privacy, search methods, rules validation |
| [docs/PLAYING.md](docs/PLAYING.md) | Pilots, replays, private rooms, audits |

Served solo bot: `planner_pv2_tiles` (owner promotion 2026-11-08; fallback `planner_offer_accept`). Its weights are a
local file, never committed:

```bash
cp /path/to/value_tiles_v1.npz checkpoints/   # sha256 must equal npz_sha256 in checkpoints/value_tiles_v1.meta.json
uv run python -c "from catan_ai.api import bots_info; print(bots_info())"   # expect 'default': 'planner_pv2_tiles'
```

Missing or unverified weights fall back to `planner_offer_accept` with the reason in `GET /api/bots`
([docs/PLAYING.md](docs/PLAYING.md)).

## Run

```bash
uv sync --dev && uv run catan-server        # API on :8000
cd web && npm install && npm run dev       # http://localhost:5173 (proxies /api)
```

A production build in `web/dist` is served by FastAPI.

## Simulate and measure

```bash
uv run catan-sim --games 100 --seed 42
uv run catan-experiment --config experiments/base-actions-dev.json
uv run python scripts/parallel_search_eval.py --config experiments/<screen>.json --output runs/<name>.json --process-cap 4
```

Every run is seeded and writes an immutable receipt under `runs/` (local, git-ignored): config, seeds,
per-game outcomes, truncations, intervals, elapsed time and source hashes. A receipt is evidence about that
comparison, not a deployment decision.

External benchmarks: Catanatron (`scripts/external_ladder*`), JSettlers2 ([tools/jsettlers](tools/jsettlers)),
StacSettlers with optional chat trading ([tools/stac/README.md](tools/stac/README.md)).

## Test

```bash
uv run pytest -q                       # Python engine
uv run --group native python scripts/build_native.py build_ext --inplace
CATAN_ENGINE=cython uv run pytest -q   # forced Cython engine
cd web && npm run build && npm test && npm run test:e2e
```

The served checkpoint pins every source file. After changing `src/catan_ai`, rebind it (it refuses if the
8 compatibility games change):

```bash
uv run python scripts/bind_parameter_checkpoint.py --write
```

## Layout

```
src/catan_ai/   engine, bots, search, observation, API
scripts/        experiments, data, training, external bridges
experiments/    predeclared configs
tools/          JSettlers and STAC bridges
web/            React + TypeScript table
docs/           contract, plan, policies, atlas, results
```

Player-facing views never include opponents' hidden cards, the seed or the deck.
