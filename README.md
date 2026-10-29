# Catan AI

Unofficial research project; not affiliated with Catan GmbH or its publishers.
The UI uses original shapes and styling rather than commercial game assets.

A small, reproducible research stack for a Catan-like four-player game:

- one authoritative, UI-free Python rules engine;
- random and heuristic policies behind one bot interface;
- seeded headless simulation;
- paired, seat-rotated experiments with JSON receipts and confidence intervals;
- a React web table for one human against three bots.

The first milestone implements random standard-board generation, snake-order setup,
dice/resource production, the robber, roads, settlements, cities, bank costs, and a
10-point terminal condition. Development cards, ports, player trading, Longest Road,
and Largest Army are explicit next milestones rather than hidden approximations.

## Run it

```bash
# terminal 1
uv sync --dev
uv run catan-server

# terminal 2
cd web
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api` to FastAPI on port
8000. A production web build is served by FastAPI when `web/dist` exists.

## Simulate and measure

```bash
uv run catan-sim --games 100 --seed 42
uv run catan-experiment --config experiments/heuristic-v-random.json
```

Experiments write an immutable JSON result under `runs/` containing the config,
seed range, per-seat results, aggregate win rate, Wilson 95% interval, duration,
and the source revision when available. A result receipt is evidence about that
exact comparison, not a deployment decision.

## Layout

```
src/catan_ai/engine.py       board, rules, legal actions, state transitions
src/catan_ai/bots.py         policy interface and baseline bots
src/catan_ai/sim.py          deterministic bot-v-bot game runner
src/catan_ai/experiments.py  paired seat-rotation experiment runner
src/catan_ai/api.py          per-player HTTP state and browser game endpoint
tests/                       engine, determinism, API, and experiment tests
web/                         React + TypeScript table
```

The server never sends opponents' resource cards. Seeds and complete state remain
available to local simulations but are absent from player-facing API views.
