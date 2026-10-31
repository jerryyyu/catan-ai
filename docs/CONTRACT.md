# Catan AI: current contract

What is served, what has been shown, and what has not. This page is the
current state; the chronological research record is `experiments/README.md`
and every receipt is indexed in [results/LADDER.md](results/LADDER.md).

## Served default

- **Policy:** `tactical` for new web and API games. `heuristic` remains the
  frozen comparator and is unchanged since its promotion screen.
- **Why:** on 155 reserved boards, all seats, Tactical beat three heuristics by
  +11.77 pp paired with a preregistered one-sided 95% seed-cluster lower bound
  of +8.47 pp, all 1,240 games complete, decision p95 2.36 ms
  (`runs/tactical-heldout-v1.json`).
- **Budget:** 2 s per decision and 10 s of compute per turn, including
  observation, belief, adapter, search and validation. A timeout plays a legal
  deterministic fallback and counts as a budget failure.

## Challenger: held-out family sealed, FAIL on comparator faults

`planner_parameter`: the bounded public-information planner with eight economic
leaf weights fitted on complete-game outcomes
(`runs/economic-parameters-v1/model.json`). Its frozen held-out family of
51,200 games (`experiments/economic-parameters-confirmatory-family-v1.json`)
sealed on 2026-10-31 with verdict FAIL: both Tactical-control panels passed
(+.110 and +.227 lift, simultaneous lower bounds +.096 and +.208 at alpha
.0125), both external-control panels were withheld on 28 external-worker faults
in 12,800 games against a zero-fault rule. The candidate is therefore shown
stronger than the served default on this engine, but the family criterion was
not met and the served default is unchanged until the owner decides otherwise.

## What is not claimed

- No competitive strength against ranked online humans. No Colonist
  integration exists or is authorized; its terms restrict automation and data
  use.
- Beating the depth-2 Catanatron comparator is not beating a strong player.
  The comparator faults on about 1.4% of held-out clusters.
- DEV passes are screens. Only a reserved held-out run with a predeclared
  analysis supports a promotion.
- Correctness, latency, calibration and strength are separate claims. A
  passing test suite or a fast benchmark says nothing about play strength.

## Evidence rules

- Every run is seeded and writes an immutable receipt: config hash, source file
  hashes, engine backend and binary hashes, host, process cap, per-game
  outcomes, truncations and an interval. Incomplete runs carry no estimate.
- Candidates rotate through all four seats on the same seed block; the control
  plays the same boards.
- A new idea gets a config with a numeric bar before it runs. A failed bar is a
  result. Changing the producer or the config means a new run and a new
  receipt; old receipts are preserved.
- **Power.** A screen declares `power.target_effect` and `power.planning_sigma`
  and the evaluator refuses a design whose seed count cannot detect that
  effect at 80% power (`catan_ai.power.check_design`). Every summary records the
  panel's cluster SD and minimum detectable effect. Historical 8- and 16-cluster
  screens had minimum detectable effects of 7 to 22 pp against a 5 pp bar;
  their failures stand, but an unchanged candidate may be rescreened under a
  new, powered, predeclared config.
- Hidden information never crosses the player boundary. Bots see their own hand,
  public counts and the observed event stream only. Adding private state
  requires a boundary test.

## Where things live

| What | Where |
|---|---|
| Rules engine | `src/catan_ai/_engine_py.py`, selected through `engine.py` |
| Policies | `src/catan_ai/bots.py` factory; planners in `planner.py`, `tactical.py`, `parameter_planner.py` |
| Paired screens | `src/catan_ai/search_eval.py`, configs in `experiments/` |
| Power tooling | `src/catan_ai/power.py`, `scripts/screen_power.py` |
| Receipt index | `docs/results/registry.json`, rebuilt by `scripts/build_results_registry.py` |
| Research record | `experiments/README.md` |
| Plan | `docs/PLAN.md` |
