# Catan AI: current contract

What is served, what has been shown, and what has not. This page is the
current state; the chronological research record is `experiments/README.md`
and top-level receipts plus run-directory summaries and readouts are indexed in
[results/LADDER.md](results/LADDER.md).

## Served default

- **Policy:** `planner_parameter`, the fitted economic planner, for new solo web
  and API games since 2026-11-01 (owner's decision). Tactical is the fallback,
  and the three-human pilot rooms keep Tactical as their default. `heuristic`
  remains the frozen comparator.
- **Why:** both Tactical-control panels of the sealed held-out family passed
  (+.110 and +.227 paired lift; simultaneous lower bounds +.096 and +.208). The
  external-control panels have twice been withheld on comparator faults. The
  v13 rerun's owner-requested descriptive read is +.061 and +.151 against
  Catanatron, and dropping or penalising the three faulted boards changes
  neither figure by more than .002. The promotion therefore rests on the
  Tactical-control result, not on a passed external test.
- **Checkpoint:** `checkpoints/planner_parameter.json` carries the evaluated
  weights (b1785ccb…, unchanged), bound to the current source on the
  pure-Python engine. Any source change unbinds it. `scripts/bind_parameter_checkpoint.py`
  rebinds only if the 8 compatibility games replay the evaluated snapshot's
  traces exactly. Unbound, `/api/bots` reports it and new games use Tactical.
- **Previous default:** Tactical, which beat three heuristics by +11.77 pp on
  155 reserved boards with a one-sided 95% seed-cluster lower bound of +8.47 pp
  (`runs/tactical-heldout-v1.json`).
- **Budget:** 2 s per decision and 10 s of compute per turn, including
  observation, belief, adapter, search and validation. A timeout plays a legal
  deterministic fallback and counts as a budget failure. The fitted planner's
  held-out step p95 was about 10 ms.

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
- **Power.** A screen declares `power.alternative_effect` (a true lift above the
  bar), `power.planning_sigma` and optional `power.power` (default .8). The
  evaluator refuses a design unless every panel clears the configured gate at
  that alternative with at least that family probability; the gate is the
  point bar plus the LCB margin when an LCB gate is set
  (`catan_ai.power.check_design`). A legacy `target_effect` block planned
  against zero and is refused. Summaries record the panel's cluster SD and a
  zero-null MDE, which is descriptive sensitivity, not power to clear the bar.
  Historical 8- and 16-cluster screens had zero-null MDEs of 7 to 22 pp against
  a 5 pp bar; their failures stand, but an unchanged candidate may be
  rescreened under a new, powered, predeclared config.
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
