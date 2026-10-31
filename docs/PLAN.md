# Plan: from a validated planner to a competitive agent

Written 2026-10-31 from the audit of this repository against the shengji
workflow and Felix Rieseberg's ambition framing (speed, altitude, depth, range).
Each lane has an owner surface, a cheap first test and a numeric bar written
before it runs. A failed bar parks the lane; it does not get retuned on the
boards it was opened on.

## Outcome goal

A frozen bot that, within 2 s per decision on one core:

1. beats the verified external comparator by at least +25 pp paired win rate
   over its four-seat rotation on reserved boards, and
2. wins at least 25% of four-player games at a table of three experienced
   humans under a predeclared, randomized candidate/control design.

Target: first quarter of 2027. The number is deliberately unreasonable; the
bars below are the honest steps toward it.

## Diagnosis the plan answers

- Thirty-plus lanes have run; three passed. All were hand-written changes to a
  shallow public-information planner with a hand-weighted leaf. The one strength
  success is the lane where a fit, not a person, chose the numbers.
- The standard DEV screen (8 or 16 boards per panel) had a zero-null minimum
  detectable effect of 7 to 22 pp against a 5 pp bar, and far less than even
  odds of clearing the bar at a true +10 pp. Many parked lanes were never
  measurable. A game costs one to two seconds, so a 5 pp screen is an hour.
- The learned baseline that regressed trained on 1,408 games with a linear
  model. The field's reference result used hundreds of millions of decisions.
- The only external judge is depth-2 Catanatron, which faults on about 1.4% of
  held-out clusters and is not a strong player.

## Lanes in order

### 0. Protect the live family, then read it (now)

- No heavy work on the host until the supervisor exits. Run the family reader
  once on sealed output. Record the result, including an external-panel fail
  on faults if that is what the reader says, as a ladder row.
- Bar: none. This is bookkeeping of a predeclared run.

### 1. Fix the instrument (this week)   DONE in code, pending first use

- `catan_ai.power` admits a screen only if, at its declared
  `alternative_effect` and `planning_sigma`, every panel clears the configured
  gate (point bar, plus the LCB margin when an LCB gate is set) with family
  probability at least `power`. Summaries record cluster SD and a zero-null
  MDE as descriptive sensitivity only. `scripts/screen_power.py` reports past
  receipts and recommends cluster counts with the same admission rule.
- Correction after Codex's review of PR #2: the first version planned against
  zero and was used by the three rescreens below. Their verdicts stand as
  declared; their `power` blocks overstated power to clear +5 pp, and the
  evaluator now refuses that legacy block.
- Predeclared rescreens of two parked but unchanged candidates under a powered
  design: `experiments/finish-powered-dev.json` and
  `experiments/opening-search-powered-dev.json` (512 clusters per panel, fresh
  seeds 1600000 to 1603511, planning sigma .40). Bar unchanged: +5 pp in both
  panels, zero truncations, 2 s step, 10 s turn.
- Cost: about 8,200 games per rescreen, under an hour at cap 4 after the
  family finishes.

### 2. Loss attribution from games already played (days)   PASS 1 DONE 2026-10-31

- Read the 12,800 completed tactical-panel games in
  `runs/economic-parameters-tactical-heldout-v1.json` and the external panel
  once sealed. Attribute candidate losses by game phase, seat, robber exposure,
  discard events, award swings and final two-point races. Hold out one panel to
  check that the ranking of causes replicates.
- Bar: a ranked list of loss causes with effect sizes whose top three agree
  across the two panels. This produces lanes; it is not itself a strength claim.
- Memory note: the reader peaked at 5.7 GB on this file. Stream it.

### 3. Engine throughput (two weeks)

- Unreasonable target: 10,000 complete bot games per minute on this host.
  Current rate is near one per second. Candidates: a Rust engine behind the
  existing action protocol, or a batched array engine. Verification is
  differential: 10,000 seeded Python games replay to identical outcomes and
  identical legal-action sets at every step.
- Bar: identical replay on 10,000 seeds, then throughput measured on a fixed
  workload. This unlocks lanes 4 and 5; it is not a strength claim.

### 4. Learned value and policy from self-play at scale (weeks)

- Train a small network on millions of self-play positions to predict final
  win probability and a policy prior over action families. Use it as the
  planner leaf and as the rollout policy of the existing root search.
- First bar, offline: held-out AUC for win probability beats the economic
  leaf's ranking on the same positions. Second bar: a powered DEV screen against
  Tactical and the external comparator at +5 pp. Third: a reserved held-out
  family.
- This is the lane that produced shengji's largest gains. Everything before it
  is preparation.

### 5. Exact finish search

- When any player reaches eight points, enumerate purchase and build sequences
  over the next two to three turns under the dice distribution with the belief
  sampler, and pick the sequence maximizing win probability.
- The powered rescreen measured the existing finish candidate at about +2 to
  +2.5 pp, with approximate intervals below +5 pp. That measured effect, not
  the terminal-margin share from loss attribution, is why this lane ranks
  below lanes 4, 6 and 7. The exact search is a new candidate with its own
  config.

### 6. Trading as negotiation

- Replace the once-per-turn one-for-one offer with bundle offers priced by own
  gain minus help to the current leader, and a per-opponent acceptance model
  fitted from observed responses. Humans beat bots at this step, so it is
  where human transfer lives.
- Bar: acceptance rate and paired lift on the mixed panel under a powered
  screen.

### 7. Leader denial

- Add all three opponents' victory threats to the evaluator: robber targeting,
  trade refusal to the leader, blocking roads. The current leaf maximizes own
  value only.
- Bar: powered DEV screen, +5 pp both panels.

### 8. Judges

- Fix the external-worker fault cause before the next family, or predeclare a
  fault budget in the plan. Add Catanatron's deeper configuration under the
  hard clock and the Settlers-RL checkpoint as a second judge after a rules
  audit. Stand up a seat-swapped red-team lane that plays the served bot head
  to head.
- Human transfer stays gated on an authorized platform route; local pilots
  continue.

## Repository hygiene done in this pass

- Ruff configuration, `.python-version`, GitHub Actions CI: lint and pytest on
  the pure Python engine, pytest on the forced Cython engine, web ESLint, build
  and unit tests, and Playwright browser tests. Solver-group tests (z3) still
  skip in CI. The remote and branch protection are the owner's call.
- Vitest scoped to unit tests; trade-form logic extracted and tested.
- `docs/results/registry.json` and `LADDER.md` generated from receipts;
  rebuild after every completed run.
- `docs/CONTRACT.md` for the current state; this file for the plan;
  `experiments/README.md` remains the chronological record.

## Not yet done

- Archiving completed runs with SHA256SUMS outside the working tree.
- Splitting the historical ladder out of `experiments/README.md`; the generated
  ladder covers receipts, not the prose decisions around them.
