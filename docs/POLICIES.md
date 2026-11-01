# Bot policy ledger

Last reconciled: **2026-11-01**.

**Served default: `planner_unified_offer`** (ladder row 5), for new solo web and API games. Rollback:
`planner_unified` (row 4). Tactical is the fallback whenever the fitted checkpoint is unbound. The three-human
pilot rooms keep Tactical. **New screens compare against the served default**, on the standard tactical / mixed /
self tables with a predeclared, powered +5 pp bar (`docs/PLAN.md`).

This file is the list of served bots and what each one measured, plus the ideas that were screened and not taken.
It is not a run log: receipts live under `runs/` and are indexed in [results/LADDER.md](results/LADDER.md); the
current claims are in [CONTRACT.md](CONTRACT.md). Add a row here in the same PR that changes `SERVED_POLICY`.

## How to read a row

- **vs** is the comparator the result was measured against. That is the previous served bot only when the
  config says so; rows 2 and 4 were measured against an older bot.
- **lift** is the paired candidate-minus-control win rate: each seed is played with the candidate rotated through
  every seat, and the control in the same seats. Three tables: **tactical** (three Tactical opponents), **mixed**
  (Builder, Tactical, Expander) and **self** (three copies of the control).
- **LCB** is the predeclared lower bound (paired-normal or seed-cluster). A row's status names its own gate.
- Tactical accepts any trade that gives it at least as many cards as it gives; Builder and Expander refuse all
  trades. Lifts from trade behaviour on those tables partly measure that, so the self table is the cleaner read
  for trading changes.

## The ladder: every served change and what it measured

| row | served | policy (registry name) | change | vs | held-out lift: tactical / mixed / self [LCB] | status | PR | receipt |
|---:|---|---|---|---|---|---|---|---|
| 5 | 2026-11-01 | `planner_unified_offer` | Player trade offers: when the turn search would end the turn, propose the 1-for-1 or 2-for-1 exchange that most raises the shared score, weighted by the card-counted chance the recipient holds the card; at most one per turn, margin .25 | row 4 | **+.271 / +.115 / +.059** [+.247 / +.093 / +.037]; DEV +.270 / +.096 / +.082 | DEV PASS, HELD-OUT PASS | this PR | `planner-offer-heldout-v1` (receipt names the arm `planner_offer`, snapshot 14e7ab3) |
| 4 | 2026-11-01 | `planner_unified` | One shared score `V(me) − 0.5·max V(opp)` for robber, steal, discard, development-card play and trade answers; build search unchanged | row 2 | **+.112 / +.066 / +.023** [+.085 / +.041 / +.001]; DEV +.078 / +.088 / +.053 | DEV PASS, HELD-OUT PASS (self bound thin) | #17 | `planner-unified-heldout-v1` |
| 3 | 2026-11-01 | `planner_devfix` | Correctness fix: no random development-card play before the roll | row 2 | DEV only: −.026 / +.022 / −.008 | non-inferiority PASS (−3 pp floor) | #15 | `planner-devfix-dev-v1` |
| 2 | 2026-11-01 | `planner_parameter` | Bounded public-information planner with eight fitted economic leaf weights | Tactical | **+.110 / +.227** (tactical / mixed) [+.096 / +.208]; external panels withheld on comparator faults | family FAIL (external faults), promoted by owner on the Tactical panels | #9 | `economic-parameters-confirmatory-supervisor-v1/readout` |
| 1 | 2026-10-31 | `tactical` | Rule-based tactical bot (now the fallback) | Heuristic | **+.118** vs three heuristics [+.085] | HELD-OUT PASS | initial commit | `tactical-heldout-v1` |
| 0 | 2026-10-29 | `heuristic` | Baseline heuristic; the frozen comparator | — | — | baseline | — | — |

Attribution: rows 4 and 5 each changed one thing against their stated comparator. Row 4 was measured against
row 2, not row 3, so no direct claim is made that it beats row 3.

## Tried and not taken

Each was screened against the served bot of its day with a predeclared bar. A failed bar is a result and is not
retuned on the same boards.

| date | policy | idea | vs | DEV lift: tactical / mixed / self | outcome | receipt |
|---|---|---|---|---|---|---|
| 2026-11-01 | `planner_shared` | Award races + leader-blocking leaf + trade offers together | `planner_unified` | +.248 / +.109 / +.031 | FAIL (self below +.05); offers alone passed | `planner-shared-dev-v1` |
| 2026-11-01 | `planner_race` | Value for closing the Longest Road / Largest Army gap | `planner_unified` | −.014 / −.012 / +.021 | FAIL | `planner-race-dev-v1` |
| 2026-11-01 | `planner_block` | Shared-score build leaf (opponent-aware builds) | — | not screened alone: changed 1 of 246 decisions | inside `planner_shared` only | — |
| 2026-11-01 | `planner_discard` | Discard-aware hand value | `planner_parameter` | −.002 / −.008 / −.017 | FAIL | `planner-discard-dev-v1` |
| 2026-10-31 | `planner_offer` (older prototype, `OfferPlanner`) | Trade offers on the robber planner | `planner_robber` | +.156 / +.043 (tactical / mixed) | FAIL (mixed) | `offer-powered-dev-v1` |
| 2026-10-31 | `planner_finish` | Endgame finishing search | `planner_robber` | +.025 / +.020 | FAIL | `finish-powered-dev-v2` |
| 2026-10-31 | `planner_card_timing`, `planner_chain`, `planner_dev_sequence`, `planner_external_opening` | Early 32-seed screens (underpowered by the later power rule) | `planner_robber` / Tactical | see LADDER.md | FAIL or not advanced | `results/LADDER.md` |

## Identity rules

- A registry name means one class forever. The v2 branch briefly registered `OfferingPlanner` as `planner_offer`,
  shadowing the older `OfferPlanner`; its DEV and held-out receipts (snapshot 14e7ab3) use that name. From this
  release `planner_offer` is again the older prototype and the served arm is `planner_unified_offer`.
- The fitted bots load `checkpoints/planner_parameter.json`, bound to the source. Any `src/catan_ai` change needs
  `scripts/bind_parameter_checkpoint.py --write`, which rebinds only when the compatibility traces replay exactly.
- Changing the leaf weights, the shared-score coefficient, the offer margin or the offer cadence makes a new policy
  and needs fresh evidence.
