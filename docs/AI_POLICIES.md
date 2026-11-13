# AI policies and benchmarks

Last reconciled: **2026-11-11**.

**Served default: `planner_pv2_tiles_va_t9`** (promoted by Jerry 2026-11-11; ladder row 9 in [POLICIES.md](POLICIES.md)).
Its 16-game trace hash is `6eae37ea1ed0312d164fc3c5df72a7c511dd319506c437426bda69cfe5b19bdf`. Fallbacks:
`planner_pv2_tiles` (row 8, trace `c7d07a76…`), then `planner_offer_accept` (row 7), then Tactical. Every production
release is logged in [Production releases](#production-releases) below.

## Production releases

One row per change to the served solo default, newest first. Add the row in the same PR that changes
`SERVED_POLICY` (`src/catan_ai/api.py`), with the evidence the promotion rests on and its receipts; a later
rollback or fallback event gets its own row. Detail per release: the ladder in [POLICIES.md](POLICIES.md).

| # | served from | policy | what changed | evidence the release rests on | PR | trace |
|---|---|---|---|---|---|---|
| 9 | 2026-11-11 | `planner_pv2_tiles_va_t9` | value transformer retrained on original + on-policy self-play (T9) + buyfix purchase scoring | held-out vs row 8: self table **+.035** (seed LCB +.024, 896 seeds), paired seeded JSettlers **+.026** (LCB +.012, 5,120 games); DEV +.046 / +.027 | #67–#69, #73 (evaluation), this PR (serving) | `6eae37ea…` |
| 8 | 2026-11-09 | `planner_pv2_tiles` | PV search scored by the tile-token value transformer | held-out vs row 7: internal +.046 / +.040 / +.096; JSettlers re-test +.061 (LCB +.032) | #34, #45 | `c7d07a76…` |
| 7 | 2026-11-03 | `planner_offer_accept` | trade offers weighted by the recipient's public acceptance rate | held-out mixed-table PASS vs row 6 | #20, #23 | `9fd2805e…` |
| 6 | 2026-11-02 | `planner_offer_guard` | no offers to 8+ point seats; offers net of the recipient's gain | DEV point floor only (no own held-out) | #19 | — |
| 5 | 2026-11-01 | `planner_unified_offer` | one player trade offer per turn | held-out vs row 4 | #18 | — |
| 4 | 2026-11-01 | `planner_unified` | one shared position score for robber, steal, discard, dev-card and trade answers | held-out vs row 2 | #17 | — |
| 3 | 2026-11-01 | `planner_devfix` | development-card fixes | see POLICIES.md row 3 | #15 | — |
| 2 | 2026-11-01 | `planner_parameter` | bounded public-information planner with fitted economic weights | +.110 / +.227 vs Tactical | #9 | — |
| 1 | 2026-10-31 | `tactical` | rule-based baseline | — | initial | — |
| 0 | 2026-10-29 | `heuristic` | first baseline | — | — | — |

This file is the map of every AI this project plays with or against: what each one is, how it decides, and what
it measured. It complements, and does not replace:

- [POLICIES.md](POLICIES.md): the served-bot ladder and the ledger of ideas screened and not taken (add rows there).
- [CONTRACT.md](CONTRACT.md): current claims, external benchmarks and what is not claimed.
- [PLAN.md](PLAN.md): the standard comparison, bars and lanes.
- `runs/` receipts, indexed by [results/LADDER.md](results/LADDER.md).

Every number below names its receipt. Numbers are point estimates with 95% intervals unless stated. Comparisons
across different seeds or opponents are **descriptive**, not paired.

## How we measure

- **Internal tables (paired).** Each seed is played four times with the candidate rotated through every seat, and
  again with the control in the same seats and the same dice. Lift is candidate wins minus control wins per game
  ("extra wins per 100 games" is the same number ×100). Tables used through 2026-11-09: **tactical** (three
  Tactical), **mixed** (Builder, Tactical, Expander), **self** (three copies of `planner_offer_accept`, the bot
  served until 2026-11-08, not of the current control). From 2026-11-10 the heuristic tables are dropped; see
  "Evaluation policy" in [PLAN.md](PLAN.md).
- **External ladders.** One bot alone against three copies of an outside AI, rotated through all four seats; fair
  share is .25. **Retirement rule** (predeclared 2026-11-07): an outside AI is retired as a benchmark for a given
  bot once that bot's win share is ≥ .50 with the 95% lower bound > .40.
- **Gates are predeclared** with a power block (`check_design`) before any game; a failed bar is a result.
- **Metric note.** On past receipts, final-VP margin over the best opponent is about as sensitive as wins (t
  4.1 / 5.2 / 10.4 vs wins 4.1 / 4.8 / 11.1 on `pv2-tiles-precise-v1`); finishing rank and raw VP are weaker. Wins
  per paired round remain the decision metric; margin may be reported as a secondary.

## Our policies

Registry names are what `catan_ai.bots.make_policy` accepts. "Branch only" means the code lives on a branch, not
on `main`.

### Baselines and rule-based bots

| name | how it decides | role |
|---|---|---|
| `random` | Uniform over legal actions | Sanity baseline |
| `heuristic` | Original hand-written rules | Ladder row 0 |
| `builder`, `expander` | Fixed styles: Builder favours cities and cards, Expander favours roads and settlements; both refuse all trades | Mixed-table opponents |
| `tactical` | Plans this turn's builds after at most one bank trade; accepts any trade giving it at least as many cards | Ladder row 1; tactical-table opponent |
| `search_fast`, `search_sequence`, `search_alloc`, `mc_tactical`, `mc_mean` | Early Monte Carlo search variants | Historical |

### Fitted planners (the served lineage)

All share an economic value function with fitted weights (`checkpoints/planner_parameter.json`, bound to the exact
source hashes; a source change needs `scripts/bind_parameter_checkpoint.py --write`).

| name | adds | measured (see POLICIES.md for full rows) |
|---|---|---|
| `planner_parameter` | Fitted economic weights over the turn search | Row 2: +.110 / +.227 vs Tactical-era control (tactical / mixed) |
| `planner_unified` | Unified turn planner (builds, cards, robber, discards) | Row 4: +.112 / +.066 / +.023 held-out |
| `planner_unified_offer` | One priced trade offer per turn to the trailing player | Row 5: +.271 / +.115 / +.059 held-out |
| `planner_offer_guard` | Guards offers to seats near winning | Row 6 |
| **`planner_offer_accept`** | Weights offers by each recipient's public acceptance rate | **Row 7, served** (mixed +.068 held-out) |

### PV search (value network over the served bot's candidates)

The served search proposes "end the turn now" plus its top-4 build/trade sequences; a value network scores the
position each leads to (encoder v2: 1,324 public features from the chooser's view, opponents' hands only from
card counting; purchases averaged over the deck prior). Offers stay the served bot's; end-now runs the served
end-of-turn path. Everything else (setup, robber, discards, trade answers) is served.
PV bots load gitignored local weights (`checkpoints/*.npz`, pinned by sha in the committed `*.meta.json`); without
the file, `make_policy` raises `FileNotFoundError` by design.

| name | value network | vs served (tactical / mixed / self) | receipt | status |
|---|---|---|---|---|
| `planner_pv2_b` | Residual MLP, 100k self-play games, 12-symmetry augmentation (test AUC .772) | −.076 / −.025 / +.008 (offer-suppression bug) | `pv2-b-screen-v1` | FAIL |
| `planner_pv2_b_fix` | Same net, offers restored | +.025 / +.015 / +.019 at 512 seeds | `pv2-b-fix-precise-v1` | research baseline |
| `planner_pv2_s1`, `planner_pv2_s1_m1` | Commit-to-sequence; fine-tuned net | +.023 / −.012 / −.008; −.025 / −.004 / .000 | `pv2-s1*-screen-v1` | FAIL |
| **`planner_pv2_tiles`** | **Tile-token transformer** (19 tile tokens + 4 player + 1 global; 555k params; test AUC .786); NumPy serving, p50 ≈ 6 ms per decision | 80-seed screen +.019 / **+.050** / **+.097** (FAIL: all-tables rule); **512-seed precise +.045 [+.024, +.067] / +.044 [+.026, +.062] / +.106 [+.087, +.125]** | `pv2-tiles-screen-v1`, `pv2-tiles-precise-v1` | strongest candidate |
| `planner_pv2_tiles_margin` | Same + auxiliary final-VP-margin head | pending (offline gate, then +.02 screen vs `planner_pv2_tiles`) | `experiments/tiles-margin.json` | in progress |

### Placement search (offline experts, branch only: `claude/placement-search`)

At its two setup settlements, play out the served bot's top-12 vertices 24 times each and place the best; ~30 s per
placement, so these are experts, not servable bots.

| name | playout opponents | vs served (tactical / mixed / self) | receipt |
|---|---|---|---|
| `planner_placement_search` | served bot in every seat | +.020 / — / **+.117** | `placement-search-screen-v1` |
| `planner_placement_mix` | per-playout mix of served/Tactical/Builder/Expander | **+.100** / +.044 / +.044 | `placement-mix-screen-v1` |
| `planner_placement_infer` | per-seat styles inferred from observed placements | **+.103** / +.025 / +.031 | `placement-infer-screen-v1` |

All three FAIL the all-tables bar. The fast learned placement evaluator (`scripts/placement_model.py`) failed its
offline gate: +.0069 [−.0103, +.0247] pick − served (`placement-model-v1`).

### Other screened ideas

Value leaves, finish search, hand-written leaves, opening pairs and more are in POLICIES.md "Tried and not taken".

## External AIs

| AI | source | how it decides | how we connect | status |
|---|---|---|---|---|
| **Catanatron AlphaBeta** depth 2 (1.5 s cap) and depth 3 (20 s cap) | [bcollazo/catanatron](https://github.com/bcollazo/catanatron), pinned `ecf9311`, MIT | Expectimax/alpha-beta over a hand-tuned value function; the author's best bot is depth 2 | `external` policy: belief-sampled adapter into Catanatron's engine (`src/catan_ai/external_worker.py`) | **Retired for `planner_pv2_tiles`**; kept for the served bot |
| **JSettlers2 robots** (SMART and FAST strategies) | [jdmonin/JSettlers2](https://github.com/jdmonin/JSettlers2) 2.6.10, pinned `c5141056`, GPLv3 (kept under gitignored `runs/external`) | Plan-based: building-speed estimates, per-player win-time estimates, a trade negotiator; no lookahead search | Third-party robot bridge (`tools/jsettlers`); optional seed patch for paired luck and all-SMART tables | **Current automated bar** |
| **STAC POMCP-TS-CR** | [ruflab/StacSettlers](https://github.com/ruflab/StacSettlers) `c9e3b66` (Dobre & Lascarides), GPLv3 + MIT MCTS jar | POMCP tree search over sampled hidden hands, 10k iterations × 4 threads; tree seeded and rollouts guided by human action-type statistics; typed actions (type, then target) | STAC bridge (PR #41, under review): our brain inside STAC's Simulation; player trading off; 10 s soft cap | Smoke only; likely the strongest automated opponent |
| STAC Stac agent | same | Rule-based with negotiation (Guhe & Lascarides 2014) | same | Smoke only |
| STAC JSettlers 1.1.16 (FAST) | same | Older JSettlers robot | same | Smoke only |
| Settlers-RL | [henrycharlesworth/settlers_of_catan_RL](https://github.com/henrycharlesworth/settlers_of_catan_RL) | PPO, tile-token attention, multi-head actions | not built (its simulator needs a rules audit) | Not benchmarked |
| HexMachina / LLM agents | [arXiv 2506.04651](https://arxiv.org/abs/2506.04651v1) | LLM agents that evolve strategy code in Catanatron | not built | Literature only |
| Colonist.io bots | colonist.io | — | Not used: its terms prohibit automated access and collecting derived statistics without written consent | Excluded |

## Benchmark results

### Internal (paired, vs served `planner_offer_accept`)

| bot | tactical | mixed | self | seeds/table | receipt |
|---|---|---|---|---|---|
| `planner_pv2_b_fix` | +.025 [+.003, +.048] | +.015 [−.004, +.033] | +.019 [+.002, +.036] | 512 | `pv2-b-fix-precise-v1` |
| **`planner_pv2_tiles`** | **+.045** [+.024, +.067] | **+.044** [+.026, +.062] | **+.106** [+.087, +.125] | 512 | `pv2-tiles-precise-v1` |
| `planner_placement_mix` (expert) | +.100 [+.038, +.162] | +.044 [−.017, +.105] | +.044 [−.012, +.099] | 80 | `placement-mix-screen-v1` |

### External ladders (one bot vs three copies; fair share .25)

| opponent | served `planner_offer_accept` | `planner_pv2_tiles` | games each | receipts |
|---|---|---|---|---|
| Catanatron depth 2 | .488 [.444, .533] | **.613 [.568, .659] — retired** | 512 | `external-ladder-v1`, `external-ladder-tiles-v1` |
| Catanatron depth 3 | .455 [.410, .501] | **.545 [.501, .589] — retired** | 512 | same |
| JSettlers2 2.6.10 robots | .373 [.332, .416] | **.447 [.405, .491]** | 512 | `jsettlers-match-v1`, `jsettlers-match-tiles-v1` |

JSettlers runs were unseeded (before the seed patch); the served bot's .373 was identical with or without a FAST
robot at the table (94/252 vs 97/260). Each Catanatron/JSettlers comparison between our two bots is unpaired.

### STAC (smoke only, not a benchmark)

| matchup | games | our wins | note |
|---|---|---|---|
| served vs 3 × POMCP-TS-CR | 8 | 0 (Wilson 95% 0–.32), mean VP 6.0 | 0 bridge errors |
| served vs 3 × STAC JSettlers 1.1.16 | 8 | 3 (.14–.69) | — |
| STAC's own runs: POMCP-TS-CR vs 3 × STAC JSettlers 1.1.16 | 39 | .615 [.459, .751] | player trading off, unseeded |

A predeclared STAC benchmark (served and `planner_pv2_tiles` vs POMCP-TS-CR, seats rotated) follows the bridge's
review.

## What we have learned

- **Where wins are decided.** Oracle playout diagnostics (`search-diagnostic-v1`, `phase-diagnostic-v1`): a
  selector choosing among build/trade options on 32 noisy playouts gains under 1 pp per decision (label
  reliability .23–.30), while opening settlement placement shows +.105 [+.035, +.180] per decision vs Tactical
  (reliability .82). Robber placement and trade answers show no headroom. The split-half figures estimate a
  finite-sample selector's gain, not perfect play's.
- **Search gains follow the opponent model.** Placement search gains against whichever opponents its playouts
  assume (served-only playouts: +.117 self, +.020 Tactical; mixed: +.100 Tactical, +.044 self). Inferring styles
  from observed placements did not recover both.
- **Architecture.** On the same 100k games, tile tokens beat flat features: AUC .786 (tiles) vs .781 (hex CNN),
  .774 (who-wins + turns-left heads), .772 (residual MLP), .746 (plain MLP before symmetry augmentation)
  (`arch-bakeoff-v1`). Within-decision cross-eval: tiles +.0062 [+.0027, +.0098] (old pv2b data, truncation
  unrecorded). Every architecture over-ranks "build a city" against "end the turn" (.92 vs .56 in playouts), a
  training-distribution bias.
- **Strength gap.** Our bots answer in milliseconds and look one turn ahead; STAC's POMCP spends ~0.8 s and 10k
  full-game simulations per decision. The 0/8 smoke points at deeper sampled-world search with a learned value and
  prior (AlphaZero-style) as the next large lever.

## Rules for adding to this file

- Add a policy family or an external AI here when it is registered or connected; add a measured row only with its
  receipt name.
- Keep the served ladder and "tried and not taken" in POLICIES.md; summarise here, don't copy.
- Never commit learned weights (metadata and hashes only) or GPL source from JSettlers/STAC (they live under
  gitignored `runs/external`).
