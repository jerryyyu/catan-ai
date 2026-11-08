# Bot policy ledger

Last reconciled: **2026-11-03**.

**Served default: `planner_offer_accept`** (ladder row 7), for new solo web and API games. Rollback:
`planner_offer_guard` (row 6). Tactical is the fallback whenever the fitted checkpoint is unbound. The three-human
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
| 7 | 2026-11-03 | `planner_offer_accept` | Guarded offers weighted by each recipient's public acceptance rate (accepted + 2) / (answered + 2) over its answers to any seat; identical to the guard until a seat answers | row 6 | mixed only: **+.068** [+.052]; DEV mixed +.050 (bar +.03); point floor tactical −.001 / self −.007 / strict +.002; mixed accept rate .31 → .71 | DEV PASS, HELD-OUT PASS (mixed); table held-out non-inferiority PASS: tactical −.009 [LCB −.027], self +.002 [−.016], strict .000 [−.018] (floor −.03, α .05/3) | #20 (bot), #23 (serving) | `offer-accept-heldout-v1`, `offer-accept-mixed-dev-v1`, `offer-accept-dev-v1`, `offer-accept-diagnostics-v1`, `offer-accept-heldout-tables-v1` |
| 6 | 2026-11-02 | `planner_offer_guard` | Red-team fixes to offers: none to seats on 8+ public points; gain net of .5 × the recipient's next-turn build gain | row 5 | DEV only: +.006 / +.006 / −.022, strict −.007; accepted trades to 8+ point seats 41.4 → 0 per 100 games (tactical) | point-floor PASS (−.03 on observed lift, not a confidence bound); no own held-out; does not inherit row 5's lift | #19 | `offer-guard-dev-v1`, `offer-diagnostics-v1` |
| 5 | 2026-11-01 | `planner_unified_offer` | Player trade offers: when the turn search would end the turn, propose the 1-for-1 or 2-for-1 exchange that most raises the shared score, weighted by the card-counted chance the recipient holds the card; at most one per turn, margin .25 | row 4 | **+.271 / +.115 / +.059** [+.247 / +.093 / +.037]; DEV +.270 / +.096 / +.082 | DEV PASS, HELD-OUT PASS; acceptor-dependent: +.003 vs `planner_unified` at a strict-acceptor table (`offer-strict-dev-v1`, bar +.03 FAIL) | #18 | `planner-offer-heldout-v1` (receipt names the arm `planner_offer`, snapshot 14e7ab3) |
| 4 | 2026-11-01 | `planner_unified` | One shared score `V(me) − 0.5·max V(opp)` for robber, steal, discard, development-card play and trade answers; build search unchanged | row 2 | **+.112 / +.066 / +.023** [+.085 / +.041 / +.001]; DEV +.078 / +.088 / +.053 | DEV PASS, HELD-OUT PASS (self bound thin) | #17 | `planner-unified-heldout-v1` |
| 3 | 2026-11-01 | `planner_devfix` | Correctness fix: no random development-card play before the roll | row 2 | DEV only: −.026 / +.022 / −.008 | non-inferiority PASS (−3 pp floor) | #15 | `planner-devfix-dev-v1` |
| 2 | 2026-11-01 | `planner_parameter` | Bounded public-information planner with eight fitted economic leaf weights | Tactical | **+.110 / +.227** (tactical / mixed) [+.096 / +.208]; external panels withheld on comparator faults | family FAIL (external faults), promoted by owner on the Tactical panels | #9 | `economic-parameters-confirmatory-supervisor-v1/readout` |
| 1 | 2026-10-31 | `tactical` | Rule-based tactical bot (now the fallback) | Heuristic | **+.118** vs three heuristics [+.085] | HELD-OUT PASS | initial commit | `tactical-heldout-v1` |
| 0 | 2026-10-29 | `heuristic` | Baseline heuristic; the frozen comparator | — | — | baseline | — | — |

Attribution: rows 4, 5, 6 and 7 each changed one thing against their stated comparator. Row 4 was measured against
row 2, not row 3, so no direct claim is made that it beats row 3.

## Tried and not taken

Each was screened against the served bot of its day with a predeclared bar. A failed bar is a result and is not
retuned on the same boards.

| date | policy | idea | vs | DEV lift: tactical / mixed / self | outcome | receipt |
|---|---|---|---|---|---|---|
| 2026-11-05 | `planner_pv2_b_fix` (precise) | Same arm, 512 fresh seeds per table (2,048 pairs) | `planner_offer_accept` | +.025 [+.003, +.048] / +.015 [−.004, +.033] / +.019 [+.002, +.036] | Gate (lift ≥ 0 every table) PASS; held-out eligibility (CI excluding zero on every table) NOT met: mixed CI touches zero. Real but small gain (~+2 pp), consistent with the 128-seed screen | `pv2-b-fix-precise-v1` (ultra) |
| 2026-11-07 | `planner_placement_infer` | Placement search with playout opponents drawn per seat from a style posterior inferred from their observed setup settlements (served/Tactical/Builder/Expander; MAP 106/120 offline) | `planner_offer_accept` | **+.103** (Tactical) / +.025 (mixed) / +.031 (self) | FAIL (+.05 on all three tables; 80 seeds). Tactical approx 95% CI [+.041, +.165]; mixed [−.028, +.078]; self [−.018, +.080]. No better than the uniform mix (+.100/+.044/+.044) and self stays far below served-only playouts (+.117). Likely: no evidence at our first placement when we place early (uniform), and served vs Builder placements are often identical (ambiguous posterior). Offline expert; no latency claim | `placement-infer-screen-v1` |
| 2026-11-07 | `planner_placement_mix` | `planner_placement_search` with each playout's three opponents drawn from {served, Tactical, Builder, Expander} | `planner_offer_accept` | **+.100** (Tactical) / +.044 (mixed) / +.044 (self) | FAIL (+.05 on all three tables; 80 seeds). Tactical passes, approx 95% CI [+.038, +.162]; mixed [−.017, +.105]; self [−.012, +.099]. With served-only playouts (v1) the pattern was reversed (+.020 Tactical / +.117 self): the playout opponent model decides which opponents the placement beats. Pool contains the table styles (stated in advance). Offline expert; no latency claim | `placement-mix-screen-v1` |
| 2026-11-07 | `planner_placement_search` | Served bot except its two setup settlements: served top-12 vertices × 24 playouts (served bot in all seats; deck/dice sampled; common random numbers), most wins placed. Follows `phase-diagnostic-v1` (placement headroom +.105 vs Tactical) | `planner_offer_accept` | +.020 (Tactical) / **+.117** (self) | FAIL (both panels required at +.05). Self-play passes with normal-approx 95% CI about [+.066, +.168]; Tactical does not (≈[−.059, +.098]). Opposite of the diagnostic's pattern: rollouts model every opponent as the served bot, so the search optimizes against served opponents. Offline expert (~30 s/placement); no latency claim | `placement-search-screen-v1` |
| 2026-11-05 | `planner_pv2_s1_m1` | `planner_pv2_s1` with value-v2-m1 (arm B fine-tuned on 100k + pv2b visited rows + pv2b rankings; selected epoch 3, validation within-decision loss flat .01003–.01025) | `planner_offer_accept` | −.025 / −.004 / .000 | FAIL (+.05 bar); fine-tune did not move the ranking loss | `pv2-s1-m1-screen-v1`, `value-v2-m1` (ultra) |
| 2026-11-05 | `planner_pv2_s1` | `planner_pv2_b_fix` + commit to the picked sequence within a turn + purchases scored by averaged predictions | `planner_offer_accept` | +.023 / −.012 / −.008 | FAIL (+.05 bar). Per-table SE ≈ .023 at 128 seeds: this, the fix arm (+.023/+.020/+.025) and S1+M1 are within about one SE of each other and of zero | `pv2-s1-screen-v1` (ultra) |
| 2026-11-05 | `planner_pv2_b_fix` | `planner_pv2_b` with the red-team integration fix (served offers kept; end-now through the served end-of-turn path) | `planner_offer_accept` | +.023 / +.020 / +.025 | FAIL (+.05 bar), but positive on every table and Tactical non-inferiority (>= −.02) met; integration diagnostic: accepted offers per own turn −2.6% vs served (parked arm −33%) | `pv2-b-fix-screen-v1`, `pv2-b-fix-offer-diagnostics-v1` (ultra) |
| 2026-11-05 | `planner_pv2_b` | PV search: served search proposes end-now + 4 best sequences; encoder-v2 state-value net (100k self-play games, 12-symmetry augmentation, residual MLP; test AUC .772; pv2b within-decision cross-eval +.0045 [LCB +.0008]) picks | `planner_offer_accept` | −.076 / −.025 / +.008 | FAIL (+.05 bar). Self-play near the offline per-decision gain; worse against Tactical and the mixed table, which the self-play-only training data never contains | `pv2-b-screen-v1` (ultra) |
| 2026-11-03 | `planner_handwritten_leaf` | Build-search leaf written by hand from game knowledge (no fitting; weight-space-programming idea): 12/VP, ~1/pip + sqrt diversity, .8/card, >7 penalty, ports, spots, −.5 × strongest opponent | `planner_offer_accept` | −.006 / −.002 / −.063 | FAIL (+.05 bar). Parity vs Tactical/mixed, worse in self-play; descriptive AUC .661 vs shared .659 / leaf .583 (`handwritten-leaf-auc-v1`) | `handwritten-leaf-dev-v1` |
| 2026-11-03 | `planner_finish_next` | Next-turn finish search at ≥ 8 points: first action maximizing the sampled chance of winning next turn (depth-3 continuations, 128 dice samples, bank-trade affordability); overrides served at ≥ .05 more chance | `planner_offer_accept` | +.016 / +.020 / +.020 | FAIL (+.05 bar); consistent small gain like the older same-turn finish (+1 to +3 pp). v1 aborted on an IllegalAction (outcomes unread), v2 after a legality fix | `finish-next-dev-v2` |
| 2026-11-03 | `planner_value_leaf_end` | Option A: learned model refit on end-of-turn positions (offline gate PASS: AUC .706 vs .659 shared / .583 leaf, LCB +.032, `value-gate-end-v1`), train-only scale, as build-search leaf | `planner_offer_accept` | −.072 / −.104 / −.119 | FAIL (+.05 bar). Closer than the turn-start arms; a 12-game check shows ~2× 4:1 bank trades (170 vs 78), fewer cities, smaller final hand. **Linear-state-model leaf lane closed at this scale** (owner decision before the run) | `value-leaf-end-dev-v1` |
| 2026-11-03 | `planner_value_leaf_scaled` | Repair of `planner_value_leaf`: same model, log-odds mapped onto served leaf units (train-only mean/SD match, slope 76.4) | `planner_offer_accept` | −.492 / −.418 / −.197 | FAIL (+.05 bar). Cause: trained at turn start, applied to within-turn afterstates; negative weights on purchase deficits penalize building (spent cards raise deficits) and reward 4:1 bank trades (232 vs 78 in a 12-game check; dev buys 14 vs 34) | `value-leaf-scaled-dev-v1` |
| 2026-11-03 | `planner_value_leaf` | Build-search leaf = learned win log-odds (offline gate PASS: test AUC .703 vs .652 shared / .588 leaf, `value-gate-v1`) | `planner_offer_accept` | −.184 / −.230 / −.148 | FAIL (+.05 bar). Integration defect: the search adds `purchase_value` (≈8, fitted-value units) to the log-odds leaf, so development-card purchases dominate (126 vs 34 buys in a 12-game check) | `value-leaf-dev-v1` |
| 2026-11-03 | `planner_opening_pair` | First settlement chosen for the best pair it allows after modelled pip-greedy opponent picks (top 10 × 10); everything else served | `planner_offer_accept` | +.016 / −.012 / +.039 | FAIL (+.05 bar); followed the self-play diagnostic `selfplay-features-v1`; code and config parked on branch `claude/opening-pair` (6d1e92c) | `opening-pair-dev-v1` |
| 2026-11-02 | `planner_offer_sparse` | Guarded offers, margin .5, at most one per three own turns | `planner_unified_offer` | −.139 / −.063 / −.027, strict −.004 | FAIL (non-inferiority, −.03 floor); cadence matters against acceptors who take most offers | `offer-sparse-dev-v1` |
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
