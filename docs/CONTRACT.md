# Catan AI: current contract

What is served, what has been shown, and what has not. This page is the
current state; the chronological research record is `experiments/README.md`
and top-level receipts plus run-directory summaries and readouts are indexed in
[results/LADDER.md](results/LADDER.md).

## Served default

- **Policy:** `planner_pv2_tiles` for new solo web and API games since
  2026-11-08, by owner promotion (Jerry). It is `planner_offer_accept` (below)
  with its own-turn build decisions chosen by PV search: the served search
  proposes ending now plus its four best build sequences, and the arch-bakeoff-v1
  `tiles` value transformer (19 tile tokens + 4 player + 1 global, 555k
  parameters, served in NumPy) scores each afterstate. Against
  `planner_offer_accept`, a precise 512-seed-per-table measurement
  (`runs/ultra/pv2-tiles-precise-v1`) gives paired lifts of **+.045 [+.024,
  +.067]** tactical, **+.044 [+.026, +.062]** mixed and **+.106 [+.087, +.125]**
  self. The predeclared 80-seed screen (`runs/ultra/pv2-tiles-screen-v1`) FAILED
  its +.05-on-all-three-tables rule (tactical +.019, mixed +.050, self +.097).
  Descriptive external benchmarks (below): Catanatron AlphaBeta depth 2 .613,
  depth 3 .545 (`runs/ultra/external-ladder-tiles-v1`); JSettlers robots .447
  vs .373 for `planner_offer_accept` (`runs/jsettlers-match-tiles-v1`).
- **Identity:** the 16-game trace hash of `planner_pv2_tiles`
  (`scripts/trace_hash.py --seed 2690000 --games 16 --policy planner_pv2_tiles`)
  is c7d07a76c9f3249f170ade344e3564df62b3a127c4b311b05eeea586334cb815 on both
  the evaluated snapshot 029fc07 and main. The weights
  (`checkpoints/value_tiles_v1.npz`) are a local artifact, never committed; the
  committed `checkpoints/value_tiles_v1.meta.json` pins their SHA256
  (27dd7818…), and the loader refuses a file that differs, a changed tokenizer,
  or a NumPy forward pass that disagrees with the stored torch logits. Games
  record both the parameter checkpoint and the weights SHA256; a restart refuses
  a game whose weights changed.
- **Fallback chain:** without NumPy or verified tiles weights, new games use
  `planner_offer_accept` (`/api/bots`: `fallback_reason` "Tiles value weights
  missing or unverified", with the loader's error in `fallback_detail`); without
  a bound parameter checkpoint, Tactical. The weights are verified once per file
  version, not per request.
- **Rollback / previous default, `planner_offer_accept`:** served 2026-11-03 until
  the tiles planner (owner-approved). It is `planner_offer_guard` (below) with each
  offer's gain and help penalty scaled by the recipient's public acceptance rate,
  (accepted + 2) / (answered + 2) over that seat's answers to any player this
  game; before any answer it offers exactly like the guard. Against the guard:
  DEV mixed +.050 (bar +.03, `runs/offer-accept-mixed-dev-v1.json`); point floor
  tactical −.001, self −.007, strict +.002 (floor −.03, not a confidence bound,
  `runs/offer-accept-dev-v1.json`); held-out mixed **+.068, paired-normal LCB
  +.052** (2,400 pairs, `runs/offer-accept-heldout-v1.json`). A second held-out
  (`runs/offer-accept-heldout-tables-v1.json`, 3,600 pairs per table) passed
  confidence-based non-inferiority on the other tables: tactical −.009 (LCB
  −.027), self +.002 (−.016), strict .000 (−.018), floor −.03 at α .05/3. No
  human-play transfer is claimed.
- **Guard (`planner_offer_guard`, served 2026-11-02 until the acceptance model):**
  Since 2026-11-02 (owner-approved). It is `planner_unified_offer` with two red-team
  fixes: it never offers a trade to a seat showing 8 or more public points, and
  an offer's gain is net of half the recipient's own next-turn build gain.
  Non-inferiority DEV vs `planner_unified_offer` (`runs/offer-guard-dev-v1.json`,
  1,280 pairs per table, floor −.03): +.006 tactical, +.006 mixed, −.022 self,
  −.007 strict-acceptor table. Accepted trades to 8+ point seats fall from 41.4
  to 0 per 100 games at the tactical table (`runs/offer-diagnostics-v1.json`).
  The screen is a point-estimate floor, not confidence-based non-inferiority:
  it does not establish zero strength cost. The guard has no held-out test of
  its own and does not inherit `planner_unified_offer`'s held-out lift. It is a
  safety change served on owner approval; no human-play transfer is claimed.
- **Offers (`planner_unified_offer`, served 2026-11-01 until the guard):** It is `planner_unified`
  plus player trade offers: when the turn search would end the turn, it proposes
  the 1-for-1 or 2-for-1 exchange that most raises its position score, weighted
  by the card-counted chance the recipient holds the card, at most once per turn.
  Held-out test against `planner_unified` (`runs/planner-offer-heldout-v1.json`,
  7,200 pairs on fresh boards): +.271, +.115 and +.059 paired lift on the
  tactical, mixed and self tables, lower bounds +.247, +.093 and +.037. The
  tactical-table lift partly measures exploitation of Tactical's count-based
  acceptance (it takes any offer giving it at least as many cards); the self
  table, whose acceptors take only offers that raise their own hand score, is
  the cleaner read. No human-acceptance claim is made. Against a strict acceptor (takes only clear gains, only from seats behind it) the offers measure +.003 vs `planner_unified` (`runs/offer-strict-dev-v1.json`, bar +.03 FAIL): the lift depends on the partner. The receipts name the arm
  `planner_offer` (snapshot 14e7ab3, where it shadowed the older offer
  prototype); it is registered as `planner_unified_offer` from this release.
- **Previous default, `planner_unified`:** keeps the fitted build search of
  `planner_parameter` and scores every other decision (robber, steal, discard,
  development-card play, answering trade offers) with one position score,
  `V(me) − 0.5 · max V(opponent)`. Opponents' hands are public card-counting
  estimates, never hidden state. Held-out vs `planner_parameter`
  (`runs/planner-unified-heldout-v1.json`): +.112, +.066, +.023, lower bounds
  +.085, +.041, +.001. `planner_unified`, `planner_devfix` and
  `planner_parameter` stay selectable. Tactical is the fallback, the three-human
  pilot rooms keep Tactical as their default, and `heuristic` remains the frozen
  comparator. Every served change is in [POLICIES.md](POLICIES.md).
- **Why the fitted line (`planner_parameter`) was promoted:** both Tactical-control panels of the sealed held-out family passed
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

## Next challenger

None declared. Parked on 2026-11-01 (DEV FAIL vs `planner_unified`, not
retuned): award-race value (`planner_race`) and the combined race + block +
offer arm (`planner_shared`, self table +.031). See POLICIES.md "Tried and not
taken".

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

The external comparison was rerun on the repaired comparator v13 as a new,
predeclared two-stage plan. The 64-board DEV stage passed with zero faults
(+.082 and +.195). The 1,600-board held-out stage sealed on 2026-11-01 with
verdict FAIL on 27 external fault violations (26 failed decisions and one
turn over 10 s) in 25,600 games, again against a zero-fault
rule, so its lifts are withheld. The legality mismatches did not recur. The new
faults are belief-solver timeouts late in very long games, 24 of them in one
game (`experiments/README.md`, 2026-11-01). A post-hoc descriptive read,
requested by the owner, gives lifts of +.061 (tactical) and +.151 (mixed) with
plain 95% intervals well above zero. Excluding or penalising the three faulted
boards changes neither by more than .002. The owner decided to promote on
2026-11-01; the serving change is a separate PR.

## External benchmarks (descriptive, not gates)

The owner reopened external benchmarking on 2026-11-07 as information, with a
predeclared retirement rule: an external bot is retired as a benchmark when the
served bot's win share, alone against three copies, is at least .50 with the 95%
lower bound above .40 (fair share .25). Neither has been retired. The first three
rows measured `planner_offer_accept` (served until 2026-11-08); the last three
measured `planner_pv2_tiles` before its promotion. Against Catanatron its shares
meet the retirement rule's numbers at both depths; retiring a benchmark is the
owner's call and has not been recorded.

| served bot | opponent | games | win share | 95% interval | receipt |
|---|---|---|---|---|---|
| `planner_offer_accept` | Catanatron AlphaBeta depth 2 (1.5 s cap) | 512 | .488 | [.444, .533] | `runs/ultra/external-ladder-v1` |
| `planner_offer_accept` | Catanatron AlphaBeta depth 3 (20 s cap) | 512 | .455 | [.410, .501] | `runs/ultra/external-ladder-v1` |
| `planner_offer_accept` | JSettlers2 2.6.10 built-in robots | 512 | .373 | [.332, .416] | `runs/jsettlers-match-v1` |
| `planner_pv2_tiles` | Catanatron AlphaBeta depth 2 (2 s cap) | 512 | .613 | [.570, .654] | `runs/ultra/external-ladder-tiles-v1` |
| `planner_pv2_tiles` | Catanatron AlphaBeta depth 3 (25 s cap) | 512 | .545 | [.502, .588] | `runs/ultra/external-ladder-tiles-v1` |
| `planner_pv2_tiles` | JSettlers2 2.6.10 built-in robots | 512 | .447 | [.405, .491] | `runs/jsettlers-match-tiles-v1` |

Catanatron games ran on our engine with the belief-sampled adapter (zero
faults, zero truncations). JSettlers games ran on the JSettlers server through
the third-party-robot bridge (`tools/jsettlers`; zero bridge errors, one
fallback in 50,586 decisions); its server RNG is unseeded, so those games are
not reproducible and are identified per game instead. JSettlers' robots are the
harder table. Colonist.io was not used: its terms prohibit automated access and
collecting derived statistics without written consent.

## What is not claimed

- No competitive strength against ranked online humans. No Colonist
  integration exists or is authorized; its terms restrict automation and data
  use.
- Beating Catanatron (either depth) or JSettlers' robots by the margins above
  is not beating a strong player; both are open-source baselines, and the
  JSettlers win share is well below the retirement bar. Catanatron's belief
  recovery still faults rarely: on 3 of 3,200 held-out clusters in the v13
  rerun, all through solver timeouts in long games (none in the 2026-11-07
  ladder).
- DEV passes are screens. Only a reserved held-out run with a predeclared
  analysis supports a promotion.
- The served `planner_pv2_tiles` is the exception, and no held-out claim is
  made for it: its predeclared screen FAILED (tactical +.019 against a +.05
  bar on every table), it has no reserved held-out run, and it was promoted by
  the owner on the precise 512-seed measurement and the descriptive external
  benchmarks. Its lift over `planner_offer_accept` is a measurement, not a
  passed gate.
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
