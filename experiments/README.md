# Experiments

## Research: competitive four-player Catan AI

### Loss attribution, pass 2: event-level replay of swing boards (2026-10-31)

`scripts/replay_swing_boards.py` replayed 100 boards per direction from the
sealed tactical-control panel (`planner_parameter` vs `tactical`), both arms,
with the evaluator's exact construction under the frozen snapshot: 400 games,
zero outcome mismatches (`runs/swing-replay-tactical-v1.json`). Means are for
the candidate's seat; "candidate arm" is `planner_parameter` in that seat,
"control arm" is Tactical in the same seat on the same board.

| Metric (candidate seat) | Boards control won, candidate lost | Boards candidate won, control lost |
|---|---|---|
| final points, candidate arm / control arm | 6.5 / 10.1 | 10.1 / 6.3 |
| production received | 65 / 85 | 85 / 68 |
| cards discarded | 11.6 / 5.3 | 12.8 / 5.5 |
| development cards bought | 1.8 / 5.6 | 2.6 / 4.5 |
| largest army at end | .05 / .44 | .06 / .19 |
| longest road at end | .20 / .52 | .61 / .18 |
| player offers made | 0 / 22.4 | 0 / 22.5 |
| bank trades | 4.8 / 11.3 | 6.9 / 9.1 |
| first city turn | 46.7 / 48.1 | 48.9 / 57.3 |
| never built a city | .30 / .05 | .02 / .35 |

Readings, descriptive, from 200 boards:

- **Discards.** The fitted planner discards about twice as many cards as
  Tactical on the same boards in both directions. Its leaf charges .5 per card
  over seven; the hand term rewards holding up to three of each resource. It
  hoards for a build it then loses to a seven. A discard-aware hand valuation is
  a concrete, cheap candidate.
- **Awards.** On boards the planner lost, Tactical held Largest Army 44% and
  Longest Road 52% of the time from the same seat; the planner held them 5% and
  20%. It buys a third as many development cards. Two awards are four points,
  which matches losses by three or more. Valuing knights toward Largest Army
  and roads toward Longest Road as award races, not just as development cards
  and road count, is the second candidate.
- **Production.** Production received is the symmetric driver: 65 vs 85 when
  losing, 85 vs 68 when winning. Placement and expansion already carry most of
  the fitted weights; the robber lands on the planner's tiles slightly more
  often in both directions, consistent with it being the apparent leader.
- **Trading** is zero offers, as already known; row 6 measured what one offer
  per turn is worth.

Each of these is a lane proposal with a powered config and a bar before it
runs, not a result.

### Row 6 first test: one priced offer per turn, FAIL on the family rule, PASS tactical (2026-10-31)

`offer-powered-dev.json`: `planner_offer` (the `planner_robber` turn search plus
one economically-priced one-for-one offer to the trailing opponent when the
search would otherwise end the turn) against unchanged `planner_robber`. 512
fresh boards per panel (1604000 to 1605511), all seats, cap 4, 8,192 games,
zero truncations, 15 minutes (`runs/offer-powered-dev-v1.json`).

| Panel | Pairs | Paired lift | Hoeffding LCB | Cluster SD | MDE at 80% | Bar | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| tactical | 2048 | +.1558 | +.0357 | .297 | .037 | +.05 | pass |
| mixed | 2048 | +.0430 | -.0771 | .256 | .032 | +.05 | fail |

The predeclared bar required both panels, so the screen **fails** and this
exact candidate is parked; nothing is retuned on these boards. The tactical
panel lift of +15.6 pp is the largest single-change DEV effect recorded apart
from the fitted economic weights, and it was reached with a 0.048 s worst step.
The mixed panel (builder, expander, tactical) shows +4.3 pp with an MDE of
3.2 pp: a smaller real gain. The next row should ask why the same offers help
less at a mixed table before any pricing model is added: candidate hypotheses
are that heuristic-style partners accept more of the offers that also help
them, and that the trailing-recipient rule picks a different partner there.
That diagnosis belongs to loss attribution pass 2 on this receipt's swing
boards, with its own predeclared config.

### Powered rescreen: finish search, FAIL, precisely small (2026-10-31)

`finish-powered-dev.json` reran the unchanged `planner_finish` against the
unchanged `planner_robber`: 512 fresh boards per panel (1600000 to 1601511),
all seats, cap 4, 8,192 games, zero truncations
(`runs/finish-powered-dev-v2.json`; v1 aborted at shard 39 on a source edit and
is preserved as an aborted run, not a result).

| Panel | Pairs | Paired lift | Cluster SD | MDE at 80% | Bar | Verdict |
|---|---:|---:|---:|---:|---:|---|
| tactical | 2048 | +.0254 | .080 | .010 | +.05 | fail |
| mixed | 2048 | +.0195 | .071 | .009 | +.05 | fail |

The cluster SD is a quarter of the opening lane's because the two arms play
identical games until someone nears ten points, so this screen could see a
1 pp effect. The finish search is worth about +2 to +2.5 pp: real, precisely
measured, and below the bar. That matches loss attribution pass 1, where only
14% to 19% of candidate losses were by one point. Candidate worst turn 1.543 s
against the 10 s gate. The lane stays parked; the component is cheap and
combinable.

### Powered rescreen: opening search, FAIL but now measured (2026-10-31)

`opening-search-powered-dev.json` reran the unchanged `planner_opening` against
the unchanged `planner_robber` on 512 fresh boards per panel (seeds 1600000+
block 1602000 to 1603511), all seats, cap 4, 8,192 games, zero truncations
(`runs/opening-search-powered-dev-v1.json`).

| Panel | Pairs | Paired lift | Cluster SD | MDE at 80% | Bar | Verdict |
|---|---:|---:|---:|---:|---:|---|
| tactical | 2048 | +.0493 | .304 | .038 | +.05 | fail |
| mixed | 2048 | +.0298 | .319 | .040 | +.05 | fail |

The bar stands and the candidate stays parked. What changed is the knowledge:
the 8-cluster screen had reported 0 pp and -9.4 pp with a minimum detectable
effect near 11 pp; the powered screen puts the true effect of this bounded
opening search in the +3 to +5 pp range, positive but below the bar, with a
design that could see 4 pp. Candidate worst step 0.991 s against the 2 s gate.
Opening search is therefore a small real gain, not a null, and a candidate
for combination with lanes that act on the whole game rather than a lane to
deepen on its own.

### Frozen economic-parameters confirmatory family: sealed, FAIL (2026-10-31)

The detached supervisor finished both comparisons and the reader wrote
`runs/economic-parameters-confirmatory-supervisor-v1/readout.json` (plan sha
e6a01ab0…, reader sha 0a2fe4b5…). 51,200 games, zero truncations.

| Control | Panel | Paired lift | Simultaneous LCB (alpha .0125) | External faults | Panel verdict |
|---|---|---:|---:|---:|---|
| tactical | tactical | +.1098 | +.0958 | 0 | pass |
| tactical | mixed | +.2266 | +.2085 | 0 | pass |
| external | tactical | withheld | withheld | 5 | fail |
| external | mixed | withheld | withheld | 23 | fail |

Family verdict: **FAIL**, as predeclared: every panel had to pass. The two
Tactical-control panels pass with lower bounds far above the +.05 target, so
`planner_parameter` is stronger than the served Tactical default on this engine
over 1,600 fresh boards per lineup. The two external-control panels fail on the
zero-fault rule: 28 external-worker faults in 12,800 games (0.2%) withhold the
lift estimate. Descriptively only, the candidate's raw win rate on those panels
was .376 vs .305 and .568 vs .411 over 6,400 games each. Candidate decision p95
was under 10 ms; the external baseline's worst step was 1.768 s against the 2 s
gate and its worst turn 3.452 s against 10 s.

Nothing is retuned on these boards. The external result is about comparator
reliability, not strength; fixing the fault cause or predeclaring a fault budget
precedes any external-control claim. Promotion of `planner_parameter` to the
served default is a separate decision; the plan's criterion was not met.
Loss attribution pass 1 on the sealed external panel
(`runs/loss-attribution-external-heldout-v1.json`) matches the tactical panel:
69% / 60% of candidate losses by three or more points, seat flat within 9 pp.

### Loss attribution, pass 1 (2026-10-31)

`scripts/loss_attribution.py` streams a receipt and attributes candidate losses
from terminal fields only (winner, turns, final points). Run on the sealed
tactical-control panel of the frozen economic-parameters family, 6,400 games per
arm per panel (`runs/loss-attribution-tactical-heldout-v1.json`). This reads a
completed panel of a running family for diagnosis; no tuning or stopping follows.

| Panel | Cand. win | Ctrl win | Losses by 1 / 2 / 3+ pts | Second-place share | Turns win / loss (cand.) | Turns (ctrl) |
|---|---:|---:|---|---:|---|---|
| tactical | .360 | .250 | .14 / .17 / .69 | .48 | 89.8 / 91.9 | 95.8 / 95.8 |
| mixed | .578 | .351 | .19 / .21 / .59 | .67 | 94.8 / 100.1 | 105.8 / 109.8 |

Readings, descriptive only:

- Most candidate losses are by three or more points. One-point losses are 14%
  to 19% of losses, so an exact finish search can address at most that slice;
  the mid-game economy and tempo gap is the larger target. The finish lane
  stays predeclared but drops below lanes that act on the whole game.
- Win rate by seat is flat within 6 pp for the candidate (.34 to .40 tactical,
  .56 to .61 mixed); seat is not a loss cause.
- In the mixed lineup (builder, expander, tactical, inserted around the
  candidate's seat), the opponent at seat offset 2 wins 41% of candidate losses
  against 31% and 28% for the others. Which style occupies offset 2 varies with
  the candidate's seat, so this points at a lineup effect to resolve in pass 2,
  not at a named style yet.
- The candidate wins faster than the control (89.8 vs 95.8 turns) and its losses
  are only slightly longer than its wins. It is not losing long games.
- 200 swing boards per direction per panel are listed for pass 2 replay.

Pass 2 replays swing boards deterministically from seed to attribute causes at
the event level (robber targeting, discards, award swings, trade refusals). It
waits for the family to finish and will use the powered-screen seeds policy for
anything it proposes.

### Opening-placement research (2026-10-30)

Opening-search v1 DEV is now specified in `opening-search-dev.json`: fresh
boards 1428000–7 / 1428100–7, all seats, 128 games. Candidate `planner_opening`
and control `planner_robber` have identical post-setup play. The candidate keeps
eight economic root placements, completes each setup under economic and raw
production opponent models, and averages final economic value. Own subsequent
placements are greedy conditional choices; this is bounded scenario search,
not exhaustive minimax. Real setup transitions grant starting cards; roads use
reachable-site geometry. Diagnostics 1427000–1 passed full snake setup and
observation immutability; first-position choice took 0.654s on this host.
Predeclared gate: +5pp in both panels, zero truncations, decision p95 <=2s,
maximum complete step <=2s and completed-turn wall time <=10s in both arms.
Turn wall time includes response work and is a conservative timing screen, not
a per-player hard compute limiter. No external comparator or human claim follows.

Opening-search v1 result: **FAIL; parked**, `runs/opening-search-dev-v1.json`.
All 128 games completed. Relative to the unchanged `planner_robber` opening,
paired win lift was 0pp in the Tactical panel and -9.375pp in mixed (32 pairs
each). Maximum candidate complete step was 0.6952s / 0.6888s; completed-turn
wall maximum was 0.1019s / 0.1819s. Both timing gates passed; both strength gates
failed. Twelve focused tests passed. This does not show that openings are
unimportant: it rejects this bounded continuation/economic-evaluation package
under the declared screen. Next useful diagnostic is whether setup evaluations
predict early city/settlement progress under realistic opponent placements;
deeper search with the same unvalidated objective is not yet justified.

Opening component diagnostic preregistration: `scripts/probe_opening_components.py`
uses fresh DEV boards 1429000–3 / 1429100–3, all seats, Tactical and mixed panels.
Four arms select baseline, searched settlements only, searched roads only, or
both, with identical post-setup policy and policy seed. This is 128 games.
Record final setup, economic evaluation, resource production, first-city and
first-post-setup-settlement turn (null means never reached before termination),
completion and maximum candidate complete-step time. An arm advances only with
>=+5pp paired win lift in each panel, no truncation and maximum step <=2s.
This small multi-arm diagnostic selects hypotheses, not a promotion claim;
later fresh confirmation must address selection. Original complete results are
preserved; this probe collects new component and progress evidence.

Component diagnostic completed: 128/128 games, maximum candidate step 0.6785s.
Relative win lifts (Tactical / mixed) were settlements-only -12.5pp / -6.25pp,
roads-only 0pp / +6.25pp, and both +6.25pp / +31.25pp. Each panel has only
16 comparisons per arm. Settlement-only increased average realized setup score
58.898→59.685 / 57.586→58.364, but first-city-by-turn-40 counts stayed 6/16 in
both arms and both panels. Thus higher economic setup scores did not deliver
better wins or this early-city diagnostic in this sample. The turn-40 cutoff is
a post-run descriptive diagnostic, not a preregistered acceptance gate.
Road-only changes can alter later settlement choices too: 7/16 final settlement
pairs changed in Tactical, 1/16 mixed, despite the same settlement policy.
These are policy-component interventions, not fixed-position causal effects.

The full search clears this small batch's gate, but its earlier larger screen
failed. Pooling the two disjoint DEV batches descriptively (48 pairs per panel)
gives +2.083pp Tactical / +4.167pp mixed, below +5pp in both. Do not cherry-pick
the second batch or promote. Preserve `runs/opening-components-v1.jsonl` and
`runs/opening-components-v1-summary.json`; the reader requires all expected
seed/seat rows. Next evaluator work should model resource bottlenecks toward
concrete early purchases; these results do not justify simply maximizing the
current economic score more deeply.

Next isolated hypothesis: `planner_purchase_opening` retains the same bounded
setup search and opponent models but adds an own-position affordability score.
It estimates rolls to afford a city or a road-plus-settlement from current cards
and pip income, permitting fractional conversion at current bank/port rates.
This fluid horizon is a proxy, not expected stochastic waiting time: dice
correlation, robber/discard losses, bank contention and human trades are absent.
Bonuses are fixed before DEV at 30/(1+city_rolls/4) and
20/(1+expansion_rolls/4); later play remains exactly `planner_robber`.
`opening-purchase-dev.json` reserves boards 1430000–7 / 1430100–7, all seats,
128 games against unchanged `planner_robber` opening. Gates remain +5pp in each
panel, zero truncations, p95 and maximum complete step <=2s, completed-turn
wall maximum <=10s. Six focused tests passed, including zero-income impossibility,
already-affordable purchases, port conversion and both full setup variants.

Purchase-opening DEV **failed; parked** (`runs/opening-purchase-dev-v1.json`).
All 128 games completed. Paired lift versus unchanged `planner_robber` opening
was -3.125pp at Tactical tables and +3.125pp at mixed tables, below +5pp in both.
Candidate complete-step maximum was 0.7154s / 0.7136s; timing gates passed.
The fluid purchase proxy did not establish strength improvement. Do not tune
its coefficients on these opened boards or promote either searched opening.
Further opening work needs a better validated target; comparator deadline and
belief reliability repair remains independently necessary for the overall goal.

External timeout diagnosis: `capture_external_timeout.py` reproduced board
1426003, external seat 1 versus Tactical, at step 675 / turn 117 / roll phase.
The observation-only capture and trace are in `runs/external-timeout-regression-v1`.
Replaying those observed decisions without policy search required 346,624 particle
transitions at that update, despite only 19 new events: loss of the warm population
triggered a full history replay. Reconstruction alone took 2.115s. A profiled cold
run spent 7.644s of 7.730s in belief sampling (profiling overhead included), so
this failure precedes AlphaBeta search. Generic deep copying was a material cost.

`Particle.__deepcopy__` now copies its known mutable containers directly while
sharing immutable keys/counts/card tuples. Seventeen focused belief/search tests
passed. On the captured cold replay, generic copying took 2.0383s and specialized
copying 1.6865s with exactly identical final particles and RNG state
(`copy-comparison.json`). This is one path measurement, not a general speed claim.
Next regression gate: rerun the full known failing seed/seat at the unchanged
2s decision / 10s turn-compute limits, requiring completion and zero fallbacks;
the existing complete-game probe also verifies source stability and identity.

The full timeout regression **passed**: external seat 1 won on turn 121 with
zero fallback decisions; maximum complete external step 1.7244s, full game
6.4196s (`runs/external-timeout-copy-repair-v1.json`). This repairs the reproduced
deadline case without changing the particle distribution or nominal budgets.
It does not resolve the separate belief failure from board 1426102, external
seat 2 in the external/builder/external/expander lineup. That regression is the
next reconstruction target; broader external strength claims remain gated.

The separate belief failure reproduced on board 1426102, external seat 2, step
311 / turn 82. Cold replays lost every particle at an earlier hidden steal,
event 266. Retrospective requirements forced the recipient to have four wool
after stealing from a three-card starting hand, hence the stolen card must be
wool. The victim also needed to retain wool. The old backward pass relaxed all
recipient resource bounds independently without reserving the forced transfer
in the victim. The revised pass detects a resource lower bound exceeding the
recipient's pre-steal total, propagates that resource to the victim and subtracts
it from the recipient. Sampling still applies the existing proposal correction.

Eleven focused tests passed, including a one-particle forced-transfer/discard
witness across ten seeds. The captured observation now reconstructs for five
RNG seeds in 0.828–1.495s (`runs/external-belief-regression-v4/forced-steal-repair.json`).
This changes retrospective proposals, so external adaptation is now v4; old v3
evidence is preserved. Next gate is all seven known failure lineups in
`probe_belief_repair.py`, every game complete, zero external faults, all external
steps <=2s and accumulated turn compute <=10s. No fresh strength claim follows
from regression success.

External v4 regression **passed all seven games**
(`runs/external-belief-repair-v4/summary.json`): every game complete, zero faults
in every external seat, maximum complete step 1.7287s and maximum accumulated
turn compute 1.7387s. Both newly reproduced failures and all five older scenarios
passed. Fresh external strength evaluation is eligible again, but these targeted
cases do not establish reliability across arbitrary boards or candidate strength.
The full Python test suite passed: 138 tests in 54.84s after these repairs.

Fresh external-v4 DEV preregistration: `planner-robber-external-v4-dev.json`
reserves 1431000–3 (Tactical opponents) and 1431100–3
(builder/external/expander), all seats, 64 games. Candidate remains
`planner_robber`, control is external v4. Require >=+5pp paired lift in both
panels, zero truncations or external faults, p95 and maximum complete step <=2s,
completed-turn wall maximum <=10s, plus all external-seat compute gates.
This is a new producer and fresh DEV data; the failed v3 journals remain intact.
Run serially on this host, one game at a time; external workers persist per seat
and are invoked sequentially. No held-out data is opened and no default changes.

Fresh external-v4 DEV **failed** (`runs/planner-robber-external-v4-dev.json`).
All 64 games completed. The fault-free Tactical panel had 0pp paired candidate
lift, missing +5pp. Mixed-panel lift was withheld: board 1431103, candidate
seat 0, external opponent seat 2 produced three belief failures and eleven
timeouts. Its first fault was a belief failure at turn 98 in the roll phase
(1.273s), followed by failed reconstruction/timeouts. The reader's 25 fault
count includes overlapping timing violations, not 25 independent fallbacks.
This invalidates mixed-panel inference; no promotion. Seven passing regression
scenarios remain valid evidence for those cases, not general reliability.

The new failure reproduced at step 421 / turn 98; observation and trace are
preserved in `runs/external-belief-regression-v5`. A cold reconstruction with
the original seed succeeds, ruling out a universally inconsistent history for
this observation. Replaying the actual warm tracker reproduces failure: its
population is eliminated by observed Year of Plenty at event 417, and both
cold retry populations are eliminated by observed Monopoly at event 235.
`warm-collapse.json` records these witnesses. Resource-only proposal guidance
does not protect particles against revealed development-card identities. The
next repair target is retrospective conditioning of hidden development draws
on already-observed plays and card-age constraints, with proposal weights and
deck conservation verified. Raising time limits or reporting fallback wins would
not resolve this inference failure.

Development proposal repair v5: before sampling a hidden purchase, reject card
identities that violate necessary bounds from the already-observed suffix.
At each later play, cumulative required cards must be coverable by existing
eligible cards, known eligible purchases and remaining unknown purchases made
strictly before that play's turn. Global deck bounds reserve known draws and
remaining per-player play deficits. These are necessary constraints, not a claim
of exact posterior sampling or complete constraint solving. Allowed identities
are sampled by remaining deck count; importance weight is allowed mass divided
by total deck mass. Resource guidance and actual transition validation remain.
Identical development/deck states share cached feasibility calculations within
one observation update; the cache is cleared before the next update.

Fourteen focused tests passed, including same-turn exclusion, eligible later
purchases, known-draw deck reservation, exact forced-draw mass 2/25 and card
conservation. Before caching, the captured failure reconstructed for three RNG
seeds in 1.592–1.618s (`development-proposal-cold-v1.json` under the capture).
Next gate: all eight known regression scenarios, complete games, zero external
faults, maximum complete steps <=2s and turn compute <=10s. Source label is now
external v5; v4 evidence remains preserved and is not relabeled.

External v5 regression **passed all eight scenarios**
(`runs/external-belief-repair-v5/summary.json`): all games complete, zero external
faults, maximum complete step 1.7587s and turn compute 1.7703s. The newest failure
board completed with maximum step 0.4657s. Sixteen focused constraint, resource
belief and deadline tests passed after caching and version labeling. A fresh
external screen is still required; regression wins are not strength evidence.

Local human-pilot infrastructure now records private source-tagged matches under
`runs/pilots` (or `CATAN_PILOT_DIR`), atomically saving action batches and server
processing times. Tactical/Heuristic games reconstruct their engine and policy
RNG state from recorded actions after a restart; incompatible source or unsupported
experimental recovery is refused while preserving records. The browser resumes
its latest match on refresh and offers a human-view replay export without seed,
opponent resource identities or hidden development cards. Private server logs
are not the shareable export. This is a single-process local pilot service.

Validation: ten focused API/integration tests passed initially; after extending
recovery to 15 human actions, eight API/pilot tests passed including identical
continuation and all bot RNG states. All 15 browser tests passed, including real
setup, refresh/resume and redacted replay export. No participant game has been
played yet; these are software tests, not human evaluation or Colonist transfer.

The local pilot server is now served from a frozen copy at
`runs/pilot-source-v1/src`, with the same private record directory and web build.
It was restarted while empty so subsequent AI edits do not change its producer
or source identity. This remains a local service, not a deployed online bot.

Monopoly hypothesis DEV: `planner_cards` changes only the resource selected when
`planner_robber` would play Monopoly. It replays observed resource deltas and uses
proportional expectations for hidden steals/discards, then normalizes to public
hand sizes. This is an approximate moment filter, not exact posterior inference;
clipping after known expenditures is a deliberate approximation. Choose the
resource maximizing weighted estimated opponent holdings. Timing of card play,
other development choices, opening and post-setup search remain unchanged.
`monopoly-history-dev.json` reserves fresh boards 1434000–7 / 1434100–7, all seats,
128 games versus unchanged `planner_robber`, Tactical and mixed panels. Require
>=+5pp lift per panel, zero truncations, p95 and maximum step <=2s, completed-turn
wall <=10s. Diagnostic 1433000 tests resource choice; observed-history privacy,
count conservation, transfer ratios and existing planner tests passed (five).

Monopoly-history DEV **failed; parked** (`runs/monopoly-history-dev-v1.json`).
All 128 games completed, with 0pp paired win lift in both panels. Maximum
candidate complete step was 0.0347s; timing gates passed. The Tactical panel
had no net seed-cluster differences; mixed gains and losses offset. This does
not establish that estimated resource choice is worse than random, but it fails
the predeclared bar and is not promoted. No holdout or external comparison is
justified for this isolated variant based on this screen.

Turn-chain hypothesis: the existing planner only permits resource acquisition
at depth zero, so it cannot evaluate two bank trades followed by a city.
`planner_chain` uses the same opening/evaluation/robber policy, but searches up
to five own actions with 96 expansions per depth, an eight-state beam, round-robin
parent expansion and duplicate-state suppression. Trades and supported progress
cards can occur below the root. Bank lower bounds come from own holdings, total
opponent card counts and observed legal acquisition actions; known payments
return cards to these bounds and known acquisitions subtract them. No opponent
hand identities are read. Development purchases remain root-only terminal value
estimates because future deck availability and identity are not established.

Five focused tests passed, including an actual two-trade-then-city win and a
one-ore bank where the second projected ore trade is refused. DEV preregistration
`chain-planner-dev.json` reserves 1436000–7 / 1436100–7, all seats, 128 games
against unchanged `planner_robber`. Require +5pp paired lift in each Tactical
and mixed panel, zero truncations, p95 and maximum complete step <=2s and
completed-turn wall <=10s. This tests deeper turn planning, not opening changes
or an external-baseline strength claim.

Turn-chain DEV **failed; parked** (`runs/chain-planner-dev-v1.json`). All 128
games completed. Paired lift versus `planner_robber` was -9.375pp at Tactical
tables and 0pp at mixed tables. Maximum complete candidate step 0.0916s and
completed-turn wall 0.3425s cleared the timing gates. The constructed multi-trade
win remains a correctness capability, not evidence of improved match strength.
This screen does not justify promoting deeper search with the current evaluator.

Evaluator diagnostic preregistration: `probe_planner_disagreements.py` takes the
first two action disagreements between `planner_chain` and `planner_robber` on
each of four fresh all-planner trajectories, boards 1437000–3. Both selectors
receive the same observation and policy RNG state. For each root, four matched
fresh RNG continuations follow each action using `planner_robber` in every seat;
up to 64 continuation games, capped at 400 additional turns. Preserve every
outcome, including truncation. Record immediate public evaluator scores separately
from eventual wins; neither is the deeper planner's internal best-leaf score.
The simulator retains the real hidden world for these offline diagnostics; no
policy receives it. These outcomes are not attainable hidden-information search
or promotion evidence. Before outcome inspection, diagnostic bar for prioritizing
an action-family evaluator repair is >=10pp adverse paired continuation win
change for chain choices, zero truncations and at least four sampled roots.
Otherwise treat the small probe as inconclusive and avoid coefficient tuning on it.

The first disagreement probe completed 64/64 continuation games from eight roots.
Road choices (four roots) had 0pp mean paired lift; trades (three roots) -8.33pp;
the one settlement choice 0pp. No action family clears the repair-priority bar.
One forced action followed by baseline replanning may erase a multi-action plan;
the result does not validate the evaluator or explain the full-game regression.
Preserve `runs/planner-disagreements-v1` as this limited diagnostic.

Follow-up protocol before outcomes: `probe_planner_disagreements.py --whole-turn`
uses fresh boards 1438000–3, first two disagreements each, four continuations per
arm. Each arm retains its respective planner for the remainder of the root turn,
then switches to common `planner_robber` continuation. Other seats use the common
policy throughout. Same zero-truncation, >=four-roots-per-action-family and
>=10pp adverse lift diagnostic bar; same true-world/non-promotion limitation.
Both package and probe script identities are recorded. This changes the causal
intervention, so it is a new run, not a replacement read of old outcomes.

Whole-turn diagnostic completed 64/64 continuations from eight roots. Road
choices (five roots across three boards) had 0pp mean paired lift; the single
trade root 0pp; two settlement roots on one board +12.5pp. No action family
met the predeclared adverse-lift repair bar. Some roots came from consecutive
decisions in the same turn and are dependent; neither roots nor four RNG forks
are independent board samples. These small probes do not validate the economic
evaluator or explain the full-game loss. Preserve
`runs/planner-disagreements-turn-v1/summary.json`; do not tune coefficients or
claim strength from it. The deeper planner remains parked.

Outcome-value learning v1 preregistration: `train_outcome_value.py` generates
256 training games (1440000–1440255) and 64 validation games
(1441000–1441063), rotating planner_robber/Tactical/builder/expander across seats.
All four observers are sampled at global turns 8, 24, 48 and 80 if reached;
only completed-game wins supply labels. Any truncated game stops generation
and fails the corpus gate; it is not silently excluded. Each completed shard
is source-tagged and resumable. Feature diagnostics use 1439000. Test boards
1442000–1442063 remain reserved and unopened.

A fixed 100-epoch logistic fit (SGD seed 7, learning rate .05/(1+epoch/20),
L2 .001, training-only standardization) uses 38 observation-only features:
existing state features plus own development types, purchase deficits, port
rates and public opponent production. Compare with a separately fitted six-feature
baseline using own/opponent points, own production, turn and own hidden VP count.
Weight each game equally in training and metrics. Before validation, require
>=5% lower game-averaged log loss than that baseline and no worse Brier score.
This is a prediction gate, not calibration certification or gameplay strength.
No policy consumes the model before this gate. First run four training shards to
measure the exact generation path; preserve those shards for the full corpus.

The four-shard capacity probe completed all games, producing 60 examples in
1.603s total (slowest game 0.705s). Two feature/privacy/prediction tests passed.
Proceed serially on this host with the remaining training shards, then the
predeclared validation shards; each terminal game is saved independently. The
existing four files are reused only after producer and completion checks.

Outcome-value v1 **passed the predeclared prediction gate**. All 320 games
completed. On 64 validation boards / 956 observation rows, learned log loss
was 0.473676 versus 0.500782 for the fitted points/production baseline, a 5.413%
relative reduction (bar: 5%). Brier score was 0.157330 versus 0.164717. These
are game-averaged prediction metrics on the specified mixed-policy distribution,
not policy-strength, human-transfer or statistical-confidence claims.
Checkpoint: `runs/outcome-value-v1/model.json`, SHA256
`b55eab9866dce201c28b67b76b5e3a71ad0250854b56e35f125c9baff5b56a0f`;
readout: `runs/outcome-value-v1/validation.json`. No validation-based coefficient
tuning occurred. Test boards remain unopened. The model is eligible for a fresh,
predeclared planner-integration DEV; no served policy consumes it yet.

Outcome-value planner integration: new `planner_value` retains the existing
opening, three-action/128-node search, robber policy and development purchase
estimate. In regular turn leaves it adds 20 times the frozen learned log-odds
to the existing economic value. This fixed scale is an integration hypothesis,
not validation-based tuning; immediate winning positions retain the terminal
score. Checkpoint SHA and feature-producing source files are checked before use;
missing/mismatched models fail rather than silently substitute a comparator.
Simulator metadata and paired-run identity include the checkpoint.

Predeclared gameplay DEV: `outcome-value-planner-dev.json`, boards
1444000–7 / 1444100–7, all seats, 128 games against `planner_robber` in Tactical
and mixed panels. Require >=+5pp paired lift per panel, zero truncation, p95 and
maximum complete step <=2s, completed-turn wall <=10s. Diagnostics 1443000–1
are separate. Eleven focused feature/planner/reader tests passed; the actual
frozen 38-feature checkpoint loaded with matching feature producers. No served
pilot changes or promotion follow from integration alone.

Outcome-value planner DEV **failed; not promoted**
(`runs/outcome-value-planner-dev-v1.json`): all 128 games complete, paired lift
-28.125pp Tactical / -34.375pp mixed, maximum complete step 0.0466s. Prediction
validation remains valid but did not transfer to this integration. Before new
model tuning, audit the development-purchase branch: the inherited planner
subtracts cost without adding a sampled card, then adds a fixed heuristic bonus.
The learned leaf can therefore charge resources without crediting the unknown
card's learned contribution. This is a concrete integration hypothesis, not a
claim that correcting it will recover the observed strength loss.

The purchase audit found a missing +7.8804 learned-score contribution under the
standard 25-card draw prior, nearly as large as the original +8 heuristic bonus.
`planner_value_draw` adds that expected learned marginal to the purchase bonus;
all other behavior and the checkpoint stay fixed. The prior is explicitly an
approximation to an unknown draw, not knowledge of the depleted deck. A test
compares the bonus against explicit feature changes for all five card types.
Fresh DEV in `outcome-value-draw-dev.json` reserves 1445000–7 / 1445100–7,
same control, panels, 128 games and +5pp/completion/time gates. This tests the
integration repair; it does not reuse the failed run as favorable evidence.

Purchase-credit integration DEV also **failed; both learned-value variants parked**
(`runs/outcome-value-draw-dev-v1.json`): all 128 games complete, paired lift
-37.5pp Tactical / -21.875pp mixed, maximum complete step 0.0462s. Because the
two variants used different fresh boards, do not infer the repair's causal effect
by subtracting these aggregate lifts. Both independently failed their gates.
The frozen model's prediction result remains valid on its validation distribution;
neither integration establishes useful action ranking. Retain the corpus and
checkpoint. Further learning should test action-conditioned continuation targets
or ranking objectives rather than repeatedly rescale this state classifier on
opened gameplay DEV boards. No holdout opened or served policy changed.

Fresh v5 external validation: `planner-robber-external-v5-dev.json` reserves
1446000–3 / 1446100–3, all seats, 64 games, unchanged `planner_robber` challenger
versus repaired external v5. Tactical and builder/external/expander opponent
panels retain the >=+5pp paired-lift, zero-truncation/fault, 2s step and 10s
turn gates. This is the first fresh-board screen of v5 after its eight known
regression cases passed. Run serially on this host; no source edits during the
run, and no default promotion or held-out claim follows from a DEV pass.

Fresh external v5 DEV **passed**, `runs/planner-robber-external-v5-dev.json`:
64 complete games, zero external faults, paired win lift +0.0625 with Tactical
opponents and +0.25 with builder/external/expander opponents. The largest focal
external step was 1.1546s and completed-turn wall time 1.8627s. Across all external
seats, including mixed-panel opponents: 6,093 decisions, maximum complete step
1.5075s and accumulated per-seat turn compute 1.9572s. Each panel has
only four board clusters; its Hoeffding lower bound remains -1. This is screening
evidence, not a held-out strength claim or promotion.

Replication declared before execution: freeze `planner_robber` and external v5;
run 16 new boards per panel, all four seats, against Tactical control on
1447000–1447015 / 1447100–1447115 and external control on
1448000–1448015 / 1448100–1448115. Configs are
`planner-robber-tactical-replication-dev-v1.json` and
`planner-robber-external-replication-dev-v1.json`. Each run contains 256 games.
Require >=5pp paired lift in each panel, zero truncations/external faults,
maximum complete decision <=2s and completed-turn wall <=10s. Run serially on
Jerrys-Mac-mini.local, process cap one match, retaining completed JSONL pairs.
No policy edits or coefficient tuning during these runs. Failure parks the
strength claim; success permits planning a separately reserved held-out test.

Tactical replication **passed**: `runs/planner-robber-tactical-replication-dev-v1.json`,
256 complete games, +0.140625 paired lift with Tactical opponents and +0.078125
with builder/expander/Tactical opponents. Maximum candidate complete step
0.0376s and completed-turn wall 0.1678s. Both panels clear the predeclared +5pp
DEV bar, but their conservative lower bounds remain negative. Tactical remains
default; external replication and confirmatory held-out evaluation remain open.

Confirmatory planning diagnostic (no held-out games opened):
`runs/planner-four-panel-calibration-v1.json` uses synthetic RNG seed 1449000,
per-comparison alpha 0.0125 and 1,024 board clusters in its +5pp alternative.
Observed marginal power is 0.822 over 1,000 replicates. Null rejection rates
are 0.014, 0, 0 and 0 for the existing symmetric/sparse/rare-tail probes.
This passes the existing loose calibration screen; it does not establish exact
coverage or joint power. Four simultaneous comparisons require accounting for
both controls, not only the two panels inside each existing runner invocation.
The current runner hardcodes alpha=0.05/number_of_panels and therefore cannot yet
consume this four-comparison calibration for a two-panel invocation. Repair
that reporting/configuration boundary after the active DEV finishes, preserving
its gameplay producer and outcomes rather than rerunning unchanged games.

Planning implication: with assumed cluster SD 0.5, a normal approximation gives
1,511 boards per panel for 95% marginal power at +5pp and alpha 0.0125;
four marginal failure probabilities <=0.05 would bound joint failure by 0.20.
This is a conditional planning calculation, not a measured joint-power result
or a universal variance bound (cluster differences lie in [-1,1]). Final sizing
must use the DEV variance conservatively and validate the full joint criterion.
The confirmatory sample size remains unselected; held-out games remain unopened.

Joint calibration predeclaration: `scripts/calibrate_joint_inference.py` will
test four simultaneous lower bounds at alpha 0.0125 using 1,000 synthetic
families, RNG seed 1449001, and 1,600 clusters per alternative panel. Test
independent and perfectly shared symmetric-null samples, a sparse/rare-tail
null family, and independent +5pp alternatives with cluster SD about 0.5.
Diagnostic gates: 95% Wilson upper bound on familywise null rejection <=0.075
and 95% Wilson lower bound on all-four alternative success >=0.80. The null
tolerance is a finite-simulation diagnostic, not a proof of exact 5% coverage.
Success requires every lower bound >0; a true +5pp effect is the power target,
not an additional observed-lift >=5pp gate (which would cap power near 50%
per panel at that boundary). DEV retains its >=5pp screening gate. Preserve
the first calibration result even if it fails. This tests inference, not bots.

Joint calibration **passed its diagnostic gates**:
`runs/planner-joint-calibration-v1.json`. All-four alternative power was
0.857 (95% Wilson interval 0.8339–0.8773). Familywise null rejection was
0.056 for independent symmetric panels, 0.015 for shared panels, and 0.014
for mixed-tail panels; corresponding upper interval endpoints were 0.0721,
0.0247 and 0.0234. The independent-null estimate exceeds nominal 0.05, though
its interval contains 0.05; this is not evidence of exact coverage. Synthetic
power at SD about 0.5 does not guarantee power for the actual board distribution.

Implemented `scripts/read_confirmatory_family.py` as the stricter final reader,
without editing the active gameplay package or invalidating completed DEV.
It consumes a predeclared family plan with exact run configs and source identity,
requires both controls, reconstructs each summary from complete raw pairs, and
requires all family-adjusted lower bounds >0 as well as every original validity
and clock gate. Source/checkpoint drift, missing runs, duplicate panel seeds,
changed summaries and external faults cannot produce a family PASS. The CLI
also verifies its package dependencies match the frozen producer. Six focused
reader/calibration tests passed. No held-out family plan has been populated or
executed yet. The existing runner can retain its two-panel admission calibration;
its local PASS will be insufficient for promotion without the stricter family
readout. This supersedes the proposed package-level reporting repair above.

The final family reader also rejects nonfinite, negative and boolean timing
values before any bounds; ten focused tests pass. During the still-running
external replication, board 1448110 exposed a baseline-arm belief failure
(focal seat 0, roll phase, turn 76) and a complete external step above two
seconds. The affected panel cannot pass its predeclared zero-fault gate.
Keep the batch running to completion and preserve every outcome; do not open
held-out games on the strength of the earlier small v5 DEV pass. Reproduce the
failing observation after the batch terminates before changing package source.

External replication finished: **FAIL**, all 256 games complete,
`runs/planner-robber-external-replication-dev-v1.json`. Tactical-opponent panel
passed DEV with +0.09375 paired lift and zero faults. Mixed panel has two
fallback decisions on board 1448110 (belief failure then timeout); the reader
counts three fault conditions because the timeout also exceeds the complete-step
clock. Mixed lift and bounds are withheld, and the whole run fails. Maximum
complete external focal step was 2.0059s. Preserve the earlier small DEV pass,
but it does not override this larger reliability failure. Capture reproduction:
`scripts/capture_external_timeout.py --seed 1448110 --seat 0 --policies external
builder external expander --output runs/external-belief-regression-v6`.

Reproduction succeeded at step 300 / turn 76. Cold probes with seeds 24617871,
0 and 1 all collapsed at hidden steal event 278 (turn 70), not a development
draw. The receiver's later observed purchases require a six-card bundle, but
the backwards single-resource bounds independently relaxed each resource by
one for the same single stolen card. Earlier discards therefore retained hands
with several shortages that one steal could not repair.

External v6 adds necessary bounds for all 31 nonempty resource subsets, using
only the already observed history. Backwards hidden steals relax each subset
by at most one; public resource deltas and offers propagate subset requirements.
Only bounds stronger than existing singleton constraints are checked forward.
This retains the existing allowed-proposal mass correction and does not read
the true opponent hands. The captured history now reconstructs 512 particles
for each of the three seeds in 1.151–1.157s. Seventeen focused tests passed,
including a one-particle bundle constraint and existing privacy-related belief
boundaries. Before execution, the v6 full-game gate remains completion of every
known regression, zero faults, max complete step <=2s and turn compute <=10s.
`scripts/probe_belief_repair.py` now includes the new failing lineup plus the
eight previous scenarios; output `runs/external-belief-repair-v6`. Cold replay
success alone does not establish full-game reliability or fresh strength.

V6 full-game regression **failed**: eight of nine scenarios passed, including
1448110 (zero faults, maximum step 1.1062s). Prior timeout board 1426003 now had
24 deadline/turn-budget fallbacks, maximum step 2.0119s and turn compute 10.0045s.
Preserve `runs/external-belief-repair-v6/summary.json`; fixing consistency did
not satisfy the latency gate.

V7 retains the subset constraints but removes those implied by a smaller subset
plus singleton bounds, omits full-hand bounds already enforced by observed card
counts, and precomputes positive singleton bounds. In the older captured replay,
extra subset checks fall from 2,396 to 604. Unprofiled cold reconstruction takes
1.7289s there and 0.8929s for the new capture, both returning 512 particles.
Sixteen focused tests pass, including a simulator-truth oracle over two complete
games (diagnostic seeds 1449200–1), checking every saved prefix constraint from
all four observer seats against its generating history. Truth is used only by
that test, never by the inference implementation. Next run:
`runs/external-belief-repair-v7`, same nine cases and unchanged completion,
zero-fault, <=2s step and <=10s turn gates. Cold timings do not replace that gate.

V7 full-game regression **passed all nine cases**:
`runs/external-belief-repair-v7/summary.json`. All complete, zero faults,
maximum complete step 1.7481s and maximum per-seat turn compute 1.8164s.
The previously failing new case completes with maximum step 0.8103s.
This repairs known reliability cases; it does not establish fresh strength.

Fresh v7 DEV declared before execution:
`experiments/planner-robber-external-v7-dev.json`, unchanged `planner_robber`
versus external v7, boards 1450000–1450015 with Tactical opponents and
1450100–1450115 with builder/external/expander opponents, all seats, 256 games.
Same gates: >=5pp paired lift in each panel, zero match truncations and external
faults, maximum complete step <=2s and completed-turn wall <=10s. One match
process on Jerrys-Mac-mini.local. Do not tune the candidate on these outcomes
or open held-out games unless the complete DEV passes.

Before fresh v7 launch, the full Python suite passed: 167 tests in 57.28s.
The evaluation producer is copied to `runs/competitive-v7-source/src/catan_ai`
with exact Python-byte hashes and a link to the separately pinned external
installation. A source-identity check and a real external setup decision on
diagnostic seed 1449300 passed from that copy (0.1328s, no fallback).
Launch with `PYTHONPATH=$PWD/runs/competitive-v7-source/src CATAN_ENGINE=python`
and config `runs/competitive-v7-source/planner-robber-external-v7-dev.json`.
This isolates measured gameplay from subsequent web development; never edit
the copied producer while its evaluation is active.

Human pilot seat selection: API and browser now support all four human seats,
including bot placements before the human, seat-aware private hand/trade views,
restart reconstruction and redacted replay exports. Fifteen API/persistence
tests and eighteen browser tests passed. The browser build is isolated at
`runs/seat-ui-dist`; the previous pilot server and build remain untouched.
A frozen `runs/pilot-source-v2` serves the new pilot at localhost:8001 with
records under `runs/pilots-v2`. The older localhost:8000 pilot remains available.
These are still one-human/three-bot pilots; no participant games or human-transfer
results have been collected. The v7 evaluation runs from its separate unchanged
producer. Browser validation ran briefly at lower process priority on the same
host; timing gates still apply to every measured decision without exclusions.

Interactive pilot replay implemented in the working source: `GET /replay?step=N`
reconstructs a recorded prefix with the authoritative engine and returns only
the human seat's view. The browser's Review match screen provides a timeline,
previous/next decisions and return to the unchanged live match. No raw seed,
hidden opponent hand or private action trace is exported. Nineteen API tests
and eighteen browser tests pass, including four-seat privacy and unchanged
live state/RNG/persisted bytes after replay navigation. Build artifact:
`runs/replay-ui-dist`. Existing frozen pilot servers on 8000/8001 still serve
their prior builds; this replay enhancement has not replaced those sessions.
No human games are inferred from these automated tests.

Current-challenger native compatibility: extended the complete-trajectory
differential test to `planner_robber` / Tactical / builder / expander. The Python
and strict Cython backends produce identical action/legal-action digests, all
four observer public states at every step, and engine RNG states through two
complete seeded games (1400002–3). Focused test passed in 14.54s. This is
correctness/seeded-behavior evidence for the actual challenger, not a new speed
or strength measurement. The previous native-v4 19.5694% timing reduction and
failed 20% performance threshold remain unchanged.

The idle localhost:8001 pilot was upgraded to frozen `runs/pilot-source-v3`
and `runs/replay-ui-dist` after confirming zero live games and zero stored pilot
records there. The process shut down cleanly; the replacement serves the replay
step API and Review match UI, verified by HTTP. Records remain at `runs/pilots-v2`.
The original localhost:8000 pilot remains untouched. No human pilot has been
recorded as of this upgrade.

Fresh v7 DEV **failed and stopped early** after 90 complete pairs (180 games),
preserved in `runs/planner-robber-external-v7-dev.jsonl`; no final summary file
or strength claim. Board 1450103, baseline focal seat 0, failed belief
reconstruction at discard turn 71. The first failure already invalidated the
zero-fault gate, so the remaining costly games were stopped to prioritize repair;
this supersedes the earlier intention to finish every game. The in-flight pair
was not published. Frozen producer remains unchanged.

`runs/external-belief-regression-v8` reproduces the failure at step 301. Cold
seed 6 collapses at hidden steal event 287 (turn 66); seeds 0/1/4/5 succeed,
while 2/3/7 need retries, some exceeding budget. The recipient needs grain for
a later observed settlement, but sampled victims have discarded it earlier.
The remaining gap is a cross-player constraint: a card transferred by a steal
must have been held by either its recipient or its victim beforehand.

An initial conservative forward per-resource upper-bound addition is implemented
and passes the real-history oracle and a focused forced-transfer test (17 tests
total). It is insufficient for this capture: seed 6 still collapses; other seeds
still retry. Do not label it a repair or launch another fresh screen. Next step
is retrospective resource bounds over groups of players to preserve transfer
conservation across earlier hidden discards, followed by captured-history timing
and the full known regression suite. No held-out games were opened.

External v8 implements necessary per-resource bounds over all nonempty player
groups. A hidden steal within a group conserves its total; a steal into the
group can add at most one card. Public resource deltas and the existing
singleton requirements propagate backwards. Forward hidden-action proposals
must satisfy the resulting group bounds, with the existing allowed-mass weight
correction. Dominated bounds are omitted. No true opponent hands are input.

The v8 capture now reconstructs all 512 particles for seeds 0–7, each in
0.918–0.929s, with no retries causing the previous failures. The older deadline
capture reconstructs in 1.8443s. Twenty focused tests pass, including a one-particle
uncertain-recipient transfer example and the complete-trajectory oracle for
group-bound soundness. These are bounded diagnostics, not universal reliability.
Before execution: `runs/external-belief-repair-v8` evaluates the new 1450103
lineup plus the nine previous full-game regressions. Require every game complete,
zero external faults, maximum complete step <=2s and turn compute <=10s.

V8 full-game regression passed all ten cases with zero faults. Maximum complete
step was 1.9525s and per-seat turn compute 2.0360s; the narrow step margin warrants
further reliability work before another expensive fresh screen.

Bounded next diagnostic: model hidden resource discards/steals as finite integer
choices constrained by the entire already-observed history. Use isolated pinned
`z3-solver==4.15.4.0` in `runs/constraint-probe-env`; do not alter the external
installation or served policy. Primary references: [Z3 Python guide](https://microsoft.github.io/z3guide/programming/Z3%20Python/Introduction/)
and [solver parameters](https://microsoft.github.io/z3guide/programming/Parameters/)
describe model construction and timeout controls. A satisfying assignment is a
consistent witness, not a calibrated posterior sample.

Before execution, the resource-only probe bar is: every preserved failure
observation has a witness whose independently replayed resource transitions,
counts, bank constraints and final own hand agree; encoding plus solve plus
verification <=1s per capture. Solver timeout 500ms. Preserve any failure.
Development-card inference and policy integration are outside this diagnostic;
it cannot establish a playable comparator or authorize promotion by itself.

Resource solver diagnostic **passed all six preserved captures**:
`runs/resource-solver-dev-v1/summary.json`, 0.0757–0.4811s per capture. Each
witness independently replays every resource transition, public production/bank
shortage, offer feasibility, counts, Monopoly exhaustion and final own hand.
The encoding cache was then corrected to use stable Z3 AST identifiers instead
of Python wrapper identities, which can be reused. Revalidation under the same
gate is preserved separately in `runs/resource-solver-dev-v2`; no original
result was overwritten. This is a resource-only witness diagnostic; development
card ages/deck counts/winner constraints and policy integration are still needed.
The solver returns one feasible history, not posterior probabilities.

Full-history solver integration diagnostic declared before its recorded run:
extend resource witnesses with hidden development draws constrained by the
25-card supply, known draws, eligible purchase ages, played identities, observed
on-turn victory/non-victory and current development counts. Replay every inferred
event through the existing `BeliefTracker._advance`, check own exact resource
and development hands, and reconstruct a world whose legal actions equal the
observation. Bar: all six preserved captures pass within 1s including encoding,
both solver calls, verification and world reconstruction; each solver check has
a 500ms timeout. Output `runs/full-solver-dev-v1`. A feasible witness is still
not posterior sampling, and the external policy is not yet using this path.

Full-history diagnostic **passed all six captures**, including exact observed
legal-action equality after world reconstruction (`runs/full-solver-dev-v1/summary.json`).
Maximum total time was 0.5694s. Four focused solver tests pass for development
card age, supply, known identities, winner constraints, immutable observations
and missing/extra witness choices. The optional dependency group `solver` pins
Z3 4.15.4.0. Next integration must avoid first spending the whole deadline on
three failed particle replays: retain an incremental particle attempt, then use
the independently verified solver witness on exhaustion and label this adapted
belief behavior explicitly. Full-game reliability and fresh strength remain open.

External v9 integrates a separate `RecoveryBeliefTracker`: one incremental
particle attempt, then a verified constraint witness if exhausted; cold histories
longer than 32 events go directly to the solver. Existing guided/MC trackers
retain their three-attempt behavior. Recovered particles are independent copies
of a feasible witness, not fresh posterior samples. Parent hard deadlines still
apply to the entire request. Worker identity now pins Z3 4.15.4.0 and labels
`depth2-guided-solver-v9`; successful decisions record cumulative solver attempts,
recoveries and elapsed solver time. The separate pinned external install gained
this dependency without modifying upstream source or prior dependency versions.
Before execution, `runs/external-belief-repair-v9` uses the same ten known
regression cases, with all-complete, zero-fault, <=2s complete-step and <=10s
turn-compute gates. A solver recovery is not an action fallback; solver or
verification failure remains a visible fault and cannot count as a valid game.

V9 full-game regression **passed all ten scenarios**, all complete, zero faults:
`runs/external-belief-repair-v9/summary.json`. Maximum complete external step
0.8615s; maximum turn compute 1.5005s. The older deadline case now takes at most
0.7030s per decision. The newest failure case exercised 11 solver recoveries
and completed without fallback, demonstrating that recovery actually runs in
the producer-to-policy path. Twenty-two focused belief/solver/deadline tests
passed before this run. Fresh-board validation is still required; no earlier
v7 strength result is relabeled as v9 evidence.

Fresh v9 DEV predeclaration: `planner-robber-external-v9-dev.json`, unchanged
`planner_robber` versus external v9. Sixteen fresh boards per panel, all seats:
1454000–1454015 with Tactical opponents; 1454100–1454115 with
builder/external/expander opponents. 256 games total. Require >=5pp paired
win lift in each panel, zero match truncations/external faults, maximum complete
step <=2s, completed-turn wall <=10s (plus external per-seat compute <=10s).
One match at a time on Jerrys-Mac-mini.local; frozen package/config at
`runs/competitive-v9-source`. A detected reliability failure may stop remaining
games, preserving completed JSONL pairs and withholding strength claims; no
stopping on favorable win rates. No holdouts are opened by this DEV launch.

Prelaunch validation passed: 188 tests with `uv run --group solver pytest -q`
in 72.08s. Frozen Python bytes match the tested source; a real external setup
decision on diagnostic board 1453900 passed from the copied package with the
expected v9 identity and pinned solver version. Launch uses
`PYTHONPATH=$PWD/runs/competitive-v9-source/src CATAN_ENGINE=python`.

Fresh v9 DEV failed reliability and was stopped with **122 complete pairs / 244
games** preserved in `runs/planner-robber-external-v9-dev.jsonl`; no final result
JSON or whole-run strength claim. The candidate arm on board 1454111, focal seat
2, failed in external seat 1 at turn 125: resource solver `unknown: timeout`,
0.8641s full external call, counted as fallback despite staying within 2s.
The last completed pair is board 1454114 seat 1. SIGINT ended the producer with
exit 130; the in-flight pair was not published. Frozen-v9 reproduction with
lineup builder/external/planner_robber/expander captured the same failure at
step 544, turn 125, in `runs/external-solver-regression-v10` (0.8819s).

Next cheap DEV probe, declared before execution: `probe_solver_tactics.py`
compares unchanged default, QF_LIA and QF_FD resource solvers using identical
constraints on all seven preserved captures at seeds 0 and 24719889 (the failing
policy seed). Retain the 500ms solver timeout. A variant must pass every case's
independent event replay and exact root-legal-actions check within 1s end-to-end
to qualify for full-game regression. No posterior-sampling claim, no increase
to the 2s decision budget, and no promotion based on these diagnostic cases.
Capacity execution waits until the reliability repair passes full-game tests.
`runs/solver-tactics-dev-v1/summary.json`: default and QF_LIA passed all 14 cold
cases, maxima 0.5872s and 0.6595s respectively. QF_FD failed 12/14 and is parked.
The default's cold passes despite its repeatable full-game timeout establish
that these snapshots alone are insufficient timing evidence. Candidate v10
changes only the resource solver to QF_LIA with the same 500ms limit and
constraints; development solving and independent witness verification remain
unchanged. Before fresh boards, require all eleven full-game regressions,
including 1454111 builder/external/planner_robber/expander, to finish without
fallbacks/truncations, full steps <=2s and external turn compute <=10s. Identity
is versioned v10 so the changed comparator cannot be confused with v9.
Full-game v10 regression result: **PASS**, all eleven games complete with zero
fallbacks, maximum full decision 0.871522s and maximum external turn compute
1.523023s (`runs/external-belief-repair-v10/summary.json`). The newly failing
1454111 trajectory now completes with max full decision 0.817183s. These are
diagnostic regression results, not fresh strength evidence. Next: full suite,
capacity probe using the v10 producer, then a fresh predeclared DEV comparison.
Full suite passed: 202 tests in 73.53s, including native differential checks.
The tested Python package and capacity script are copied to
`runs/competitive-v10-source`; first-consumption hashes match. Capacity probe
launched on `Jerrys-Mac-mini.local` with that snapshot and `CATAN_ENGINE=python`,
output `runs/match-capacity-v10`. No other match evaluation runs concurrently.
Capacity v10 **FAIL** at serial cap 1: eight completed games in 64.7274s, one
fallback on board 1455001, lineup builder/external/expander/planner_robber,
external seat 1, turn 121: resource-solver timeout at 0.8525s. No truncations;
the parent two-second limit was not exceeded. Caps 2 and 4 were not launched.
Preserve all outputs. This fresh failure supersedes the eleven known-case
passes as evidence for proceeding: no new strength screen is admitted yet.
Frozen v10 reproduction captured the same board/turn failure at step 519 in
`runs/external-solver-regression-v11` (0.8459s). Next diagnostic preregistration:
`probe_solver_tactics.py --one-hot` replaces each five-valued hidden resource
integer with five Boolean choices constrained to exactly one. Compare default
and QF_LIA on all eight captures with seeds 0, 24719889 and 24735019, retaining
the 500ms internal limit. Require all 24 cases per variant to pass independent
replay/root legality in <=1s before full-game tests. Source transformation is
confined to the probe; the production encoding is not yet changed.
Boolean diagnostic result: default solver passed all 24 cases, max 0.514938s
(`runs/solver-onehot-dev-v1/summary.json`). Boolean QF_LIA failed and is parked;
some calls exceeded its requested internal timeout, reinforcing the need for
the unchanged parent process deadline. Candidate v11 adopts only default Z3
with exactly-one Boolean resource choices; resource flow constraints and witness
verification are unchanged. Twelve focused tests pass, including all five
resource identities and rejection of creating a second card from one steal.
Before another fresh screen, require all twelve full-game regressions, now
including board 1455001, to pass zero faults/truncations and 2s/10s budgets.
The tactic probe now reads the frozen v10 integer producer by explicit
`--resource-source` (defaulting to that preserved snapshot), so its comparison
remains reproducible after production switches encoding.
Conditional fresh DEV preregistration: `planner-robber-external-v11-dev.json`
reserves 1458000–15 against Tactical opponents and 1458100–15 against
builder/external/expander, all four focal seats, 256 games. Candidate remains
`planner_robber`; control is the versioned v11 external comparator. Same +5pp
paired lift in each panel, zero truncations/external faults, p95 and every full
decision <=2s, completed turn wall and external turn compute <=10s. Launch only
after the v11 full-game regressions and validation pass. Use the existing serial
runner (cap 1); a capacity result alone does not change this declaration. Stop
remaining work on a known reliability failure while preserving completed pairs;
never stop early for favorable win rates. No held-out seeds are opened.
v11 full-game regression result: **PASS**, all twelve matches complete, zero
fallbacks/truncations, max full decision 0.871347s and max external turn compute
1.518401s (`runs/external-belief-repair-v11/summary.json`). Both recently failing
trajectories now pass, but this remains regression rather than fresh strength
evidence. The full test suite is running before the capacity retry.
Updated full suite: **207 passed in 73.53s**. The matching package, capacity
script and DEV config are frozen in `runs/competitive-v11-source`; package
hashes were verified at first consumption. Capacity retry uses that snapshot,
Python backend and the same named host/caps/cases, output `runs/match-capacity-v11`.
These capacity boards are now regression boards, not fresh strength evidence.
v11 capacity **PASS** (`runs/match-capacity-v11/summary.json`): eight identical
cases took 64.4184s at cap 1, 38.1465s at cap 2 (1.6887x), and 21.7406s at cap 4
(2.9630x). All 24 games finished with zero faults/truncations. Maximum full
decision across waves was 0.798210s; maximum completed turn wall was 1.331573s.
Winners, turns and final points matched serial in every parallel case; this is
not a claim of full action-trace identity. This small diagnostic admits further
parallel-runner development, not a change to the declared cap of an evaluation.
Fresh v11 DEV launches from the frozen package/config at cap 1, output
`runs/planner-robber-external-v11-dev.json`, under the gate and stop rule above.
Parallel evaluation implementation (not yet admitted for experiments):
`scripts/parallel_search_eval.py` launches at most 1/2/4 serial evaluator
processes. Each shard holds one board from every original opponent panel and
all four focal seats. Child commands retain stage, DEV-result and calibration
arguments, so the existing evaluator performs admission before opening any
games. The parent validates each shard's config/source/host/checkpoint, serial
process cap, complete pairs and recomputed summary, then recomputes inference
from all raw pairs at the original sample size. It never averages shard bounds
or uses small-shard strength gates as the final decision.
The output-directory lock prevents duplicate dispatch. Resume verifies the
original config, source, host, cap, runner and admission inputs; complete shards
are retained, and the child evaluator resumes its incomplete pair journal.
On a child error, already-active shards finish and persist; no additional
shards are dispatched. Twelve focused tests pass, including a simulated child
failure/restart preserving completed bytes. These tests are not a throughput
or end-to-end gameplay result.
Required next smoke test, after the current serial DEV run: two fresh diagnostic
boards per panel (1459000–1 / 1459100–1), `planner_robber` versus Tactical with
Tactical and builder/Tactical/expander opponents, all seats, at caps 1 and 2.
Require complete identical seeded winners/points/turns, zero truncations/faults,
2s/10s timing gates, and agreement of the final raw-pair aggregate with the
existing summary function. These boards are diagnostics, not promotion evidence.
No held-out or parallel strength run is admitted by the unit tests alone.
Parallel runner admission validation now includes real subprocess CLI checks:
missing DEV evidence, failed DEV, source mismatch and DEV/held-out seed overlap
each reject in both child processes before any pair journal is created. No
games are played by these rejection tests. All 16 focused runner checks pass
in 0.41s. The actual seeded gameplay comparison remains pending the active DEV
run's terminal state.
The actual-game smoke path is now implemented as `probe_parallel_runner.py`
with `experiments/parallel-runner-smoke.json`: 32 games per cap, 64 total. The
config uses stage `diagnostic`, not DEV; its permissive lift gate measures no
strength. A fifth real child-admission case proves a diagnostic result cannot
serve as passing DEV evidence for held-out admission. All 17 runner checks pass
in 0.48s. The probe compares seeded winners/points/turns, source identity, raw
aggregate and all-seat 2s/10s timings. Execution remains pending the active run.
Fresh v11 DEV **FAIL; no held-out admission**:
`runs/planner-robber-external-v11-dev.json` contains all 256 completed games.
Paired lift versus the external comparator is +0.03125 with Tactical opponents
and +0.140625 in the mixed panel; the Tactical-opponent panel misses the
predeclared +0.05 bar. Preserve this result and do not rerun the unchanged
candidate merely to seek a passing sample. The exact frozen source and raw
summary were revalidated. All 23,988 external decisions used one package
identity, with zero faults/truncations; maximum full decision across all seats
was 1.031380s and maximum completed turn wall was 1.785660s. This establishes
fresh reliability evidence for the adapted v11 comparator on this batch, not
challenger promotion.

Parallel runner gameplay smoke **PASS**, `runs/parallel-runner-smoke-v1`:
64 complete diagnostic games, identical seeded winners/points/turns and source
across caps 1 and 2, and recomputed raw aggregates agree. Cap 1 took 9.7908s and
cap 2 took 5.6445s (1.7346x throughput). All-seat maximum full decision was
0.035450s and completed turn wall 0.102225s; no faults/truncations. These
diagnostic-stage results cannot admit held-out evaluation. Together with the
external-agent capacity probe, this supports a declared parallel cap for a
future experiment; no active experiment's cap was changed.
Next cheap search-allocation diagnostic, declared before execution:
`profile_planner_budget.py` runs boards 1460000–3, all four focal seats, against
three Tactical opponents (16 games). Instrument only economic-value call counts
and root eligible-action counts. Every focal action and subsequent policy RNG
state must match an uninstrumented planner initialized from the same pre-decision
RNG state. Require zero truncations. Investigate a budget-allocation candidate
only if at least 10% of decisions with eligible search actions exhaust the
128-node limit. Report root omissions separately. This measures a mechanism,
not a gameplay improvement, and does not supersede the failed chain-search DEV.
Budget diagnostic **FAIL; allocation lane parked**:
`runs/planner-budget-profile-v2/summary.json` reports 16 complete games, 463
decisions with eligible actions, 32 saturated searches (6.9114%, below 10%),
and zero omitted root actions. Instrumented actions and RNG matched the original
at every focal decision. v1 stopped before its first decision due to an incorrect
test-harness RNG constructor; its empty result directory is preserved.

Next isolated policy hypothesis: `planner_dev_sequence` permits a terminal
development-card purchase below the root of the existing three-step/128-node
search. Existing planner search excludes these continuations, and its public
projection deliberately has no deck. The candidate derives deck availability
from the public count of observed purchases (25-card initial deck); it enables
only pay-and-score terminal leaves, never samples or exposes a hidden card.
It requests full public history. Opening, evaluator weights, purchase value,
robber policy, root acquisition restriction and search limits remain unchanged.
Seven focused planner/chain checks pass, including a concrete surplus-brick
trade into the missing grain followed by a purchase, exhausted-deck handling,
and hidden-card invariance.
DEV preregistration: `development-sequence-dev.json`, boards 1461000–7 /
1461100–7, all seats, 128 games against unchanged `planner_robber`, Tactical
and builder/Tactical/expander opponents. Require >=+5pp paired lift in both
panels, zero truncations/faults and the existing 2s/10s gates. Named host is
Jerrys-Mac-mini.local, match-process cap 2 through the validated parallel runner.
Only a passing candidate proceeds to fresh comparisons with Tactical and the
external comparator; no held-out boards are opened by this diagnostic DEV.
Development-sequence DEV **FAIL; parked**:
`runs/development-sequence-dev-v1.json` contains 128 complete games, paired
lift +0.0625 with Tactical opponents and -0.03125 in the mixed panel, versus
unchanged `planner_robber`. The mixed panel misses the +0.05 bar. Zero
truncations/faults; focal candidate max full decision 0.037685s, max completed
turn wall 0.150199s. Do not promote, tune on these boards, or run this unchanged
variant against the external comparator seeking a more favorable result.
The continuation is mechanically available, but its fixed-value evaluation
has not demonstrated an overall gameplay gain. Tactical remains the default.
Card-timing hypothesis and DEV preregistration: inherited pre-roll handling
chooses uniformly among development *actions*, including many Year-of-Plenty
resource pairs, before the economic planner sees the turn's production.
`planner_card_timing` plays an available knight before rolling (moving the
robber and advancing Army); otherwise it rolls and defers the other development
cards to the unchanged post-roll planner. All other phases and evaluation are
unchanged. This is a timing hypothesis, not a claim that delaying every card is
always optimal (for example, a pre-roll Road Building victory can matter).
Ten focused checks pass, covering retained economic cards, knight Army capture,
unchanged post-roll actions/RNG, existing planner behavior and privacy.
`card-timing-dev.json` reserves 1462000–7 / 1462100–7, all seats, 128 games
against `planner_robber` with Tactical and builder/Tactical/expander opponents.
Require >=+5pp lift in both panels, zero faults/truncations and 2s/10s limits.
Named host Jerrys-Mac-mini.local, cap 2 using the validated parallel runner.
Only a pass admits fresh required-comparator DEV; no held-out access.
Card-timing DEV **FAIL; parked**, `runs/card-timing-dev-v1.json`: 128 complete
games, paired lift -0.03125 with Tactical opponents and 0.0 mixed versus
`planner_robber`. Zero faults/truncations, candidate maximum full decision
0.032930s and completed turn wall 0.201105s. Preserve the variant and results;
do not promote or tune this timing rule on these boards. It did not meet the
predeclared improvement bar. Next research direction is paired continuation
targets for action-value learning, addressing the earlier outcome classifier's
failure to yield useful counterfactual action rankings; its implementation,
compute benchmark and DEV bar still need to be specified before execution.
Paired continuation producer and initial benchmark preregistration:
`paired_value.py` builds 143 public action features: existing action features,
projected state-feature changes, public/own-known progress interactions, and
the existing immediate economic score. The shortlist contains the unchanged
planner action plus up to three alternatives, preferring distinct action types.
It covers legal builds, bank trades, development purchases and ending the turn;
other actions retain the baseline and are outside this first model's scope.
Purchases use a pay-only feature projection, never their actual hidden draw.

`collect_paired_values.py` collects complete-game continuation targets from
simulator states. The simulator's true hidden hands and remaining deck
composition are used only for offline labels, not feature generation or any
serving policy. Each action shares initial continuation RNG seeds and a shuffled
remaining deck with its alternatives; later random draws can diverge as actions
change RNG consumption. This reduces some variance but does not promise equal
future dice sequences or an attainable perfect-information policy. Rollout bots
receive only their own observations. The focal continuation uses planner_robber;
opponent lineups alternate planner self-play and the existing mixed styles.
Training on these labels must still demonstrate gains in real observation-only
matchups, not merely agreement with the simulator labels.

Initial exact-path benchmark: boards 1463000–1, up to four positions per board,
rotating target seats 0–3 at turn thresholds 8,28,48,68, four paired replicates,
up to four actions per position, named host Jerrys-Mac-mini.local, process cap 1.
Require at least six labeled positions, zero truncated continuations, finite
143-dimensional features, and <=30s per position before sizing a larger run.
Complete board shards are resumable and source-pinned; this benchmark corpus is
diagnostic and will not be reused as model validation. Two focused checks pass
for hidden-card invariance, unchanged input state and repeated terminal labels.
Before training, reserve disjoint train/validation/test boards and freeze the
fit recipe and prediction/decision-ranking bars. No training or model promotion
is authorized by a capacity result alone.
Exact-path benchmark **PASS**, `runs/paired-value-capacity-v1/summary.json`:
8 positions, 120 complete continuations, zero truncations, finite 143-element
features. Total 27.2679s, mean 3.4085s/position, maximum 9.0845s. Serial projection
for 512 positions is 1745s, an estimate rather than a measured larger run.

Paired-value v1 learning trial, frozen before collecting/reading validation:
train boards 1464000–63 (up to 256 positions), validation 1465000–15 (up to 64),
test 1466000–15 reserved and unopened. Four states/board, four replicates/action,
same producer and 143 features. Run the train and validation collectors as two
named local processes on Jerrys-Mac-mini.local, combined cap 2; each writes
independently resumable board shards. No other match workloads run concurrently.
At observed serial rates these collections total about 18 minutes of work;
parallel speed and longer-position tails remain to be measured.

Trainer implementation is ready in `scripts/train_paired_value.py`. Fifteen
focused tests pass across the producer and trainer: closed-form ridge agreement,
zero columns, equal state weights, margin/gate behavior, invalid label rejection,
observation privacy, and an actual CLI fit with no validation directory present.
The CLI refuses checkpoint replacement and validation with an incorrect checkpoint
hash. Validation collection exited successfully; its outcomes remain unread.
Training session 59452 was confirmed live at board 1464022. No checkpoint has
been fitted from the incomplete training corpus. Keep package and collector
source fixed until this process completes; then fit once and validate the frozen
checkpoint against the existing predeclared bars.
The trainer's synthetic CLI test also exercises a successful 64-position
validation and rejects a repeated validation without replacing the original
readout. Serving integration, conditional on passing the real validation, must
register this distinct checkpoint family in `search_eval.model_identity`, pin
the feature-producing dependencies, preserve baseline behavior outside the
supported action set, and include feature construction in full-decision timing.

Conditional gameplay DEV is preregistered in `paired-value-dev.json`, before
reading model validation: candidate `planner_paired_value` versus unchanged
`planner_robber`, eight fresh boards per panel (1467000–7 and 1467100–7), all
four focal seats, 128 games total. Tactical opponents and builder/Tactical/
expander opponents use identical lineups across paired arms. Require >=+5pp
paired win lift in each panel, zero truncations/faults, p95 and every complete
decision <=2s, and completed turn wall <=10s. Run only after the fixed model
passes its validation gate and serving integration passes privacy, checkpoint
identity and baseline-fallback tests. This screen isolates the learned action
selection increment; passing it does not replace fresh Tactical and external
comparator DEV or the final held-out family. Named host Jerrys-Mac-mini.local,
process cap 2, after collection has finished.

Paired-value v1 validation **FAIL; parked**. Both collectors exited successfully.
The fixed trainer fitted 241 positions from all 64 training boards, froze
checkpoint `3d781ad5e7ad01a1aaefe35281eafdea468d0c816e862270b9a72dcc8fa47700`,
then read 58 validation positions from all 16 validation boards. Pairwise MSE
was 0.0340336809 versus 0.0308908046 for zero advantage, a **10.1741% worsening**
instead of the required >=5% reduction. The fixed 0.05 margin overrode 9/58
actions (15.5172%); observed selected-action advantage averaged 0.00431034
(+0.4310pp), below the required +2pp. All input shards passed the strict outcome,
feature and identity checks. Checkpoint and all 80 consumed shard hashes were
verified against the frozen fit/readout records. Preserve
`runs/paired-value-v1/model.json`, `validation.json` and both data directories.
Do not run the conditional gameplay DEV, tune this recipe on these validation
boards, or open reserved test boards 1466000–15. This is a failed offline
action-ranking experiment, not a measured gameplay regression. Tactical remains
default. A future learning lane needs a new hypothesis and fresh validation;
training-only diagnostics may investigate label noise and representation limits
without rebuilding or discarding the completed corpus.

Next cheap diagnostic, using only the fixed training corpus: split each action's
four paired continuation differences into replicates (0,1) and (2,3), and measure
equal-state-weighted covariance/correlation across alternatives. Report sampling
noise estimated from within-action replicate variance and aggregate directions
by action type. Predeclare a minimum split-half correlation of 0.10 and positive
covariance before treating these four-replicate labels as a useful target for a
more complex fit. Failure motivates a label-quality intervention or another
strength lane, not model tuning against opened validation. This is a diagnostic
bar, not gameplay evidence or permission to open the reserved test block.

Training-label reliability diagnostic **FAIL**:
`runs/paired-value-v1/training-label-reliability.json`, 241 positions and 685
alternative comparisons. Equal-state-weighted split-half correlation is
0.00410861 (bar >=0.10), covariance 0.000351999. Estimated noise variance of a
four-replicate mean is 0.04217093 versus mean squared target 0.04309302;
58.5754% of weighted targets are zero. This supports prioritizing variance
reduction or more reliable labels over a more complex fit, without proving
the existing representation sufficient. Comparisons share states and baseline
rollouts; these are descriptive estimates, not independent-sample inference.
The reader verifies training bytes against the frozen checkpoint and never
opens validation/test data. Synthetic positive, negative and degenerate
split-half cases verify its correlation, variance and gate calculations.
Next candidate intervention is event-specific continuation random streams so
different action paths do not shift every later dice draw. It needs a paired
reliability/capacity test before any replacement corpus or model fit; retain
all existing data and do not infer a gameplay gain from reduced variance alone.

Stream-separation diagnostic preregistration: reconstruct the first eight
training boards 1464000–7 at their saved decision steps, verify identical
observation features and actions, and reuse their original four-replicate
labels as control. New offline continuations use independent dice and theft
streams, preserving the shuffled deck and policy seeds. Each stream remains
sequential within its event type; this does not align different numbers of
thefts or development draws. Require >=24 positions, zero truncations,
split-half correlation >=0.10, estimated four-replicate noise <=80% of the
same-position control, and <=30 seconds per position. Host
Jerrys-Mac-mini.local, cap 1. These are training diagnostics, not held-out
evidence. Preserve completed per-board diagnostic shards. No engine or serving
RNG changes are part of this experiment.

Stream-separation diagnostic **FAIL; parked**:
`runs/continuation-streams-dev-v1/summary.json`. All eight boards reconstructed
exactly, yielding 30 positions and 82 alternative comparisons. Split-half
correlation was -0.0553600 versus control 0.0886236; estimated noise ratio was
1.07142857 (7.14% higher, required <=0.80). Maximum position runtime was
16.2580s. Every continuation completed with internally consistent winner/win
labels. Four focused tests passed, including theft/dice independence, copied
stream state and unchanged source game. This limited diagnostic does not prove
stream separation can never help, but it fails the predeclared bar for this
intervention. Do not generate a replacement corpus or change serving RNG on
this evidence. Preserve the original training/model and these diagnostic shards.

Next tactical hypothesis: an opt-in finish-only search may repair multi-trade
wins without the broad economic changes of the failed ChainPlanner. At >=7
known own points, search at most six deterministic actions / 768 nodes, using
conservative public bank bounds and legal build/YOP/road-building/knight moves.
Override planner_robber only when a sequence proves an own-turn win; otherwise
preserve its action and RNG behavior. Exclude hidden draws, monopoly proceeds,
and opponent-dependent negotiations. Fresh DEV: 1468000–7 versus three Tactical
opponents and 1468100–7 versus builder/Tactical/expander, all seats, 128 games,
control planner_robber. Bar >=+5pp paired lift in each panel, no truncations or
faults, every full decision <=2s and completed turn wall <=10s. Local host
Jerrys-Mac-mini.local, cap 2. Require targeted win, bank-bound and observation
privacy checks before running; no held-out admission from fixtures alone.

Finish-only search DEV **FAIL; parked**, `runs/finish-dev-v1.json`: 128 complete
games, +0.03125 paired lift in each panel, below the required +0.05 in each.
Each panel gained one net win among 32 pairs. Zero truncations/faults; candidate
maximum full step 0.187612s and maximum completed turn wall 0.858220s. Raw
outcomes recompute the stored summary and current package source matches the run.
Six focused tests pass. The concrete multi-trade fixture demonstrates a missed
win: unchanged planner_robber trades brick for lumber, builds a road and ends;
planner_finish trades for grain and ore then builds a winning city. Initial DFS
ordering exhausted its bounded budget on unhelpful trades; ordering acquisitions
by missing build resources repaired that fixture before DEV, without raising
the fixed budget. This is evidence of a real tactical gap but insufficient
gameplay improvement for this lane's bar. Preserve the opt-in implementation;
do not promote, rerun unchanged hoping to pass, or open held-out boards.

Private three-human/one-bot pilot path implemented in `rooms.py` and `Room.tsx`.
Room creation supplies per-seat capability invitations; storage retains only
their hashes. Each player registers recording consent and self-reported match
experience. Play waits for all three humans, then uses the authoritative engine
for actions, including trade responses. Action revisions reject stale retries;
seat authentication guards reads/actions/replay. Source-pinned private records
rebuild bot RNG through replay after restart. Replay is read-only and redacted
for the authenticated seat. Tactical remains the UI default and only served
room opponent; the API also permits the preserved Heuristic comparator.

Validation: 22 API/pilot/room tests passed, including a full deterministic
synthetic three-client match with restart and final replay equality. All 19
browser tests passed, including three independent browser contexts joining,
private action/state handling, refresh and replay, plus existing solo flows.
Synthetic records were isolated in temporary test storage and are not human
evidence. Browser production build passes. A frozen local room pilot is live
at `http://localhost:8002/#room`, source/build under
`runs/room-pilot-source-v1`, participant storage `runs/room-pilots-v1`.
Health check passes. Existing pilot servers and records are preserved. This
loopback server is for local access; remote participant hosting, blinded
candidate/control assignment, confirmatory analysis and actual experienced
human matches remain outstanding. No Colonist integration or human-transfer
strength claim is established by this implementation.

External-opening hybrid preregistration: `planner_external_opening` delegates
only setup settlement/road choices to the verified v11 Catanatron depth-2
adapter, then uses unchanged planner_robber. Preserve baseline planner RNG;
record all external identities, deadlines and fallback diagnostics, and close
the external worker after setup. This tests the external opening evaluator,
not a retry of the earlier failed economic opening-search implementation.
DEV boards 1470000–7 (three Tactical opponents), 1470100–7
(builder/Tactical/expander), all seats, 128 games against planner_robber.
Require >=+5pp paired lift in each panel, zero faults/truncations, full decision
<=2s and completed turn wall <=10s. Host Jerrys-Mac-mini.local, cap 2. A fixed
four-game setup integration smoke on board 1469900, all focal seats, must pass
before DEV. No held-outs or promotion from this screen alone.

External-opening hybrid DEV **PASS**, `runs/external-opening-dev-v1.json`:
128 complete games, +0.0625 paired lift against planner_robber with Tactical
opponents and +0.1875 mixed; zero truncations/faults, maximum candidate full
step 0.143704s and completed turn wall 0.157298s. Raw outcomes reproduce the
summary and source matches. The four-seat smoke passed; 27 focused policy,
evaluator and parallel-runner checks passed. Evaluator coverage explicitly
requires four alternating external settlement/road decisions for this hybrid,
including pinned identity and zero fallback/budget faults; subsequent economic
decisions remain covered by full-step timing. This is screening evidence only.

Freeze the unchanged hybrid for fresh required-comparator DEV, 256 games each:
`external-opening-tactical-dev.json`, boards 1471000–15 / 1471100–15 versus
Tactical with Tactical and builder/Tactical/expander opponents;
`external-opening-external-dev.json`, boards 1472000–15 / 1472100–15 versus
verified external v11 with Tactical and builder/external/expander opponents.
All focal seats; same >=+5pp per-panel lift, zero faults/truncations, full
decision <=2s and turn wall <=10s. Run on Jerrys-Mac-mini.local at cap 2,
one configuration at a time. Both must pass on the same source before sizing
and opening the reserved confirmatory family. No source or evaluator changes
during these comparisons; preserve valid completed shards.

Fresh direct Tactical DEV **FAIL; hybrid parked**,
`runs/external-opening-tactical-dev-v1.json`: 256 complete games, paired lift
-0.109375 with three Tactical opponents and +0.125 mixed. Zero faults or
truncations, maximum candidate full step 0.147244s and turn wall 0.161671s.
Current source and recomputed raw-outcome summary verified. The earlier
incremental pass versus planner_robber did not establish improvement against
Tactical across fresh opponent panels. Do not promote this hybrid or retune
on these boards. Skip the queued external DEV because a required comparator
bar already failed; its reserved boards remain unopened, as do all held-outs.
Preserve both screens and the opt-in implementation. The contrast between
homogeneous Tactical and mixed panels is a hypothesis about opponent dependence,
not proof that opening selection alone caused the fresh regression.

Component diagnostic preregistration: six combinations of opening policy
(Tactical, planner_robber, external v11) and turn policy (Tactical,
planner_robber), on common boards 1473000–7 / 1473100–7, all focal seats,
Tactical and builder/Tactical/expander opponents. 384 games, local host
Jerrys-Mac-mini.local, process cap 2, complete resumable seed/seat/panel shards.
Preserve the underlying policies' seeds and opening-independent turn RNG.
This is diagnostic selection, never DEV/held-out admission. Report win rates,
paired differences from Tactical, faults/truncations and maximum full clocks.
Only a new cross-component combination with >=+5pp versus Tactical in BOTH
panels and zero faults/truncations within 2s decision/10s turn may advance to
its own fresh DEV. Choose at most one by highest minimum panel lift (then mean
lift, then policy name); do not re-admit a previously parked unchanged policy.

Component diagnostic complete, 384 valid games, source/raw summaries verified.
Win counts out of 32 in Tactical/mixed panels: Tactical 8/11, planner_robber
12/19, Tactical-opening/planner-turn 10/9, planner-opening/Tactical-turn 11/19,
external-opening/planner-turn 10/16, external-opening/Tactical-turn 9/16.
All arms had zero faults/truncations and passed full clocks. The predeclared
selection chooses `tactical_planner_opening` (+0.09375/+0.25 against Tactical).
This supports testing the planner's setup evaluation with simpler Tactical
turn play; it does not establish general opening superiority. Thirteen focused
component/evaluator checks passed. Prior failed unchanged variants remain parked.

Fresh required-comparator DEV for this selected frozen combination:
`tactical-planner-opening-tactical-dev.json`, 1474000–15 / 1474100–15, and
`tactical-planner-opening-external-dev.json`, 1475000–15 / 1475100–15. Each
256 games, all focal seats, Tactical panel and builder/control/expander panel.
Same >=+5pp per-panel lift, zero faults/truncations, 2s full decisions and 10s
completed turns. Named local host Jerrys-Mac-mini.local, cap 2; configurations
run sequentially, skip external if Tactical fails. Both must pass on unchanged
source before any held-out admission. Diagnostic data cannot serve as DEV.

Selected component fresh DEV **FAIL; parked**,
`runs/tactical-planner-opening-tactical-dev-v1.json`: 256 complete games,
-0.0625 paired lift with three Tactical opponents and +0.046875 mixed, both
below +0.05. Zero faults/truncations; maximum candidate full step 0.033062s
and completed turn wall 0.119330s. Raw outcomes reproduce the stored summary
and source matches. Do not promote or rerun this unchanged combination to seek
a pass. Its external comparator block and held-outs remain unopened. Together
with the prior external-opening failure, this weakens the case for selecting
a static opening-policy swap from small common-board diagnostics. Preserve
the component harness and all six-arm outcomes for future training/diagnosis,
but do not call the selected diagnostic winner a general strength improvement.

Performance lane: `runs/search-profile-v5/profile.json` plus cProfile artifacts
identify repeated port scans as a hot path (trade_rate cumulative 0.683s of a
3.602s instrumented MC decision). Batch all five rates in one port scan for
legal-action and observation construction; retain the single-resource reference
method. Preserve pre-change package at `runs/port-batch-before/src`.
Cheap acceptance: three alternating before/after Python probes on fixed diagnostic
seed 1476000 must preserve all trajectory/action/RNG/decision diagnostics and
reduce summed search decision time by >=15%. Then rebuild Cython and run full
seeded backend differential checks and the existing >=20% native benchmark.
These are performance/correctness claims, not strength. No policy weights change.

Port batching v1 preserved checked semantics but **failed** its >=15% search
runtime reduction bar: `runs/port-batch-benchmark-v1/comparison.json`, measured
13.2519%. The first probe overlapped the end of a 1.22s focused test run, so do
not treat those timings as an isolated performance result. Nine focused checks
passed. Next distinct intervention: compute the immutable-in-enumeration bank
balance once when the first affordable maritime trade is found, rather than
once per candidate receive resource. Port+bank batching v2 uses the same frozen
original baseline and >=15% bar, three isolated alternating probe pairs, new
output directory. No rule, ordering, observation or RNG change is permitted.

Port+bank batching v2 **PASS**:
`runs/port-batch-benchmark-v2/comparison.json`, isolated alternating Python
before/after search totals show 15.6594% runtime reduction, with identical
checked trajectory/action/RNG/search-diagnostic outputs. Rates scan ports once
for all resources; legal-action enumeration computes bank balances once when
needed. Single-resource trade_rate remains as an independent reference.
Full suite: 262 passed. Rebuilt Cython successfully; 27 compiled-backend checks
passed, including complete Python/Cython trajectory equivalence for existing
policies and planner_robber, port subsets, observation equivalence and rule edges.

Native v5 performance **PASS**, `runs/native-v5/comparison.json`: three isolated
alternating probe pairs, Python search totals 1.419255/1.402822/1.421391s versus
Cython 1.081691/1.074062/1.074713s, **23.8720%** reduction (bar >=20%). All
checked semantic outputs matched and probe games completed. This supersedes
v4's failed performance gate for the new engine source; preserve v4 evidence.
Performance is measured on the fixed diagnostic workload, not a universal
speedup or gameplay-strength claim. No policy defaults changed.

Original source-pinned value comparators remain reproducible using
`PYTHONPATH=$PWD/runs/port-batch-before/src`; that frozen package links the
unchanged original checkpoint and external installation. Both planner_value and
planner_value_draw were constructed there with checkpoint SHA
`b55eab9866dce201c28b67b76b5e3a71ad0250854b56e35f125c9baff5b56a0f`.
Their current-package loader correctly rejects changed engine source; no model
bytes or historical source checks were weakened to accommodate this optimization.
Existing frozen pilot servers remain on their own matching source/builds.

Direct policy-weight fitting v1 preregistration: train eight economic parameters
(points, production, hand, geometry, ports, held development, purchase credit,
production diversity) by three generations of eight candidates, including an
unchanged incumbent each generation. Initial parameters reproduce planner_robber:
[12,.55,1,1,.5,1,8,2]. Log-normal proposals use fixed RNG seed 1476999 and initial
log standard deviation .5; elite three update log means/stds (std clipped
[.15,.7]). Bounds respectively [6,30],[.2,1.2],[.2,2],[.1,3],[.1,2],[.2,3],
[2,18],[.2,5]. Rank by minimum panel win rate, then average, then stable trial
index. Training boards 1477000–7 / 1477100–7, all seats, Tactical and
builder/Tactical/expander opponents, 64 games per unique candidate. Reuse exact
duplicate candidates' completed training results. At most 1536 training games.
Fit only on training outcomes; no rollout labels or opened validation retuning.

Before the full fit, benchmark the exact default-parameter 64-game candidate
path on Jerrys-Mac-mini.local at cap 2; require zero faults/truncations, full
decision <=2s, completed turn <=10s and <=120 seconds total. Preserve each
game and checkpoint. After fitting, freeze one checkpoint before fresh direct
Tactical DEV on 1478000–15 / 1478100–15, then (only if passing) external DEV on
1479000–15 / 1479100–15. Same >=+5pp lift in both panels, zero faults/truncations,
2s/10s clocks. These DEV boards remain unopened during training. No held-outs
without both required comparators passing on the frozen source/checkpoint.

Direct weight-fitting capacity **PASS**, default candidate completed all 64
training games in 11.7443s with valid outcomes and 2s/10s clocks. Training win
rates 0.40625/0.625 are fitting data only. Thirty-two focused policy/reader/
evaluation tests pass, including a complete default-weight trajectory/RNG
equivalence check and fail-closed checkpoint/source validation. Fit proceeds
with the fixed recipe, immutable per-trial checkpoints and per-game resumable
outputs under `runs/economic-parameters-v1`. No package or trainer edits while
it runs. Fresh configurations `economic-parameters-tactical-dev.json` and
`economic-parameters-external-dev.json` are reserved but not launched. Use the
frozen final `model.json` through `CATAN_PARAMETER_MODEL`; evaluator and game
records include its identity. Historical checkpoints remain untouched.

Direct policy-weight fit completed: 22 unique candidates, 1408 complete training
games, all candidate validity gates passed. Fixed selection chose trial 17,
checkpoint SHA `77c5ebfc5b148e8d9a22aa07732df2b62b428fe6b8dc3767be8276447206fc92`,
weights [16.954330814946953,0.6537987321930112,0.641167594927603,
1.5499286338795906,0.4445213636185654,1.4479747239779968,
7.063402747746514,2.805531684851287]. Training win rates .4375/.75 are fitting
outcomes only. Recomputed every trial summary from its 64 game records, verified
source/trainer/checkpoint identity, selected-trial ranking and exact final model
bytes before launching fresh Tactical DEV. Preserve all trials and final model;
do not refit based on the upcoming DEV result.

Frozen economic-parameter Tactical DEV **PASS**,
`runs/economic-parameters-tactical-dev-v1.json`: 256 complete fresh games,
paired lift +0.234375 with three Tactical opponents and +0.296875 mixed.
Zero faults/truncations; maximum candidate full step 0.036188s and completed
turn wall 0.149516s. Raw summaries, exact source and frozen checkpoint identity
verified. This is DEV screening, not a held-out strength claim. Proceed to the
already reserved external DEV without modifying weights, source or gates.

Conditional exact-candidate capacity check is ready: `benchmark_match_capacity.py`
now accepts `--candidate planner_parameter --seed 1479900`, preserves the
original default cases, pins the parameter checkpoint in its plan, and checks
each candidate game's checkpoint metadata. Thirteen capacity tests pass,
including all-seat case coverage and unchanged validity guards. Run only after
external DEV passes and its workers exit, with `CATAN_PARAMETER_MODEL` pointing
to the frozen v1 model; use `runs/economic-parameters-capacity-v1`. The same eight
diagnostic games run at caps 1, 2, 4, requiring zero faults/truncations, all full
decisions <=2s and turns <=10s, and parallel speedup >=1.25. This sizes the real
candidate/comparator path before any long confirmatory run; no held-out data
is needed for capacity or inference checks. Existing inference/joint-calibration
evidence may be reused only with matching inference code and justified variance
assumptions; the final confirmatory plan remains unpopulated pending both DEV
comparators and measured capacity.

Final-reader learned-policy boundary strengthened before any held-outs: every
game must contain model metadata for exactly the learned seats implied by its
lineup, matching the frozen family checkpoint and with no load fallback.
Run-level checkpoint identity alone is insufficient. Thirteen family-reader
tests pass, including wrong/missing per-game hashes, fallback, unexpected learned
seats, source/config/summary mismatch, timing validity and external failures.
Only the offline reader and tests changed; active DEV gameplay/source and
checkpoint remain unchanged. This closes a claim-verification gap rather than
adding a promotion criterion after looking at held-out outcomes.

Frozen economic-parameter external DEV **PASS**,
`runs/economic-parameters-external-dev-v1.json`: 256 complete fresh games,
paired lift +0.171875 versus external v11 in BOTH Tactical and mixed-external
panels. Zero faults/truncations, full timing gates passed. Both required DEV
results were recomputed from raw outcomes and checked against current source and
the same frozen checkpoint before capacity dispatch. Per-game learned model
metadata was also verified against the frozen hash. A non-default-weight privacy
check passes: invisible opponent resource/card identities and deck order cannot
change observation, selected action or policy RNG; eight parameter-policy tests
pass. These are successful DEV screens, not held-out promotion or human evidence.
Exact-candidate serial/2/4 capacity is running in
`runs/economic-parameters-capacity-v1`; held-outs remain unopened.

Fit one linear action-value model to alternative-minus-baseline feature and
win-target differences, equally weighting states and dividing each state's
weight among its alternatives. Scale columns by training weighted RMS (use 1
for zero columns), without centering, preserving antisymmetric differences.
Use ridge coefficient 0.1 on the mean squared objective, no intercept, no
hyperparameter sweep, and no fitting on validation. Freeze the checkpoint before
reading validation labels. The decision rule chooses the highest predicted
alternative only when predicted advantage is at least 0.05; otherwise retain
the baseline. Prediction/decision DEV bars: at least 48 validation positions,
>=5% reduction in weighted pairwise MSE versus predicting zero advantage,
mean observed selected-action advantage >=0.02 over all validation positions,
and overrides on at least 5% of positions; zero incomplete labels or invalid
features. Passing these bars admits only a fresh gameplay DEV, not promotion.
Test boards remain unopened until a frozen model clears validation. Preserve
the corpus/checkpoint on failure; the previous outcome-value corpus is untouched.

Capacity diagnostic preregistration (before execution):
`scripts/benchmark_match_capacity.py` runs the same eight mixed-lineup games
under match-process caps 1, 2 and 4 on the explicitly named local host. Diagnostic
boards 1455000–1 are reserved for this probe, with every focal seat and both
`planner_robber` and `external` focal arms. Each match contains at most two external
workers, so cap 4 permits four match processes plus up to eight external workers.
Use the frozen v9 Python package through `PYTHONPATH` and `CATAN_ENGINE=python`.
Do not run alongside the active v9 DEV evaluation. Serial must have zero faults
and truncations; a parallel wave must additionally achieve at least 1.25x serial
throughput. Every full step must be <=2s, completed turn wall time <=10s and
per-external-seat turn compute <=10s. Stop increasing concurrency at the first
failed wave. Source identity is checked between waves. Preserve each completed
game and wave; interrupted waves provide no throughput claim and require a new
output directory for a new attempt. Each game runs in its own process so external
workers are not retained across games. Report changes in winners/turns/points
from serial because wall-clock-bounded search can change its work under load.
This probe establishes neither strength nor permission to change an active
evaluation's process cap; a future evaluation must declare its own cap.
Twelve focused gate checks pass, including missing external identity/decision
coverage, invalid terminal outcome, faults, nonfinite/negative timing and each
budget limit. The probe uses the evaluation's 1000-turn limit. Execution is
pending the v9 DEV finish.

Pilot evidence hygiene: inspection found 98 legacy records in `runs/pilots`,
including repeated API seed 4/123 and browser-fixture seed 71. Four recorded wins
use fixture seed 71; these are not participant wins. Preserve these files but do
not count them as human evidence without participant provenance. Both pilot
servers report healthy with zero in-memory games; current `runs/pilots-v2` has
zero records. Python tests now force a per-test temporary pilot directory, and
the browser fixture server always creates its own temporary storage even if a
participant directory is inherited in the environment. A subprocess integration
test verifies actual browser-server writes avoid a sentinel participant directory
and clean up at exit. Diagnostic seeds 1456000–1 cover these storage checks.
Storage, API, pilot and capacity checks: 33 passed. This changes test isolation,
not the running pilot servers or the frozen evaluation source.

Scope clarification: starting-spot selection means the two initial settlements
and their attached roads, before ordinary turns. Rechecked the Colonist guide,
King of Catan production guide search extract, and official base-game setup rules
on 2026-10-30. The selection checklist is productive resource bundles, plausible
return picks in snake order, second-settlement starting cards, legal expansion
targets and competing roads, and ports supported by surplus production.
Evaluate a city/development plan and an expansion plan rather than require all
five resources or always choose the largest pip total. These are candidate
decision features, not evidence that changing their weights improves our agent.

Sources and evidence limits:

- [King of Catan: production and placement order](https://kingofcatan.net/initial-settlement-placements-production/),
  a community-authored expert guide: evaluate complementary placements, scarce
  resources, ports and plausible victory routes. Ore/wheat supports cities and
  development; expansion or port plans can exploit the board and opponents'
  commitments. This is expert advice, not a controlled win-rate study.
- [Colonist's starting-strategy guide](https://blog.colonist.io/guide-to-catan-starting-strategies/):
  connect resource selection to a build plan, consider number diversity, and
  direct roads toward expansion that opponents are unlikely to occupy first.
  These are strategic heuristics, not a universally optimal opening algorithm.
- [Official base rules, setup and tactics](https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf),
  printed pages 12–13: snake placement, starting cards from the second settlement,
  expansion space and productive-resource/port combinations. The rules' early
  brick/lumber advice and competitive guides' ore/wheat preference reflect
  different build plans; neither establishes fixed optimal resource weights.

Our deductions: optimize an ordered pair of settlements and their roads. Seat 4
can choose its pair consecutively; earlier seats must evaluate several plausible
intervening opponent placements, rather than assume their ideal return pick
survives. Evaluate first-city, first-expansion and port-conversion plans using
resource bottlenecks, actual starting cards, reachable legal sites and road races.
Scarcity should reflect accessible production and likely owners, not merely
terrain counts; do not assume humans will trade away a decisive scarce resource.

For four-player base-game setup the pick pairs are seat 1: (1, 8), seat 2:
(2, 7), seat 3: (3, 6), and seat 4: (4, 5). There are respectively six, four,
two, and zero opponent settlement picks between a player's settlements. This
counts picks, not distinct intersections removed: each settlement also excludes
adjacent intersections. A first pick should therefore be evaluated against the
remaining legal complementary options after those exclusions. For seat 4, compare
both orders of a candidate pair because the second site determines starting cards.
The [official base-game rulebook](https://www.catan.com/sites/default/files/2021-07/catan-25th-rules_eng-200313.pdf)
was checked again for the snake order and second-settlement resource rule.

Number diversity changes income variance, not expected production at equal pip
totals. For example, two one-card sources on 6 yield 10/36 expected cards per
roll, as do separate sources on 6 and 8. The former pays two cards with probability
5/36; the latter pays one with probability 10/36. Effects on purchase timing,
discard exposure and complementary resource bundles require measurement; a
blanket duplicate-number penalty is not justified by expectation alone.

Code audit: `EconomicPlanner.choose` adds only the proposed settlement to a
public projection and scores the resulting position. It does not search the
remaining setup sequence or credit the proposed second settlement's starting
cards. Its production aggregates by resource, losing dice-number correlation.
The setup-road fallback uses `HeuristicBot._road_score`, which values an edge
endpoint rather than a reachable settlement obeying the distance rule. The
planner's separate geometry feature is more capable but does not select setup
roads. `CoveragePlanner` only changes static settlement scoring; its failed DEV
does not test the ordered-pair/search hypothesis.

Recommended next opening-only experiment: retain `planner_robber` after setup,
first fix setup evaluation to apply real legal transitions, then add a bounded
beam over remaining placements and roads with multiple opponent placement
policies. Compare each change separately with the unchanged opening and identical
post-setup policy on fresh boards, all seats, Tactical and mixed-style panels.
Before execution reserve seed ranges and a fixed candidate configuration; DEV
bar is >=+5pp paired win lift in each panel, zero illegal actions/truncations,
every complete decision <=2s and accumulated turn compute <=10s. Also report
seat-level outcomes, first-city/third-settlement timing, and unreachable road
targets as diagnostics. Common random streams reduce noise but changed game
paths can consume RNG differently. This records the original proposal; the
opening-search and purchase-opening DEV results above supersede its execution
status. Both failed their strength bars and remain parked. A new candidate needs
a better validated evaluation target; a DEV pass must precede fresh held-out
evaluation. No claimed opening win-rate percentage or
universal best seat was established by this research.

Economic-planner DEV preregistration: `planner-dev.json` compares a new bounded
public-information economic planner with Tactical on eight fresh boards per
panel, four seat rotations, against all-Tactical and builder/expander/Tactical
lineups (64 pairs total). Pass requires at least +5 percentage points paired win
lift on each panel, no truncated games, and p95 decision time at most two seconds.
Its evaluator includes production, diminishing hand value, discard exposure,
ports and graph distance to viable expansion sites; search considers up to three
own actions with 128 expansions and an eight-state beam. Root-only resource
acquisition avoids inventing future bank supplies. Development purchase value is
a fixed hypothesis; special effects retain baseline handling. This first screen
does not establish strength against the external comparator. Seeds 1419000–1
are capacity/correctness diagnostics; 1420000–7 and 1420100–7 are DEV boards.

Planner v1 result: **FAIL; not promoted.** All 128 games completed. Paired lift
was +3.125pp on all-Tactical and +28.125pp on mixed, with decision p95 12.24ms and
11.34ms respectively (`runs/planner-dev-v1.json`). The first panel missed the
declared +5pp bar. This is useful DEV signal, not a confidence claim.

The next isolated hypothesis raises expected development-purchase value from
5.5 to 8.0, because the original value can be below the charged resource value
even when buying advances an army/hidden-point plan. Original `planner` remains
available unchanged in behavior. New `planner_development` uses fresh seeds
1421000–7 and 1421100–7 with the same gates in
`planner-development-dev.json`; no held-out data is involved.

External compatibility audit on diagnostic seed 1400090 reproduced stock-model
differences: a road ending at an enemy settlement counted zero instead of one,
and a non-acting player at ten points was immediately declared winner. Explicit
process-local shims correct these cases and filter Year-of-Plenty choices to
two cards when two remain available. Third-party source bytes remain pinned;
this is now labeled `catanatron-adapted-depth2-rules-v2`. The regression script
`verify_external_compatibility.py` passes the endpoint, on-turn victory (including
copied states), and scarce-bank card-selection witnesses. Earlier integration
results remain evidence for v1, not v2. Next v2 reliability probe: fresh seeds
1400110–1, all seats, builder/expander/Tactical opponents, same zero-fallback,
complete-game, two-second step and ten-second turn gates (eight games).

The development-purchase hypothesis **passed its DEV gate** on fresh boards:
+12.5pp versus Tactical in the all-Tactical panel and +6.25pp in mixed, all
128 games complete, policy p95 9.96ms / 7.16ms. Receipt:
`runs/planner-development-dev-v1.json`. The samples remain small and the
confidence bounds include no improvement. No default promotion or held-out
strength claim follows from this screen.

Conditional next DEV, declared before results: after the corrected external
adapter clears its reliability gate, `planner-external-dev.json` compares
`planner_development` directly with control `external`, using four fresh boards
per panel and all seats. Panels are three Tactical opponents and
builder/external/expander opponents; board ranges are 1422000–3 and 1422100–3.
Pass requires +5pp paired lift in both panels, all 64 games complete, no external
faults, and existing time gates. This is the first direct outside-control screen,
not a held-out promotion test. Preserve both successful and failed comparisons.

The corrected external adapter cleared its fresh reliability gate:
8/8 complete games, zero fallbacks, maximum complete step 0.967s and maximum
turn compute 0.967s, total runtime 44.209s
(`runs/external-reliability-v2/summary.json`). The direct outside-control DEV
screen is therefore admitted. This verifies the specified adapted comparator
on the tested integration and regression cases; it does not certify all possible
upstream game states or commercial-platform compatibility.

The direct outside-control screen subsequently **failed comparator reliability**
and was stopped after 18 completed pairs (16 Tactical-panel, two mixed-panel).
The preserved journal `runs/planner-external-dev-v1.jsonl` contains 27 external
belief-failure fallbacks across six seat/arm runs, affecting both panels. No
aggregate outside-baseline strength result is valid; no completed final-result
file was written. The interrupted in-flight pair was not treated as completed.
The supervisor exited with status 130 and its workers exited as well.

The earliest Tactical-panel failure was reproduced on seed 1422001, external
seat 1, turn 29, step 160. Its detached observation and diagnostic context are
preserved in `runs/external-belief-failure-v1/` for focused repair. The capture
contains no full real-engine hidden state. Increasing particles passed earlier
small probes but did not solve broader reliability; repair reconstruction before
rerunning affected external comparisons. Existing Tactical-only planner evidence
remains valid and unchanged.

External workers now explicitly set `PYTHONHASHSEED=0`, because upstream action
sets can otherwise change traversal/tie order between processes despite identical
game RNG seeds. Earlier records retain their original producer identity; the
next external run is a new producer and must record the fixed hash setting.

Belief diagnosis: the captured failure can exhaust particles at a public trade
on event 158, where seat 2 must give ore. Earlier hidden discard/steal proposals
had removed that ore. `GuidedBeliefTracker` computes sound per-resource lower
bounds backward through the already-observed history, uses them to constrain
hidden-action proposals, and corrects weights by allowed-card mass / total-card
mass. It does not observe future game events. Existing Monte Carlo policies keep
the original tracker; only the external worker uses this revised producer.

The captured observation reconstructed successfully with 512 particles for five
RNG seeds (0.436–0.446s cold replay). Eleven focused tests pass, including exact
8/9 proposal normalization, retrospective trade conditioning with one particle,
and rejection of an inconsistent history. Next gate, declared before execution:
`probe_belief_repair.py` runs all five previously failing board/lineup cases,
at most 300 turns. All games must complete; every external seat must have zero
fallbacks, complete step time <=2s and turn compute <=10s. Fixed hash ordering
and guided proposals make this a new producer; these are regression scenarios,
not promises of identical earlier trajectories or fresh strength evidence.

Guided-belief regression **passed all five scenarios**
(`runs/external-belief-repair-v3/summary.json`): all completed, zero external
fallbacks, maximum step 0.839s and maximum turn compute 1.437s. This repairs the
observed failure class without raising the particle count. The next direct
outside-control DEV uses new boards 1423000–3 and 1423100–3 in
`planner-external-guided-dev.json`, retaining the +5pp-per-panel and zero-fault
gates. The earlier failed journal stays intact and is not resumed under changed
source. Worker identities now explicitly record hash seed 0 and adaptation v3.

The fresh guided outside-control screen completed **64/64 games**, with zero
external faults across **6,057 external decisions**; maximum complete external
step 1.569s and maximum turn compute 1.833s. This establishes the specified
adapted comparator's tested reliability under the declared budgets. Its pinned
source, dependency checks, compatibility regressions, and observation-only input
remain part of the comparator definition, not a claim of exhaustive rules proof.

Strength result: **FAIL; planner not promoted.** Versus the repaired external
control, planner lift was -18.75pp in the all-Tactical panel and +31.25pp in the
builder/external/expander panel (16 pairs each). Candidate policy p95 was
10.25ms / 11.40ms. Both confidence intervals remain wide; no held-out strength
claim is made. Receipt: `runs/planner-external-guided-dev-v1.json`.
The candidate's earlier Tactical-only DEV pass remains valid but does not meet
the combined goal. Future candidate work must address opponent-lineup robustness
and beat both comparators before reserving a promotion holdout. Human/Colonist
transfer remains untested.

Next isolated DEV hypothesis: `planner_coverage` changes only initial settlement
ranking relative to `planner_development`. It scores total production pips plus
four pips per resource available across both placements, with reachable expansion
value as a tie-breaker. This tests whether the prior weighted/log-like opening
utility sacrificed too much production; it does not copy external policy code.
`planner-coverage-dev.json` uses fresh boards 1424000–7 and 1424100–7, all seats,
against Tactical control in all-Tactical and mixed-style panels. Pass requires
+5pp in each panel, all games complete and p95 <=2s. Existing variants remain
available for ablation. The initial screen is against Tactical, not evidence of
beating the external comparator.

Opening-coverage result: **FAIL; parked** (`runs/planner-coverage-dev-v1.json`).
All 128 games completed; +12.5pp in the Tactical panel, 0pp in mixed. No promotion.
Next isolated hypothesis, `planner_robber`, retains the development planner's
opening and turn search but replaces the inherited robber score. The old score
ignored number probabilities and own production. The new score counts settlement
and city production, penalizes blocking our own income, and scales enemy denial
by public point threat. `planner-robber-dev.json` reserves fresh boards
1425000–7 / 1425100–7 with the same +5pp-per-panel, completion and timing gates.

Production-aware robber DEV **passed**: +21.875pp against Tactical in the
all-Tactical panel and +25pp in mixed, all 128 games complete. Policy p95 was
11.22ms / 8.82ms (`runs/planner-robber-dev-v1.json`). These are fresh screening
results, not proof that the isolated change caused the entire observed lift or
held-out evidence. The next direct external-control screen is predeclared in
`planner-robber-external-dev.json`: four fresh boards per panel (1426000–3,
1426100–3), all seats, Tactical-only and builder/external/expander opponents,
+5pp required in each, no truncation/comparator faults, and existing time gates.

The production-aware robber external screen **failed comparator reliability**
(`runs/planner-robber-external-dev-v1.json`). All 64 games completed, but external
diagnostics recorded 27 timeouts, two exhausted turn-compute budgets, and one
belief-reconstruction failure. The reader withheld both panel lift estimates;
its 46 / 11 fault counts include overlapping diagnostic and timing violations,
not 57 independent fallback decisions. The first fallback occurred on board
1426003, baseline external seat 1, turn 117, in the roll phase (2.007s).
The earlier 64-game clean comparator screen remains valid for those cases but
does not establish broader reliability. Preserve this completed run and repair
the late-game deadline/reconstruction path before another external strength
claim. The candidate's Tactical DEV pass remains valid; no promotion follows.

Reviewed 2026-10-30. This section records primary sources, their limitations, and
our research decisions. Author-reported results below have not been reproduced
here. Our local measurements are recorded separately in the experiment readouts.
The executable evaluation requirements live in [competitive-plan.json](competitive-plan.json).

### Ambition means an externally tested outcome

Felix Rieseberg's [essay on ambition](https://felixrieseberg.com/how-to-be-more-ambitious-a-tutorial-for-humans-with-models/)
distinguishes speed, altitude, depth, and range. Our application of that framework
is to aim for a competitive player: beat Tactical and an independently developed
search agent under the same clock, then measure transfer to experienced humans.
Adding Cython, MCTS, or a neural model is an implementation milestone, not that
outcome. Expert methods here include imperfect-information modeling, controlled
seat-rotated experiments, calibrated uncertainty, and independent opponents.
Human play also tests trading and strategic behavior that a shared bot population
can miss. These are our project decisions, not Catan results claimed by the essay.

### Catanatron: inspectable heuristic search comparator

Primary sources: [repository](https://github.com/bcollazo/catanatron),
[pinned AlphaBeta implementation](https://github.com/bcollazo/catanatron/blob/ecf931181b9a65bb4116a2153fb78c16f1438e00/catanatron/catanatron/players/minimax.py),
[pinned evaluator](https://github.com/bcollazo/catanatron/blob/ecf931181b9a65bb4116a2153fb78c16f1438e00/catanatron/catanatron/players/value.py),
and [chance-outcome expansion](https://github.com/bcollazo/catanatron/blob/ecf931181b9a65bb4116a2153fb78c16f1438e00/catanatron/catanatron/players/tree_search_utils.py).

The inspected search defaults to depth two and a nominal 20-second deadline,
weights stochastic outcomes, and uses a handcrafted leaf evaluator. Its features
include production, resource variety, reachable production, buildable sites,
hand-to-build synergy, discard exposure, development cards, and awards. It
maximizes the acting agent's value and minimizes on other players' turns; that
opponent assumption is not a model of three independently self-interested agents.
The base evaluator's enemy-production term considers only one opponent.

This is our first external comparator because its implementation is inspectable
and runs locally. It is not established as a top online player. Full-state input,
deadline enforcement, board mapping and action protocols still need the adapter
checks listed under “Competitive agent and external comparator” below. Its
GPL-3.0-or-later source remains in a separate checkout; no policy implementation
has been copied into our package. Any modified wrapper must be identified in
results so we do not describe an adapted policy as an unchanged stock benchmark.

### Settlers-RL: learned policy plus root search

Primary sources: Henry Charlesworth's [writeup](https://settlers-rl.github.io/)
and [implementation/checkpoints](https://github.com/henrycharlesworth/settlers_of_catan_RL).
The project uses PPO, structured masked actions, and historical-policy opponents.
The author reports roughly 450 million decisions over a month on a 32-core,
128GB machine with a 3090 GPU. Forward search samples root actions, uses learned
policy rollouts and value estimates, and randomizes unknown information. With
32 search processes and a 10-second base decision budget, it reportedly won
47/100 games against three copies of its final policy. The author still describes
the agent as below a good human's level and reports poor trading behavior.

This motivates improving rollout/value quality before assuming a deeper tree is
necessary. It does not predict our win rate or fit our two-second budget. Its
simulator and information model require a rules audit before checkpoint reuse.
Our 122-row training prototype is pipeline validation, not comparable training
scale; the reported compute is context, not a requirement to copy blindly.

### Dobre: imperfect-information search and action categories

Primary source: [MCTS repository and research references](https://github.com/sorinMD/MCTS),
including Dobre and Lascarides, *Exploiting action categories in learning complex
games* (2017), and Dobre's *Low-resource Learning in Complex Games* (2018).
The implementation includes a factored Catan belief model, BMCTS, BMCTS with
observable rollouts, POMCP, and single-observer ISMCTS with chance nodes. It also
supports action-type priors learned from a human corpus. The README reports its
full belief-rollout variant as the slowest and weakest of those variants; that
is the author's result, not a universal ordering of algorithms.

This motivates testing action-family coverage and belief sampling cost. More
elaborate beliefs can consume the budget without improving decisions. The
repository exposes trade/negotiation and discard settings, so comparisons must
record those settings. We have not installed or benchmarked this Java system;
it is a research reference and possible second comparator.

### Online tools and commercial bots

**Target platform: Colonist.** Jerry specified Colonist for eventual human play.
Our local web player remains the development and pilot environment; local human
results alone do not establish transfer to Colonist. The planned progression is
local pilot, verified platform adapter, then private four-player Colonist matches
with informed participants and a frozen candidate/control. Record actual room
settings, player-visible observations, action acknowledgments, reconnects, and
both policy compute time and end-to-end action latency. The adapter must preserve
the existing observation-only policy interface and handle Colonist's trading
protocol without assuming it matches our engine.

Platform integration is not implemented or authorized yet. The official
[terms](https://colonist.io/terms), checked 2026-10-30 and dated October 6, 2026,
restrict scraping; require written authorization for internal API/private data
structure access outside the standard UI; and require prior written authorization
to use service data to develop or improve AI/automated decision systems. These
are concrete integration and data-use constraints, not evidence that private-room
automation is approved. Establish a permitted integration route before live
research matches. Continue training on our simulator in the meantime; do not
assume a third-party extension's existence grants platform permission.

[Noah Laforet's CatanBot README](https://github.com/NoahLaforet/CatanBot/blob/main/README.md)
describes a Colonist browser advisor built on Catanatron, handcrafted heuristics,
and one-ply evaluation, using DOM logs and WebSocket observations. It is evidence
of an integration approach, not a controlled strength result or a disclosure of
Colonist's own bot. We have not installed it or tested its claims.

The official [Colonist patch-notes page](https://colonist.io/patch-notes) was also
consulted, but no verifiable architecture description was obtained from it.
Our research has not established whether commercial online Catan bots use MCTS,
neural policies, or another method. Do not present guesses as their internals,
or equate beating an open-source comparator with beating ranked online humans.

### Performance research

[Cython's pure-Python documentation](https://cython.readthedocs.io/en/latest/src/tutorial/pure.html)
supports compiling Python sources while retaining an interpreted reference,
with static types available for further optimization. We chose that path to
avoid maintaining two rules implementations. Actual end-to-end measurements,
not general Cython speed claims, decide whether the backend meets our gate.
The local 19.57% incremental search improvement remains below the declared 20%
bar; compilation does not establish a strength improvement.

### How the research changes the implementation order

1. Finish the external adapter and differential rules/privacy checks. Require
   complete replayable four-player games before treating it as a comparator.
2. Improve strategic evaluation: expansion reachability, complementary income,
   port conversion, build timing, and all three opponents' victory threats.
   Compare against Tactical on fresh DEV boards under the shared clock.
3. Test stronger rollout policies and selective search against the same evaluator
   without search. Keep action-family coverage; spend samples on close decisions.
   A deeper tree is a candidate, not a prerequisite for progress.
4. Generate labels with a separately budgeted offline teacher, then distill a
   fast policy/value model. Inspect learning curves and board-level validation
   before scaling toward an initial 100,000 diverse positions. That scale is a
   proposed experiment, not a demonstrated sufficiency threshold or launched run.
5. Maintain diverse historical/style opponents, reserve fresh held-out boards,
   and freeze the selected agent before human evaluation. Jerry's pilot tests
   the match/replay flow; a larger controlled study must establish transfer.

Each candidate needs its own source-pinned config and numeric DEV bar before
execution. Failed local search and learned baselines remain failed; outside
research supplies hypotheses, not retroactive promotion evidence.

## Phase 3: search cost, opponent coverage and learning

Independent holdout audit completed: **learned baseline remains parked**.
`runs/learned-regression-holdout-v1.json` contains all 2,048 pairs / 4,096 games;
4,095 games completed, with one mixed-panel truncation. Execution took 806.9 s
on the named host. Recomputed summaries match the stored readout; every learned
seat loaded the pinned checkpoint, and every recorded winner has >=10 points.

| Holdout opponents | Learned wins | Tactical control wins | Paired change | Approx. one-sided 98.75% upper bound |
|---|---:|---:|---:|---:|
| Three Tactical | 0/512 | 128/512 | -25.00 pp | -20.05 pp |
| Three builders | 2/512 | 130/512 | -25.00 pp | -20.05 pp |
| Three expanders | 27/512 | 278/512 | -49.02 pp | -43.38 pp |
| Mixed | withheld | withheld | withheld: one truncation | withheld |

The three valid panels support substantial regression under the predeclared
normal-approximation assumptions. The mixed panel fails completion and receives
no aggregate strength estimate. Policy p95 was 0.102–0.117 ms and complete-turn
p95 0.601–0.621 ms; these local timings exclude HTTP/network overhead. The audit
cannot authorize promotion and does not alter any earlier DEV failure.

Phase-3 decisions: retain Tactical as the default. Park affordable root search,
sequence search and allocation search because each failed at least one panel's
paired-strength bar despite meeting latency/coverage gates. Park the learned
ranker and value head because test prediction failed, and the playable ranker
also regressed in independent gameplay. Preserve the datasets, checkpoints,
source snapshots and completed outcomes for future hypotheses; do not retune
against the opened test/holdout boards.

Completion evidence spans the actual paths: equivalent before/after optimization
traces and profiler artifacts; isolated 22-rollout ablations; four-panel fixed and
mixed games; calibrated fresh-board regression audit; adversarial rule/planning
tests; 40 replayed training/validation/test games with 199 verified observation
rows; deterministic retraining; real trained-model API play; and browser selection
with explicit fallback. The evaluator's interruption test also proves completed
pairs survive resume and a source change is rejected. Gameplay rules and the
original heuristic/Tactical decision code retain their frozen comparator behavior.
Final validation: 119 Python tests pass; the 14 browser tests passed after the
learned selector/fallback change, with no later browser changes. Work and artifacts
remain local; this milestone includes no external deployment or new Git commit.

Independent regression holdout preregistration: the learned checkpoint remains
parked after DEV and prediction-test failure. To finish the independent gameplay
measurement, `learned-regression-holdout.json` is a **non-promotion audit** of that
unchanged checkpoint and unchanged gameplay producers. It cannot return a
promotion PASS, even if its observations improve. This does not change any failed
DEV bar or open the reserved promotion holdouts. Only audit reporting differs
from the DEV runner; the runner verifies every other source hash and checkpoint.

Audit seeds are 1300000–1300127, 1301000–1301127, 1302000–1302127 and
1303000–1303127, one block per fixed/mixed panel; all are retired from future
training/tuning. Use 128 board clusters per panel, four paired seat rotations,
2,048 pairs / 4,096 games total. Primary result: approximate one-sided 98.75%
upper bound on learned-minus-Tactical win rate, computed by negating the existing
lower-bound function on negated seed clusters (Bonferroni across four panels).
Negative upper bounds support regression only under the stated normal-approximation
limits. Any match truncation withholds that panel's aggregate inference. Report
latency and completion independently; retain the original strength bars unchanged.

Power plan: with the largest DEV seed-cluster SD 0.3536 (and floor 0.25), 128
seeds target >=80% power for a regression of 0.10 under the normal planning model;
the minimum detectable magnitude is approximately 0.0964. This audit is not powered
for a +0.05 promotion claim. No optional stopping, retuning or checkpoint replacement.
Measured DEV execution implies roughly 11 minutes locally; run serially on
`Jerrys-Mac-mini.local`, process cap one, with the existing per-pair resumable journal.
The calibrated alpha is 0.0125; mirror the asymmetric null as an additional
prerun check before interpreting the upper bound.

Prerun checks passed: five focused evaluation tests, including a test that even
positive audit outcomes cannot produce promotion PASS. The mirrored calibration
(`runs/inference-calibration-four-panel-v2.json`, RNG 1240001) observed null
rejection rates 0.017 / 0 / 0 / 0 and alternative power 0.991. The source delta
from learned DEV is confined to audit prerequisite checks and summary reporting;
no simulation, model, feature or policy code changed. Frozen source:
`runs/learned-audit-v1-source`. The audit is launched with the command below;
its JSONL journal preserves completed pairs across an interrupted run.

```bash
CATAN_MODEL_PATH=runs/learned-baseline-v1.json PYTHONPATH=runs/learned-audit-v1-source uv run python -m catan_ai.search_eval --config experiments/learned-regression-holdout.json --dev-result runs/learned-dev-v1.json --calibration runs/inference-calibration-four-panel-v2.json --output runs/learned-regression-holdout-v1.json
```

Learned baseline v1 final readout: **ranking and value heads parked; no promotion.**
The unchanged checkpoint's eight-game test split has 40 retained positions:
ranking cross-entropy 1.03283 versus uniform 1.00738, agreement 0.30; value Brier
0.21337 versus constant-quarter 0.20000, ECE 0.17320. Thus the validation value
improvement did not transfer to this small test split. Artifacts:
`runs/learned-validation-v1.json`, `runs/learned-test-v1.json`.
Training used 122 positions from 24 games; validation/test games were never
used for gradients or hyperparameter tuning. All 40 generated trajectories were
replayed, verifying all 199 retained rows against legally observed features,
legal actions and terminal labels. A repeated training run reproduced both
weight vectors exactly (`runs/learned-baseline-v1-replay.json`).

Gameplay also **FAILS**: `runs/learned-dev-v1.json` contains 32 paired comparisons,
all 64 games complete. Paired win changes versus Tactical were -0.25 against
Tactical, -0.25 against builders, -0.75 against expanders and -0.125 on mixed
tables. Policy p95 was 0.000084–0.000123 s; complete-turn p95 0.000482–0.000689 s.
The checkpoint was loaded successfully in every learned seat, with its exact
SHA-256 pinned in the screen and per-game metadata. Fast inference does not
compensate for poor choices. No promotion holdout is opened for this failed model.

This is a deliberately small linear baseline trained from noisy two-world search
selection targets, on early branching positions and only the teacher's shortlist.
Inference ranks all legal actions; shortlist-to-full-action distribution shift,
weak/noisy teacher values and limited feature interactions are hypotheses for
future work, not established causes from this run. The value head estimates
wins under the data-generating lineup, not a universal game-theoretic value.

The real API loaded the checkpoint into three opponents and processed 40 human
actions through turn 36 on diagnostic seed 1290010. The web selector exposes the
model only as experimental and reports fallback when unavailable. Validation:
117 Python tests and 14 browser tests pass, including adversarial Tactical cases
for a depleted bank and an opponent at nine points. No external deployment.

Four-panel synthetic inference calibration passed at alpha 0.0125:
null rejection rates 0.008, 0, 0; +0.05 alternative power 0.988 at 2,048 seeds
(`runs/inference-calibration-four-panel-v1.json`). The limitations and prospective
power formula below still apply. The reserved gameplay holdouts remain unopened
because every candidate failed DEV; held-out prediction tests are a separate claim.

Reproduce the learned path with the frozen source and fresh output filenames:

```bash
PYTHONPATH=runs/learning-v1-source uv run python -m catan_ai.learn_pipeline generate --data runs/learning-v1 --split train --count 24 --targets 6
PYTHONPATH=runs/learning-v1-source uv run python -m catan_ai.learn_pipeline generate --data runs/learning-v1 --split validation --count 8 --targets 6
PYTHONPATH=runs/learning-v1-source uv run python -m catan_ai.learn_pipeline generate --data runs/learning-v1 --split test --count 8 --targets 6
PYTHONPATH=runs/learning-v1-source uv run python -m catan_ai.learn_pipeline train --data runs/learning-v1 --epochs 40 --seed 1 --output runs/learned-reproduction.json
PYTHONPATH=runs/learning-v1-source uv run python scripts/verify_learning_data.py --data runs/learning-v1
CATAN_MODEL_PATH=runs/learned-baseline-v1.json PYTHONPATH=runs/learning-v1-source uv run python -m catan_ai.search_eval --config experiments/learned-dev.json --output runs/learned-dev-repeat.json
```

Allocation DEV result: **FAIL strength; parked for promotion.** All 64 games
completed (`runs/search-alloc-dev-v1.json`). Paired self-win changes against
Tactical were 0, -0.125, +0.25 and 0 for Tactical/builders/expanders/mixed panels.
Coverage 0.854–0.969 and attempted-search p95 0.398–0.482 s passed their gates;
rollout truncation stayed below 0.001. All three affordable variants therefore
fail the predeclared strength requirement; none opens promotion holdouts.
The allocation run also shared the host with a 0.18-second focused test run.

Learning capacity **PASS**: seed 1230000 completed in 1.121 s and produced three
usable search-target rows from six attempts; 40 training epochs took 0.0022 s
of recorded optimizer time. The entire training command completed within the
five-second capacity limit. Preserved pilot: `runs/learning-pilot-v1.json`;
first shard: `runs/learning-v1/train/1230000.json`. Source is frozen at
`runs/learning-v1-source`, and the 24-game training generation resumes that shard.

The fixed 24-game checkpoint is `runs/learned-baseline-v1.json` (full training
call 0.0746 s). Validation used eight disjoint games / 37 retained positions:
ranking cross-entropy 0.95658 versus uniform 0.94519 (**FAIL** the required 1%
improvement); value Brier 0.16553 versus constant-quarter 0.18412 (**PASS** its
prediction bar), with ECE 0.15385. These are small, early-position samples and
not strength evidence. Keep the checkpoint unchanged for the predeclared eight
test games; no hyperparameter tuning or retraining follows this readout.

Learned gameplay DEV is a diagnostic despite failed ranking prediction, not a
promotion attempt. `learned-dev.json` compares this exact checkpoint to Tactical
on two boards per panel / all four seats, seeds 1233000–1233001,
1233100–1233101, 1233200–1233201 and 1233300–1233301. The same +0.05 lift per
panel, zero truncations and <=2 s policy-p95 gates apply. First run one complete
capacity game on seed 1232500, requiring completion and policy p95 <=2 s.
All these seeds and the synthetic tactical tests 1290000–1290001 are diagnostic
only, excluded from future training/tuning holdouts. API testing uses 1290010.

Four-panel inference preregistration: retain the paired board-seed mean (four
seat rotations per seed), one-sided normal lower bound with sample-SD floor
0.25, minimum 128 independent seeds, and Bonferroni per-panel alpha 0.0125.
Before any four-panel held-out use, run the existing symmetric/sparse/rare-negative
null probes at that alpha with 1,000 replicates each and require rejection <=0.0375;
require >=80% power on the fixed +0.05 alternative at 2,048 seeds, RNG seed
1240000. This finite synthetic calibration is a diagnostic, not a universal
finite-sample coverage guarantee. The prospective sample-size formula remains
`max(128, ceil(((z(1-alpha)+z(.8))*max(.25, SD_DEV)/.05)^2))` per panel;
it targets a positive lower bound and does not give 80% power for the additional
observed +0.05-lift gate. DEV failures do not open promotion holdouts. Reserve
1250000–1259999, 1260000–1269999, 1270000–1279999 and 1280000–1289999 exclusively
for future four-panel held-out evaluation; these intervals are outside every
learning split and all opened DEV boards. An eventual passing candidate freezes
the exact panel sample sizes from DEV before those outcomes are opened.

Learning pilot preregistration: an observation-only linear action ranker learns
soft targets from completed `search_fast` selection rollouts, with additive
half-win smoothing and temperature 0.25. A separate logistic value head learns
the eventual self-win outcome under the trajectory's actual policy lineup.
These are different targets; neither is assumed calibrated before measurement.
The producer rotates Tactical/builder/expander/heuristic seats, caps teacher calls
at six per game, retains failed-search diagnostics without using them as labels,
and never labels incomplete games as losses. The small pilot deliberately samples
early branching positions; it does not represent all phases or late-game states.

Entire board families are assigned exclusively to training 1230000–1230099,
validation 1231000–1231029 and test 1232000–1232029. The generator refuses seeds
outside those intervals. Start with one training game, six teacher calls, and
40 fixed SGD epochs to benchmark the exact producer/trainer. Capacity bars:
complete game, >=1 usable target, generation <=30 s, training <=5 s. If those
pass, use 24 training games and eight validation games before opening eight test
games. Prediction DEV requires policy cross-entropy at least 1% below uniform
on the same shortlisted actions and value Brier score below a constant 0.25
predictor; report calibration bins/ECE and top-action agreement separately.
Fixed learning rate 0.05, L2 0.001 and seed 1; no test-based hyperparameter tuning.
Failed heads remain opt-in research baselines, with explicit park decisions.
Gameplay requires the same four-panel gates against Tactical as search; prediction
accuracy cannot substitute for strength. Missing/invalid checkpoints fall back
to Tactical. Inference scores only legal observed actions and falls back on
repeated states within a turn. The value head is exposed separately for later
search integration, not silently used to select actions.

Learning entry points (not yet measured): `python -m catan_ai.learn_pipeline`
provides `generate`, `train`, and `evaluate`. Per-game shards contain the full
action trajectory, source/input identity, teacher diagnostics, extracted legal
features and terminal labels. Resume accepts only compatible complete shards;
training and metrics files are write-once. Producer fingerprints cover generation
and feature extraction, so a trainer-only repair need not regenerate valid games.
`CATAN_MODEL_PATH=/path/to/checkpoint.json` enables the `learned` simulation/API
policy. API creation reports `model_loaded`; strength screens refuse missing
models and pin exact checkpoint bytes before and after each pair. Prediction
evaluation also rejects a different dataset producer from training.

Affordable root DEV result: **FAIL strength; parked for promotion.** Receipt
`runs/search-fast-dev-v1.json`; frozen producer `runs/search-ablation-v1-source`.
All 64 games completed. Paired win changes against Tactical were +0.125 versus
Tactical opponents, -0.125 versus builders, +0.375 versus expanders, and -0.25
on mixed tables (eight pairs per panel). All latency/coverage gates passed:
attempted-search p95 0.363–0.488 s and coverage 0.852–1.0. These small DEV samples
do not establish population strength. No held-out run follows this failure.
Sequence/allocation runs share the frozen producer and the same DEV boards;
they run serially on the named local host with one evaluation process. Brief
focused learning/API checks also ran on that host (0.03 s and 0.20 s), so timings
reflect that observed load rather than a fully isolated host.

Sequence DEV result: **FAIL strength; parked for promotion.** Receipt:
`runs/search-sequence-dev-v1.json`. All 64 games completed; 78 decisions evaluated
a multi-action candidate. Paired win changes were 0 versus Tactical opponents,
+0.125 versus builders, +0.375 versus expanders and -0.25 on mixed tables.
Coverage was 0.837–1.0; attempted-search p95 0.361–0.501 s; rollout truncation
remained below 0.001. The mixed-panel loss and zero Tactical-panel lift fail the
predeclared bar. Neither this result nor the builder-panel improvement warrants
promotion or opening held-out boards.

After the serial search screens finish, benchmark the learning pilot without
competing simulation work:

```bash
uv run python -m catan_ai.learn_pipeline generate --data runs/learning-v1 --split train --count 1 --targets 6
uv run python -m catan_ai.learn_pipeline train --data runs/learning-v1 --epochs 40 --seed 1 --output runs/learning-pilot-v1.json
```

Inspect the shard's generation time, completion and usable rows and the checkpoint's
training time against the capacity bars before increasing dataset size. The pilot
checkpoint is preserved separately from the eventual 24-game training checkpoint.

Builder repair DEV (declared before execution): allow paid-road expansion below
twelve roads after city/development/settlement opportunities, and include roads
as a final bank-trade target. Require all 11 previously truncated games to finish
and zero truncations across 32 fresh seat-rotated pairs against three builders
(seeds 1210400–1210407, diagnostic DEV only). Preserve the v1 source/results.
This is an opponent-completion gate, not a relaxed candidate-strength gate;
the heuristic still requires +0.05 lift to pass a strength screen.

Builder repair result: **completion PASS**, all 11 regression games and all 64
fresh games finished (`runs/builder-repair-regression-v2.json` and
`runs/builder-repair-dev-v2.json`). Heuristic strength still FAILS: -0.15625
paired win lift against Tactical. Neither completion nor this small screen
promotes a candidate. Builders now retain city/development priority but can
expand with paid roads below twelve roads; original v1 artifacts remain intact.

Affordable-search ablations (predeclared): `search_fast` uses Tactical at the
root/fallback, heuristic continuations, three candidates, two selection worlds
and eight fresh confirmation worlds (maximum 22 rollouts). `search_sequence`
changes only candidate construction, reserving one slot for Tactical's complete
scoring sequence when it has multiple actions. `search_alloc` changes only
allocation to four selection and five confirmation worlds, retaining the same
22-rollout ceiling. All use positive paired confirmation mean, explicitly an
experimental heuristic rather than a confidence-certified override. The initial
capacity gate is an entire game completed with attempted-search p95 <=2 s on
`Jerrys-Mac-mini.local`, process cap one, seed 1220000 (diagnostic only).
Do not increase budgets or alter this gate after results. Candidates clearing
capacity can enter fresh four-panel DEV screens with no truncations, >=0.05
paired lift against Tactical per panel, >=0.8 search coverage, <=0.01 rollout
truncation and <=2 s attempted-search p95. Capacity alone cannot promote.
The shared capacity board did not exercise a sequence candidate. Add a triggered
diagnostic on seed 1220001: use the first heuristic-play position with a
multi-action Tactical plan; require an actual sequence rollout and <=2 s choice
time. This board is also excluded from held-out and learning-test splits.

Capacity result: all three full-game probes completed. Attempted-search p95 was
0.394 s for fast root search, 0.398 s for sequence search, and 0.447 s for the
allocation variant. Complete-turn p95 was 0.531, 0.541 and 0.935 s respectively.
The allocation game had 11 belief failures in 36 attempts; the other games had
none in 37 attempts. Outcomes diverge, so these are not matched-position timing
comparisons. Receipts: `runs/search-{fast,sequence,alloc}-capacity-v1.json`.
The triggered sequence probe at step 362 / turn 61 evaluated a multi-action
candidate in 0.422 s (`runs/search-sequence-trigger-v1.json`), passing its gate.
Sequence rollouts follow the planned builds; the real policy executes the first
action and replans, so sequence values assume commitment during simulation only.

Next DEV configs are `search-fast-dev.json`, `search-sequence-dev.json` and
`search-alloc-dev.json`. Each uses two boards and four seats per panel (32 pairs)
with the gates above. Shared ablation DEV seeds are 1221000–1221001,
1221100–1221101, 1221200–1221201 and 1221300–1221301. They are retired from
future held-out evaluation and learning test splits. No held-out run is
authorized by capacity results alone; these remain unpromoted candidates.

Opponent-suite DEV preregistration: `opponent-suite-dev.json` compares the
original heuristic to Tactical across four fixed/mixed lineups, eight fresh
boards per lineup and all four seats (128 pairs). Each panel requires no
truncations, >=0.05 paired win lift and policy p95 <=2 s. This small diagnostic
screen uses the existing seed-cluster Hoeffding bound and makes no held-out
promotion claim. Seeds 1210000–1210007, 1210100–1210107, 1210200–1210207 and
1210300–1210307 are DEV only; style correctness tests use 1200010–1200011.
Two deterministic, independently named opponents broaden the benchmark:
`builder` weights ore/grain placement and prioritizes cities/development;
`expander` weights wood/brick and prioritizes settlements and roads up to twelve.
Both reject player offers, use only legal observed actions and keep fixed
weights. They are benchmark opponents, not strength-validated promotions.
The screen now accepts an explicit control policy, retaining heuristic as the
historical default; held-out validation also checks control and opponent lineups.

Opponent-suite v1 result: **FAIL; no promotion or held-out launch.** Receipt:
`runs/opponent-suite-dev-v1.json`, with all 128 pairs in its resumable journal.
The heuristic's paired win change against Tactical was -15.625 percentage points
against three Tactical bots, -6.25 against expanders, and -9.375 on mixed tables;
those panels had no truncations. The builder panel had **11 truncated games**
across its 32 pairs, so its strength estimate is withheld. The builder has no
paid-road expansion path, which can strand city/development tables; repair and
a fresh-version completion screen are required before using it for promotion.
No uncertainty bound establishes a strength difference at this sample size.
Search coverage and rollout truncation fields are N/A for these nonsearch arms.

```bash
PYTHONPATH=runs/opponent-suite-v1-source uv run python -m catan_ai.search_eval --config experiments/opponent-suite-dev.json --output runs/opponent-suite-dev-repeat.json
```

The standing comparator is the validated Tactical policy; the original heuristic
is retained as a separate control. Existing held-out seeds 970000–970127,
980000–980127 (reserved), and 1011000–1011154 (opened) are excluded from new
training/tuning. Previously opened DEV and benchmark boards are diagnostic only.
New profiling and timing tests use 1200000–1200005, with search-probe policy seeds 1200100–1200102; these boards
are also excluded from future held-out trials and learning test splits.

First DEV is semantic-preserving performance work. Before changes, record four
complete heuristic/tactical trajectories and three real MC decision probes with
`scripts/profile_pipeline.py`. Acceptance: identical action/RNG/terminal-state
trace hashes and search actions/diagnostics (excluding time), all privacy and
malformed-action tests passing, and >=30% reduction in aggregate unprofiled
search-probe time. No strength claim follows from this test. Profile the actual
paths separately; do not include profiler overhead in timing comparisons.
Tactical/default selection stays unchanged during this phase.

Performance DEV: **PASS** on `Jerrys-Mac-mini.local`, one benchmark process.
`runs/phase3-profile-before-full.json` and `runs/phase3-profile-after.json` retain
the measurements and source hashes; adjacent `.prof` files retain the actual
game and search profiles. All four action/RNG/terminal-state traces and all three
search actions/diagnostics match exactly. Aggregate unprofiled search time fell
from 1.8055 s to 1.2014 s (**33.5%**); complete trajectory time fell from 2.7519 s
to 0.1808 s (**93.4%**). These are bounded diagnostic probes, not a population
p95 estimate or strength result. The late-game probe still falls back after a
belief failure; that failure is identical before and after optimization.

Direct policy observations avoid constructing browser state and other players'
trade rates. Explicitly stateless policies omit copied event history; search and
unknown policies still receive it. Real games continue recording all events.
Action validation retains canonical JSON type checking while skipping encoding
of unequal legal actions. Validation: 103 Python tests passed, including
observation equivalence against the previous API path and malformed actions.

Reproduce the optimized profile with a fresh output filename:

```bash
uv run python scripts/profile_pipeline.py --output runs/phase3-profile-repeat.json --profile
```

Simulation now records three distinct latency units: policy choice, complete
decision step (observation + choice + engine application), and complete turn
(roll through end-turn or victory, including trade responses and discards by
other players). Setup actions have step timings but are excluded from turn
timings. Interrupted turns are marked incomplete and excluded from turn p95.
These are local simulation timings, excluding HTTP/browser/network latency.
The search screen reports step and turn p95 for each arm separately; missing
timings in historical artifacts remain unavailable, not zero. The original
policy-decision gate is unchanged.

The first full-game timing probe (`runs/phase3-timing-probe.json`, seed 1200005,
MC mean against three heuristics, same host/process cap) completed in 33.41 s.
Among 55 attempted searches, policy p95 was 1.574 s, with 18 belief failures.
All candidate steps, including nonsearch decisions, had p95 0.914 s; complete
candidate turns had p95 2.379 s. The different populations explain why step p95
can be lower than attempted-search p95. This single diagnostic game is neither
a strength result nor a promotion gate. Timing accounting has six focused tests
passing, including the existing search-summary gates.

```bash
uv run python scripts/benchmark_latency.py --seed 1200005 --candidate mc_mean --output runs/phase3-timing-repeat.json
```

## Monte Carlo root-search DEV

`mc-dev.json` preregisters two panels: three heuristic opponents and a mixed
heuristic/strategic/random lineup. Each uses two fresh seeds and all four candidate
seats, paired with the same game using the heuristic in the candidate seat.
The comparison contains 16 search games and 16 matched baseline games.

Each panel must have zero match truncations, at least +0.05 paired win-rate gain,
search coverage >=80% of eligible branching decisions, p95 latency <=2 seconds
among attempted searches, and rollout truncation <=1%. DEV is a cheap signal
screen; its small sample does not establish strength. Unsupported phases and
single-candidate decisions are counted separately, and cannot dilute search latency.

Run on the local host with one evaluation process:

```bash
uv run python -m catan_ai.search_eval --config experiments/mc-dev.json --output runs/mc-dev-v1.json
```

The `.jsonl` journal preserves each completed candidate/control pair. Re-running
the command resumes only when config, source hashes and host match; completed
output is never overwritten. A crash can require repeating the unfinished pair.

`mc-heldout.json` is conditional on DEV passing at the same source. It reserves
128 new seeds per panel with four seat rotations each. Promotion additionally
requires a strictly positive one-sided paired seed-cluster Hoeffding lower bound,
with Bonferroni correction across the two panels. It retains DEV's latency,
coverage and truncation gates. No held-out results are opened to tune DEV.

The initial capacity check on `Jerrys-Mac-mini.local`, one process, seed 940000,
finished a full search-vs-heuristic game in 19.99 seconds with 960 rollouts and
zero belief failures or rollout truncations. No challenger was confirmed. Its
all-decision p95 was 1.02 seconds and maximum 2.58 seconds; the DEV gate uses the
stricter attempted-search p95. This is capacity evidence for that recorded source,
not strength evidence; the later source also conditions beliefs on public
production-shortage and victory observations. Receipt: `runs/mc-benchmark-940000.json`.

Result: **FAIL overall; parked for default promotion.** All 32 matches completed.
Receipt: `runs/mc-dev-v1.json`, with resumable pair records in
`runs/mc-dev-v1.jsonl` (local, ignored by Git).

| Panel | Paired win-rate change | Search coverage | Search p95 | Rollout truncation | DEV |
|---|---:|---:|---:|---:|---|
| Three heuristics | 0 percentage points | 94.26% | 2.155 s | 0.141% | FAIL |
| Heuristic/strategic/random | +12.5 percentage points | 99.44% | 1.894 s | 0.671% | PASS |

The 20,908 rollouts confirmed only three overrides across sixteen search games.
The three-heuristic panel missed the required gain and latency gates. The mixed
panel's gain is one additional win in eight games, not established strength; both
panels' conservative paired seed lower bounds are -1. Python validation briefly
shared the local host during the first panel, so timings describe that observed
host load rather than an isolated capacity measurement. The zero win-rate gain
independently fails its gate. The held-out run was not launched, as preregistered.
At that milestone the default remained heuristic; MC stayed opt-in.

Next DEV idea: allocate more confirmation samples to a small set of consequential
decisions, using an independent selection stage and a newly declared compute
budget. Measure override frequency and paired outcome change without weakening
the statistical guard after looking at outcomes. Belief failures also remain a
separate coverage target. Tree search and learned value/policy models remain
follow-on work.

### Search debugging: shortlist repair DEV

Before testing the shortlist repair, its cheap pass bar is to remove at least 12
of the 16 useful-maritime-trade omissions in exact replays of the recorded DEV
games, while keeping the baseline first, all alternatives legal, and the five-action
budget intact. This measures candidate coverage, not win-rate improvement. The
equal-weight resampling repair must preserve distinct particles in its private-draw
regression and pass the belief/engine privacy and conservation tests. Neither test
changes the failed promotion result or authorizes a default change.

Debugging result: the original search did **not** lose more matches than its
paired heuristic control. They each won 2/8 against three heuristics; search won
4/8 versus the control's 3/8 in the mixed panel. Exact replay reproduced all sixteen
search-game winners, turn counts, point totals, and all 2,077 recorded decisions.

The principal measured bottleneck is conservative abstention. Four selection
samples left the baseline tied for best in 401 of 494 baseline-selected decisions.
Of 154 completed confirmation comparisons, 67 had positive estimated improvement;
64 of those were rejected. With 32 samples the Hoeffding penalty is 0.4327, so the
estimated improvement must exceed 43.27 percentage points. This is sampling
uncertainty control, not evidence that the rejected challengers were better.
Only three decisions changed. The existing bound would require more than 2,396
samples to clear a five-percentage-point observed advantage; simply increasing
the budget everywhere would worsen the already-failed latency gate.

Two implementation defects were reproduced with failing regressions and repaired:
maritime candidates followed enum order instead of prioritizing an existing
heuristic build-deficit trade, and equal non-unit likelihoods unnecessarily
resampled particles, adding avoidable loss of diversity. The shortlist repair
eliminated all 16 omissions (bar: at least 12), preserving baseline-first, legality
and the five-action cap throughout exact replay. The resampling regression passes.
These are coverage and correctness results; repaired-policy strength is unmeasured.

Offline probes of the three original overrides used actual hidden hands and
remaining deck composition with new future randomness, solely to diagnose those
decisions. The two fully completed 128-pair probes still favored the selected
roads under heuristic continuations (+47.66 and +43.75 percentage points). The
other probe truncated one pair and is inconclusive. These privileged diagnostics
do not establish attainable strength, and mixed-panel probes do not model the
actual mixed opponents. There is no evidence here that the sole changed move in
the all-heuristic panel was an obvious rollout-value regression; it still lost
the one realized game.

Artifacts: `runs/mc-debug-v1.json`, `runs/mc-shortlist-debug-v2.json`, and the preserved
original producer under `runs/mc-v1-source/`. Replay the original producer with:

```bash
PYTHONPATH=runs/mc-v1-source uv run python scripts/debug_search.py --output runs/mc-debug-repeat.json
PYTHONPATH=runs/mc-v1-source uv run python scripts/debug_search.py --shortlist-source src/catan_ai/search.py --probes 0 --output runs/mc-shortlist-repeat.json
```

The current source includes the two repairs. Original match numbers apply to the
archived producer, not this repaired policy; that debugging step did not change the default.

### Red-team recommendation: beating the frozen heuristic

The next objective should be measured improvement over the frozen heuristic,
not a larger search tree. Independent read-only reviews covered policy tactics,
search/beliefs and evaluation. Findings below were checked against the code;
unverified small probes are not strength evidence.

1. **Build a separate tactical policy first.** Replace random Year of Plenty
   choices with build-aware choices, check guaranteed winning build sequences,
   and preserve resources needed for those sequences. A synthetic action-phase
   fixture (board seed 31, turn 8, three cities and three settlements on spaced
   vertices 0/2/4/6/8/10, own ore=3/grain=1, old Year of Plenty) demonstrates the
   gap: heuristic RNG seed 1 takes brick+wool, while grain+ore followed by a city
   wins immediately. This is a tactical counterexample, not a measured match gain.
   Implement entirely from public board and own private information; never hand
   a live omniscient Game to the policy. First bar: 20 preregistered tactical
   cases covering winning builds/card combinations, all solved without illegal
   actions or hidden-state access. Then compare a separate policy against the
   frozen heuristic; do not silently redefine the comparator.
2. **Ablate the per-move veto.** Keep the current sampler, shortlist and sample
   budget fixed, but test an opt-in policy choosing a challenger when its fresh
   confirmation mean exceeds the baseline's mean, without demanding a positive
   Hoeffding LCB for each move. Keep the LCB version as a control. This changes
   decision-making risk, not the required evidence for promotion. More overrides
   are diagnostic only; the variant must actually win more games. If this has
   signal, separately test sequential halving to allocate root samples rather
   than stopping on four-sample ties. Root simple-regret research motivates
   this direction; it does not establish Catan strength:
   [Tolpin and Shimony](https://ojs.aaai.org/index.php/AAAI/article/view/8126),
   [Sagers et al.](https://arxiv.org/abs/2411.07171).
3. **Search short action sequences.** Explicitly consider maritime trade → build,
   Year of Plenty → build, and road → settlement/award sequences. The current
   rollout does simulate follow-ups, but the heuristic chooses them: it does not
   optimize the continuation. Reserve candidate coverage for these tactical
   plans; round-robin insertion order can still starve entire action families
   despite the repaired ordering within the trade family. First bar: all 20
   tactical winning lines retained and completed under a bounded turn search.
4. **Test robber and road improvements separately.** Robber scoring should account
   for blocked production, city multipliers and harm to one's own income. Roads
   should have a reachable settlement or award objective rather than a fixed
   60% spending coin flip. These are plausible cheap lanes, not proven gains.
   Only after an improvement passes should it become a search rollout policy;
   compare tactical-only, search-only and combined policies to isolate causes.

Proposed DEV for each surviving lane: 32 fresh independent board seeds, four seat
rotations, paired with the frozen heuristic against three heuristic opponents.
Predeclare >=+0.05 paired self-win lift, zero match truncations and search-decision
p95 <=2 seconds on a named host with one evaluation process and no concurrent
timing tests. This is a screening bar, not a significance claim. Register exact
disjoint seeds and implementation before execution; do not reuse the opened DEV
boards as evidence for a newly tuned policy. Benchmark search cost before running
the full screen. Cross-play against other styles follows a positive primary
screen, and does not replace the all-heuristic target.

The current proposed held-out test is underpowered for modest gains: with 128
independent seeds and two panels, its Hoeffding penalty is 0.2401. A five-point
observed gain would need about 2,952 independent seeds just to clear that bound.
The formula is correct: averaging four balanced seat rotations first produces
the same mean as the existing code. Before launching a new held-out test,
preregister a paired seed-level inference method and sample size appropriate for
the desired effect, with a null-calibration and power check; an approximate
bootstrap must be labeled approximate. Keep held-out seeds unopened during tuning.

Defer a neural value/policy model and full information-set MCTS until these lanes
show signal. A per-action confidence interval covers sampled continuations under
an approximate belief/opponent model; it is not a guarantee about actual playing
strength. Public production, public build costs and Monopoly hand-count changes
are observable or inferable; the review did not substantiate the initially alleged
resource-history privacy leak.

### Tactical and mean-search implementation DEV

Registered before outcome execution: `tactical` uses a public/own-hand projection
to search at most three deterministic actions and 128 nodes. Only the root may
acquire resources (a known-legal maritime trade or Year of Plenty); subsequent
builds cannot invent unknown bank supply. Road Building and road/build sequences
are included. Twenty winning-turn fixtures must pass before the match screen.
`mc_mean` changes only the final selection rule to positive fresh paired mean;
`mc_tactical` also uses the tactical policy as baseline and own continuation.
Opponent continuations remain heuristic. The original heuristic is unchanged.

Cost gate before scaling: benchmark a real branching observation from board 29
with policy seed 4, and require <=2 seconds for each new search variant. This
single-position gate can reject an obviously excessive implementation; passing it
does not establish game-wide p95. Surviving policies use the following separate,
fresh-seed screens: tactical 1001000–1001031; mean search 1002000–1002031;
combined search 1003000–1003031. All use four seats, paired heuristic controls,
three heuristic opponents, >=+0.05 paired win lift, zero match truncations, and
p95 <=2 seconds. Search additionally requires >=80% coverage and <=1% rollout
truncation. All screens are DEV, not evidence of statistical significance.

Alternative inference is explicitly approximate: a one-sided normal bound on
independent seed averages with a standard-deviation floor of 0.25 and a minimum
of 128 seeds. Its floor protects against misleadingly constant tiny DEV gains;
it is not a distribution-free guarantee. Before use, a seeded calibration runs
1,000 replicates at 128 seeds under symmetric, sparse and rare-negative nulls;
each rejection rate must be <=7.5%. A 1,024-seed +/-0.5 alternative with mean
+0.05 must have >=80% simulated power. Record the source and all calibration
settings. Plan any held-out sample size from DEV variance (with the same floor)
for a five-point gain and 80% approximate power, and freeze that size and new seeds
before execution. Existing Hoeffding results and failed gates remain unchanged.

Implementation results so far:

- All 20 tactical winning-line cases pass; full validation is 100 Python tests
  and 13 browser tests.
- The mean-search cost probe took 0.313 seconds. The initial combined tactical
  rollout policy took 17.42 seconds and failed the <=2-second cost gate, so no
  combined strength screen was launched. Receipt: `runs/strength-cost-v1.json`.
  Subsequent tactical placement tie-breaking retains baseline production scoring;
  the combined policy needs a fresh cost pass before any larger run.
- Inference calibration passed: symmetric null 5.0%, sparse null 0%, rare-negative
  null 0%, and tested-alternative power 93.3%. Receipt:
  `runs/inference-calibration-v1.json`. These checks cover the specified synthetic
  cases, not every possible outcome distribution.
- Tactical DEV passed: 45/128 candidate wins versus 32/128 paired control wins,
  +10.15625 percentage points, zero truncations, decision p95 0.00214 seconds.
  Receipt: `runs/tactical-dev-v1.json`; 32 independent seeds, not 128 independent
  observations. Its Hoeffding lower bound remains negative; this is a DEV signal.
  Search coverage and rollout-truncation fields do not apply to this non-search
  policy. The DEV variance-based plan suggests 155 held-out seed clusters.

`tactical-heldout.json` freezes 155 new seeds, 1011000–1011154, before opening
held-out outcomes. It requires matching-source DEV PASS and inference calibration,
positive approximate paired lower bound, observed lift >=0.05, all games complete
and the latency gate. Promotion was conditional on that held-out result.

Mean-search DEV completed all 128 pairs: 40/128 candidate wins versus 32/128
control wins, +6.25 percentage points, zero match truncations. It made 564
mean-based overrides over 170,164 rollouts, with 89.51% search coverage and
0.2803% rollout truncation. Search-decision p95 was 2.350 seconds, so the overall
DEV gate **failed on latency**, despite clearing the observed-gain bar. Its
Hoeffding lower bound was negative, so the gain remains exploratory. Receipt:
`runs/mc-mean-dev-v1.json`; paired execution took 4,257 seconds on the named local
host with one evaluation process. Tactical DEV took 228 seconds. The two screens
use different fresh boards and do not establish a tactical-versus-search ranking.
No mean-search held-out run is authorized by this failed gate.

The exact producer shared by both screens is preserved under
`runs/strength-v2-source/`. Tactical's preregistered held-out run completed all
155 seed clusters before its outcomes were read.

**Held-out PASS:** Tactical won 228/620 games versus the frozen heuristic control's
155/620, an observed +11.774 percentage-point gain. The approximate one-sided
95% paired seed-cluster lower bound is +8.471 points. All 1,240 games completed;
policy-decision p95 was 0.002363 seconds. This excludes observation construction
and HTTP overhead, and is not an end-to-end web latency measurement. Paired
execution took 1,235 seconds. Receipt: `runs/tactical-heldout-v1.json`; config:
`tactical-heldout.json`. The normal approximation, variance floor, calibration,
minimum sample count and disjoint seeds were fixed before this run. The power
planning targets the positive-lower-bound test; the additional observed-gain bar
is a separate criterion and does not inherit that power claim.

After verifying source/config equality and recomputing the entire result from
saved pairs, only the web/API default selection was changed to Tactical. Policy,
engine and inference modules remain identical to the measured producer; the
original heuristic remains available and unchanged. There was no external deploy.
This validates improvement against the named heuristic in our engine, not online
human strength. Mean-search is not promoted, and the combined policy remains
parked for cost. New work should optimize search cost or test transfer to other
opponent styles without reopening these held-out boards for tuning.

Post-promotion validation: 101 Python tests and 13 browser tests pass, including
an assertion that new API games construct Tactical bots by default. An AST check
confirmed the original `HeuristicBot` is unchanged; source comparison against the
held-out receipt found only `api.py` different (default selection), with every
policy, engine and inference module unchanged. All recorded terminal winners
have at least ten points. To reproduce the frozen producer after the API default
change, prefix evaluation commands with `PYTHONPATH=runs/strength-v2-source` and
use a new output path; original receipts are never overwritten.

## Resource-aware placement DEV

Sources: Colonist's [starting-strategy guide](https://blog.colonist.io/guide-to-catan-starting-strategies/)
and [victory-point strategy guide](https://blog.colonist.io/how-to-gain-victory-points-in-catan/).
The guides motivate complementary placements, matching resource income to build
plans, and exploiting compatible ports. They do not establish our scoring weights.

`placement-strategy-dev.json` tests `strategic` against three `heuristic` opponents
on 100 seeds, rotating through all four seats. Before execution its bar was all
400 games complete and the seed-cluster lower bound at least 0.25. The candidate
uses diminishing returns for repeated resource income, grain/ore weights,
production bottlenecks for expansion versus cities/development, and port synergy.
Other baseline decisions are retained.

Result: **FAIL; parked, not promoted.** 399/400 completed, 104 candidate wins,
one truncation; the runner correctly withheld the aggregate win rate and interval.
Receipt: `runs/placement-strategy-dev-20261029T170621734794Z.json` (local, ignored
by Git). The web default was kept as `heuristic` at that milestone. A missing-resource placement
fixture passes, but it is correctness evidence, not strength evidence.

The results below describe the initial simplified ruleset. They do not measure the
expanded engine with development cards, ports, player trades, or bonus awards.
The original runner's Wilson interval also treats seat rotations as independent;
rotations share a board seed. Schema 2 now uses a conservative two-sided Hoeffding
interval over each seed's four-seat average, assuming independent seed clusters.
It refuses promotion when any game truncates or source files change during the run.
The initial result remains historical evidence only.

`heuristic-v-random.json` is the first DEV screen. Its pass rule was declared as:
all 100 games complete and the candidate win-rate Wilson 95% lower bound is at
least 0.35.

The 2026-10-29 implementation check completed 100/100 games in 0.85 seconds.
The heuristic won 44 (44.0%, Wilson 95% interval 34.67%–53.77%), so the screen
**failed** its declared lower-bound bar by 0.33 percentage points. This establishes
that the runner, seat rotation, truncation guard, and fail-closed pass rule work; it
does not promote the heuristic. Improve the policy, preregister a new seed block,
and write a new config for the next comparison.
## Cython engine DEV gate

The optional native backend compiles the authoritative Python rules, with the
Python backend retained as the differential reference. Before measurement, the
acceptance bar is identical complete-game trace hashes, winners and turn counts
and identical search actions/diagnostics, plus at least 20% less aggregate search
time across three alternating Python/native pipeline probes on this host with
one process at a time. Seeds 1400000–1400001 and search RNG seeds 1400100–1400102
are diagnostic-only. Both backends must pass the complete Python test suite.
Compilation/performance does not count as gameplay-strength evidence. Native
activation is explicit through `CATAN_ENGINE=cython`; stale builds fail closed.

Measured on the local host, one benchmark process at a time: compiling only the
rules reduced search time by 16.07% (`runs/native-v1-comparison.json`). Compiling
road scoring as well gave 19.31% (`runs/native-v2/comparison.json`). Replacing the
recursive scoring generator with a loop gave 18.56% relative to the improved
Python reference (`runs/native-v3/comparison.json`). Removing redundant copies
of fresh legal actions and using detached payload copies gave 19.57%
(`runs/native-v4/comparison.json`): aggregate Python search time 4.9418s versus
native 3.9747s across the three replicates. Every probe matched complete game
traces, search actions and diagnostics. The 20% incremental-native gate remains
**FAIL**; do not round this into a pass. Both the Python and native path benefited
from the general optimizations. Native remains explicitly selected, not default.
The initial v1 single probes are diagnostic only; use the dedicated alternating
measurement files for timing. Differential-test seeds 1400002–1400004 and the
external smoke-test seed 1400020 are also diagnostic-only.

## Competitive agent and external comparator

`experiments/competitive-plan.json` declares the four-player format, shared
2-second decision / 10-second turn compute budget, DEV and held-out criteria,
and human-transfer design. Jerry volunteered as the first human pilot; pilot
games establish usability and timing, not competitive transfer.

Catanatron is checked out locally under `runs/external/catanatron` at
`ecf931181b9a65bb4116a2153fb78c16f1438e00`, with a separate environment at
`runs/external/catanatron-env`. Its unmodified depth-two AlphaBeta policy returned
a legal initial placement in 0.0131s on seed 1400020. This is an installation
smoke test, **not** a verified external matchup. Audit findings to resolve:

- `Game.play_tick` passes full state to `Player.decide`. Monopoly transitions
  read opponent resources, and opponent move generation can use hidden cards.
  Supply observation-derived worlds; never pass the actual hidden state.
- `expand_spectrum` eagerly creates successors, while the timeout is checked
  only on recursive entry. Enforce the shared wall deadline outside the worker.
- Robber movement and victim selection are combined externally, while our
  engine separates them. Player trade response protocols also differ. Verify
  action and state mappings before any gameplay comparison.
- Use identical board geometry, tokens, ports, turn ownership, development-card
  timing, bank supplies and legal moves. The external default number-placement
  scheme differs from ours, so regenerating both boards from one seed is not
  board equivalence.

External dependencies installed for the smoke test: Catanatron 3.3.0 from the
pinned checkout, click 8.5.0, markdown-it-py 4.2.0, mdurl 0.1.2, networkx 3.7,
pygments 2.21.0, rich 15.0.0. No external policy code has been copied into this
repository. The comparator is still unverified, and no new strength claim or
default-policy promotion is supported.

The first adapter diagnostic now passes on eight boards (seeds
1400030–1400037, reserved for diagnostics), covering 128 depth-two Catanatron
setup decisions in 1.468 seconds. `catanatron_adapter.py` maps the 54 vertices,
72 edges, 19 terrain/token locations, and nine exact port edges. The verifier
compares legal actions at each setup step, seat order, starting resources,
public points, and port trade rates. It drives both engines with the external
search's chosen actions and ends at the first regular roll.
Receipt: `runs/external-setup-v1.json`. Reproduce with a new output path:

```sh
PYTHONPATH=src runs/external/catanatron-env/bin/python scripts/verify_external_setup.py --output runs/external-setup-NEW.json
```

The check verifies the pinned checkout revision and installed search-module
bytes. Full dependency verification remains part of the eventual comparator
gate. This diagnostic uses only public setup positions; it does not authorize
passing true hidden state to search later in a game. Full-game state transitions,
observation-derived worlds, trade/robber/development mappings, and enforced
deadlines remain unfinished. The external comparator remains **unverified**.

Action mapping now includes bank trades and the four playable development cards.
`scripts/verify_external_actions.py` checks synthetic one-action transitions,
including resource-bank scarcity and pre-roll card use. All 216 cases passed
(`runs/external-actions-v2.json`); this does not cover complete robber/trade
protocols or award edge cases.

`from_observation` constructs regular-turn external states through the existing
belief sampler, rejects full engine-state input, and requires exact agreement
with observed root legal actions. The first full-trajectory probe on diagnostic
seed 1400050 produced 318 successful translations out of 321 opportunities and
12 legal external search decisions. Hidden-state perturbations left the sampled
external state unchanged. Three opportunities failed explicitly because the
128-particle belief tracker exhausted its feasible population (turn 39 seat 3;
turn 41 seat 1 before and after rolling). The Tactical-driven diagnostic game
completed at turn 82; this is not an external-policy matchup.
Receipt: `runs/external-observations-v1.json`.

Next bounded belief diagnostic, declared before execution: use 512 particles
only in the external adapter probe, with the same trajectory and RNG seeds.
Pass requires all 321 translations and privacy comparisons to succeed and every
probe (including its duplicate privacy reconstruction) to take under two seconds.
This is a conservative capacity screen, not a hard deadline implementation or
fresh strength test. Existing search policies retain their particle settings.

The 512-particle diagnostic **passed**: 321/321 translated states and privacy
checks, 12 legal external search decisions, maximum probe 0.1510s and p95 0.1210s.
Total probe time was 17.717s. Receipt:
`runs/external-observations-512-v1.json`. Reproduce with:

```sh
PYTHONPATH=src runs/external/catanatron-env/bin/python scripts/verify_external_observations.py --particles 512 --output runs/external-observations-512-NEW.json
```

This selects 512 particles for the next external integration DEV probe only.
One trajectory does not establish general belief reliability, and most states
in this diagnostic did not run search. Full-game external-policy decisions,
unsupported phase mappings and hard clock enforcement remain required.
Nine focused observation/belief unit tests also passed after this adapter work.

Next integration gate: run the adapted external policy in seat 0 against three
Tactical players on diagnostic seed 1400070, at most 300 turns. Pass requires
game completion, no fallback/error decisions, legal submitted actions, and all
observed external steps (observation through application) within two seconds.
This is a full-policy wiring probe, not a strength comparison. Deadline tests
use diagnostic seed 1400071. The policy uses 512 particles and depth-two stock
AlphaBeta, split robber/victim decisions, and external leaf evaluation of our
atomic bilateral accept/reject outcomes. Stock search does not support that
trade-response protocol. A parent process kills an unresponsive worker and
records the resulting legal fallback as a failure. The worker verifies the
installed external Python source against the pinned checkout and dependency
versions before searching. Tactical remains the default.

The first complete external-policy game passed (`runs/external-game-v1.json`):
81 turns, Tactical seat 1 won, zero external fallbacks/errors, maximum complete
external step 0.510s, and 3.695s total game time. External decisions covered
setup, roll, actions, robber, victim selection, discard, and 19 trade responses.
Two deadline/install-failure unit tests passed, including killing a deliberately
unresponsive worker. This is one wiring result, not a strength estimate.

Before expanding: next integration batch uses diagnostic boards 1400080–1400083,
all four external seats, serially on this host (one active search process).
Pass requires all 16 games complete within 300 turns, zero external fallbacks
or illegal actions, every external step at most two seconds, and at most ten
seconds accumulated external compute per game turn. The runner also requires
unchanged source and a verified external package identity. Failures stay in the
batch; no selective omission. Wins are descriptive only at this stage.

The seat-rotated integration batch **passed 16/16**
(`runs/external-integration-v1/summary.json`): no fallback decisions, maximum
external step 1.955s, maximum accumulated turn compute 2.040s, total game runtime
78.339s. The external agent won six games; this small diagnostic is not a
paired strength screen. The adapter is now available as policy name `external`
in the simulation factory. The regular simulator records external source identity
and per-decision failures separately from Monte Carlo rollout diagnostics.

The paired experiment reader now checks comparator evidence for every external
seat in both arms. Missing evidence or package drift is an error; fallbacks,
budget failures, or excess step/turn latency withhold the win-lift estimate and
prevent promotion. Regression bounds and sample-size suggestions are also
withheld for those invalid comparisons. Eleven focused runner, timing, deadline,
and missing-evidence tests passed. The remaining comparator work is the focused
rare-rule/protocol audit and a fresh reliability screen, before a challenger
can claim wins against a verified external baseline.

### Frozen economic-policy confirmatory family

Exact-candidate capacity **PASS**, `runs/economic-parameters-capacity-v1/summary.json`:
24 diagnostic games across serial/2/4 caps, unchanged outcomes, no faults or
truncations. Wave times 57.2171 / 31.3791 / 19.5767 seconds; cap4 speedup
2.92271. Maximum full decision 1.083604s, completed turn 1.421255s.

`economic-parameters-confirmatory-family-v1.json` predeclares four unopened
panels, 1600 board clusters each, four seats and two arms: 51,200 games.
Tactical-control panels use 1250000–1251599 and 1260000–1261599; external-control
panels use 1270000–1271599 and 1280000–1281599. Opponents match their passing
DEV panels. The model and Python gameplay source remain identical to DEV;
`runs/economic-parameters-confirmatory-source-v1` preserves that producer and
the runner/reader. Cython performance evidence remains a separate claim.

Every panel must have positive lift and positive normal lower bound at
alpha .0125 (Bonferroni family .05), zero truncations/faults and the existing
2s full-decision/10s completed-turn gates. There is no early success stop or
retuning on opened outcomes. Planning retains target +.05, assumed cluster
SD .5 and 1600 clusters from the existing joint calibration (power .857,
Wilson lower .833935). Largest observed DEV SD is .35904; that does not prove
the assumed variance bounds every future population. Synthetic null family
rejection .056 also means this is approximate inference, not a proof of exact
5% coverage. Per-run alpha .025 admission calibration additionally passed at
`runs/economic-parameters-run-calibration-v1.json`; its symmetric null rejection
was .039 under the diagnostic's loose tolerance. The stricter four-comparison
reader determines the final claim.

Capacity extrapolation at the mixed-external workload is about 34.8 hours for
51,200 games, before overhead and workload variation. DEV result bytes scale
to about 3.77 GB for aggregate JSON, plus retained shards/journals; host free
disk was about 20 GiB and physical memory 16 GiB. The terminal reader now loads
one run at a time and hashes incrementally, preserving raw-summary validation.
31 reader/parallel-runner tests passed, including the real terminal CLI on a
synthetic family and overwrite refusal. These fixtures are not scientific
outcomes. A resource-scale aggregation/reader rehearsal and detached resumable
supervision remain required before launch. No held-out games have started.

Full-scale resource rehearsal completed using repeated DEV payloads under
explicit `SYNTHETIC-*` names and diagnostic seeds 900000000+, without playing
held-out boards. `scripts/rehearse_confirmatory_capacity.py` exercised the real
shard assembly and summary function at 1600 clusters/panel; 69.34s, peak RSS
2,267,021,312 bytes. Real final-reader CLI completed in 26.44s with peak RSS
5,691,752,448 bytes and zero swaps. Output totals were 3,773,361,149 bytes.
The synthetic readout's PASS is only a terminal-path test, never strength
or independent-outcome evidence. Rehearsal artifacts remain preserved under
`runs/confirmatory-resource-rehearsal-v1`. 33 focused reader, supervisor and
parallel-runner tests passed, including resume skipping completed results,
terminal execution and overwrite refusal.

The frozen held-out family is now launched via detached supervisor PID 64534
on Jerrys-Mac-mini.local. Its live supervisor, child runner and four match
processes were verified after dispatch. Launch command and paths are recorded
in `runs/economic-parameters-confirmatory-supervisor-v1/launch.json`; progress
is in its `supervisor.log`. Tactical comparison runs first; external comparison
then runs with the same cap4 and final family reader runs only after both
complete. Completed shards and pair journals survive a supervisor restart;
resume the exact command in launch.json, with the frozen PYTHONPATH, Python
backend and original absolute parameter-model path. No concurrent heavy work
is scheduled on this host. Source/config/model/inference remain frozen; the
prelaunch plan's status prose is historical and is not rewritten mid-run.
No interim outcome-based tuning or early success stop is allowed. Tactical
remains default; neither held-out success nor human transfer is yet established.

Human-pilot evidence boundary: `scripts/audit_human_pilot.py` now consumes one
explicit private room record using its frozen source, replays legal actions and
recomputes bot choices, validates terminal metadata, consent, experience fields
and finite nonnegative recorded times, and produces a report without participant
names or private state. Seven focused tests passed, including synthetic actor
labels that must NOT become verified human evidence, action/source/terminal
mismatch, false consent and invalid timing. Only scripts/tests/docs changed;
the running held-out producer remains frozen. The private room server on port
8002 returned HTTP 200, and its isolated room directory contained zero records.
Actual participant play, randomized candidate/control evaluation and eventual
authorized Colonist integration are still outstanding.
