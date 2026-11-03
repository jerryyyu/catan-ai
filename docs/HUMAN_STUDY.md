# Human-transfer study: draft protocol

Status: design for review, not a registered study or permission to recruit,
launch, deploy, or promote. No participant results are available as of
2026-11-03. Jerry offered to be the first pilot participant. Catanatron was
retired as a promotion gate by Jerry; this protocol does not reopen it.

## Questions and separate claims

1. Can a person complete a local match, recover after a server restart, and
   inspect a replay without private opponent information leaking?
2. Does a frozen candidate win more often than a frozen control against three
   experienced humans, at the declared compute budget?
3. Does the candidate exceed a 25% win rate at those tables?

A solo game against three bots answers usability questions, not questions 2
or 3. A three-human pilot is descriptive until assignment and analysis are
registered. Local play does not establish performance on Colonist or authorize
any integration with that service. Passing question 2 does not imply passing
question 3, or vice versa.

## Stage A: usability and instrumentation pilot

Start with Jerry in the existing solo pilot. Record consent, self-reported
experience, seed, policy/checkpoint identity, actions, completion and feedback.
Confirm actual participation separately from actor labels in the replay.

Before inviting a three-person group, verify the room route supports:

- An independently chosen seat credential per participant; private observations
  and no access to another seat's hand or credential through APIs or replay UI.
- Exact source, checkpoint and backend identities, persisted before first action.
- Restart recovery without changed bot decisions or lost accepted actions.
- Timings covering observation, belief, policy, validation and deterministic
  fallback, with bot compute accumulated across every action of an own turn.
  Human thinking and network waiting are excluded from compute time.
- Explicit termination reasons: winner, human withdrawal, disconnect, server
  fault, truncation, or budget violation. Preserve incomplete games.

Pilot exit bar: one completed solo participant game and one completed
three-human participant game with successful independent replay audit; no
known private-state leak, lost action, unlogged fallback or unidentified policy.
This is an operational gate, not a strength test. Failures require fixes and
new pilot records; original records remain unchanged. A single successful game
does not establish general reliability.

The current `scripts/audit_human_pilot.py` validates seeded actions and recorded
step times, but explicitly does not establish randomization, actual human
participation, or complete turn-time coverage. Extend those capabilities in a
separately claimed implementation before Stage B. Do not relabel existing
pilot output as confirmatory evidence.

## Stage B: registered candidate/control comparison

Recruit adults who consent to local replay collection and report at least 50
completed base-game matches. Record recruitment source and experience before
assignment. The experience threshold is an eligibility rule, not independently
verified skill. Describe the recruited population in the eventual claim.

Use four-player base-game rules implemented by the authoritative Python engine.
Freeze an explicit rules/exclusions manifest, source revision, backend identity,
checkpoint hashes and both policy names. Choose candidate and control before
any study outcome is opened; neither may change during the registered study.
Do not assume the newest bot or current default is the registered candidate.

Each three-human group plays a balanced block of eight games: each policy once
in each of the four bot seats, with human seats rotated. Generate the complete
assignment schedule from a committed seed; randomize order within each block.
Use a distinct previously unopened board seed for every game. Do not replay the
same board to the same humans: memory would contaminate a paired comparison.
Keep the schedule, policy identities and future board seeds unavailable to
participants until the block closes. Show a neutral opponent label.

Pair comparisons within human group, not within repeated boards. A returning
participant must stay in the same group; repeated blocks for that group remain
one statistical cluster. Record protocol deviations and do not count repeated
blocks as independent groups. If recruitment cannot supply the registered
number of independent groups, report an incomplete study rather than substitute
bot games or assume independence.

Primary strength estimand: average group-level difference in bot win fraction
(candidate minus control) over the registered schedule. Secondary estimand:
candidate win fraction minus .25. Equal group weighting is primary; report
per-game counts and seat breakdowns descriptively. Report both, even when one
fails. For formal success require simultaneous one-sided lower bounds above
zero for both estimands, using the frozen analysis and multiplicity correction.
The zero primary bar is superiority, not a minimum practically useful gain.

Before launch, create an executable registration containing:

- Exact group count, blocks per group, maximum games, seed schedule commitment,
  recruitment stop rule and calendar limit. No optional stopping on wins.
- `power.alternative_effect`, `planning_sigma`, target power at least .80,
  alpha allocation across both claims and the numerical success thresholds.
  Size by independent groups, accounting for within-group and participant
  correlation. An alternative must exceed its gate.
- Analysis code/source hashes, exact interval method and deterministic analysis
  seed. Simulate group-correlated nulls, dropout and alternatives to check
  calibration and power at that exact design before participant outcomes open.
  The bot evaluator's independent-board power calculation is not sufficient.
- A missing-outcome rule: preserve all assigned and started games; primary
  conservative bounds treat missing candidate outcomes as losses and missing
  control outcomes as wins. Report observed-game estimates separately. Never
  silently replace a difficult board, a loss or a timeout.
- Latency gates of 2 seconds per bot decision and 10 seconds of bot compute per
  own turn, on a named host/core limit, with zero budget failures or illegal
  actions for a correctness/latency PASS. A legal fallback keeps a match moving
  but still fails the relevant gate. Report strength independently of that fail.
- A maximum-turn rule consistent across arms, explicit truncation handling,
  participant confirmation procedure, privacy checks and artifact retention.

No sample count is declared here: recruitment capacity and defensible group
variance are unknown. Stage A informs feasibility and instrument repairs, not
policy selection on confirmatory outcomes. Stage B stays closed until the
complete numerical registration passes independent review and Jerry authorizes
launch. Under-recruitment is a dependency, not evidence of success.

## Results and custody

Keep private records under a dedicated new `runs/` directory. Publish only
aggregated results and non-identifying identifiers; exclude credentials and
private hands from player-facing artifacts. Store consent/participation
confirmation separately from public summaries.

Persist immutable assignment/config/source hashes, complete per-game outcomes,
all truncations and failures, uncertainty intervals, elapsed study time and
compute time. Audit replays with the corresponding frozen source. Publish
planned and actual counts, deviations, all gates and their results, and the
limitations of the participant population. Archive old sources and receipts;
corrections to documentation do not invalidate or overwrite games.
