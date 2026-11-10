# Engine, search and validation notes

Moved from the README (2026-11-08); content unchanged. Results for every comparison: [experiments/README.md](../experiments/README.md); served bots: [POLICIES.md](POLICIES.md).

> Statements below that Tactical is the default predate 2026-11-03; the served bot is listed in [POLICIES.md](POLICIES.md).


## Rules coverage

The engine implements snake-order setup, finite resource and piece supplies,
production, selected discards, robber movement and victim selection, roads,
settlements, cities, ports, maritime trading, and player trade offers with resource
bundles. Development cards include knights, Monopoly, Year of Plenty, Road Building,
and private victory points. Longest Road and Largest Army contribute to victory.
The browser exposes these actions for solo play and private rooms.

Player trades target one partner at a time; rejected offers can be revised and sent
again. Bots currently offer simple one-for-one exchanges at most once per turn.
The API stores games in process memory and has no multiplayer accounts or durable
reconnection. Board generation randomizes terrain, ports, and number tokens while
keeping 6s and 8s nonadjacent, as required for random setup in the
[base-game almanac](https://www.catan.com/sites/default/files/2021-07/catan-25th-rules_eng-200313.pdf).
Exact parity with Colonist's board distribution is not yet established.
Colonist ranked four-player base game is the competitive target. The local game
does not enforce its 60-second turn clock or connect to Colonist matches. Current
bots include handwritten baselines, opt-in Monte Carlo rollout policies, and a
small trained linear policy-ranking / win-value baseline. Tactical remains the
default: the newer search and learned candidates failed their DEV strength gates.
Competitive online human playing strength has not been established.
Expansions are outside this base-game project.

## Privacy

The server never sends opponents' resource cards. Seeds and complete state remain
available to local simulations but are absent from player-facing API views.
Development-card identities and hidden victory points are also withheld from other
players until the game ends (card identities remain private in the current view).

## Base-game implementation validation

The base-action milestone passed 54 Python tests and 12 Chrome browser tests.
A fresh simulation check (`--games 100 --seed 910000`) completed all 100 games
within the 2,000-turn ceiling. Completion and wins against random bots are not
evidence of competitive strength.

| Requirement | Evidence |
|---|---|
| Development purchase, timing, four playable card types, private VP | `test_development.py`, `test_rules_edges.py`, browser card and purchase scenarios |
| Ports, maritime trades, player bundles and responses | `test_ports.py`, `test_trades.py`, browser port/offer/acceptance scenarios |
| Selected discards, robber movement and victims | `test_robber.py`, browser discard/victim and live-game scenarios |
| Finite bank, shortage exception, piece limits | `test_supplies.py`, `test_rules_edges.py`, full-game conservation checks |
| Longest Road, Largest Army, on-turn victory | `test_scoring.py`, `test_rules_edges.py`, browser city-to-victory scenario |
| Private bot observations and deterministic simulation | `test_observation.py`, `test_integration.py`, `test_sim.py` |
| Experiment integrity | `test_experiments.py`: complete seed clusters, no truncated strength claims, source hashes |

The rules review used the linked official base-game almanac. Remaining platform
differences include targeted sequential trade offers rather than a broadcast
negotiation interface, local seat zero always starting (experiments rotate policies
through all seats), and no turn clock. Random number placement obeys the red-token
constraint but is not claimed to reproduce Colonist's exact map distribution.
The research placement candidate `strategic` failed its DEV bar and remains parked;
it was not promoted. The later tactical promotion is described below.
See `experiments/README.md` for the recorded comparisons.

## Monte Carlo search

Select **Monte Carlo (experimental, slower)** on the web start screen, or use
policy name `mc` in simulations. The API accepts `opponent: "mc"` when creating
a game. The web/API default is now `tactical`; `heuristic` remains the frozen comparator.

`belief.py` maintains 128 particles from viewer-filtered action history. Public
production, costs, trades, card counts, development plays and victory status
constrain possible hands; steal identities and purchased cards are known only to
the entitled players. Sampled worlds use fresh RNG and shuffled remaining decks,
never the authoritative engine RNG, opponent hands or deck order. Private-card
permutations, event redaction, conservation, replay and sampled legality have tests.

`search.py` compares up to five root actions, including the baseline choice, with
four sampled full-game heuristic continuations per action. It selects one
challenger and uses 32 fresh paired simulations to confirm it. A one-sided
Hoeffding lower bound on win-probability improvement must exceed zero to override
the baseline. Confirmation uses independent draws from the frozen approximate
particle belief; the bound covers rollout sampling noise conditional on that
belief and opponent model, not belief accuracy or real opponent strength.

Search currently covers ordinary action turns. Setup, pre-roll cards, robber,
discard, free-road and trade-response decisions retain heuristic behavior.
Maritime trades and a baseline-proposed player offer can enter the root shortlist;
arbitrary negotiation sequences are not exhaustively searched. Future decisions
use private observations for each rollout actor. A 1,500-decision rollout ceiling,
any incomplete sample, or exhausted belief population triggers explicit fallback.
Per-decision records include action values, confirmation differences, fallback
reason, rollout counts, truncation and latency; simulation and experiment results
preserve these records.

Limitations: uniform-card hidden-discard proposals, finite-particle impoverishment,
heuristic opponents, a small action shortlist and conservative sample budgets.
The method is root rollout search, not information-set tree search. Its current
heuristic continuations cannot anticipate skilled negotiation. See
`experiments/README.md` for numerical gates and measured promotion status.

The earlier search integration passed 74 Python tests and 13 Chrome browser tests. Its
first DEV comparison completed all 32 paired candidate/control matches but failed
the overall promotion gate: no win-rate gain against three heuristics, and search
p95 of 2.155 seconds exceeded the two-second budget. The small mixed-opponent
panel gained one win in eight games; that is exploratory signal, not a strength
claim. MC remains opt-in, and the conditional held-out run was not launched.
Debugging subsequently repaired maritime-trade shortlist ordering and unnecessary
equal-weight belief resampling. The original results are preserved with their
source snapshot; the repaired policy has not yet had a strength screen.

## Tactical and exploratory search policies

The `tactical` policy searches short, deterministic build sequences from public
geometry and its own hand. It handles maritime trade → build, Year of Plenty →
build, and road/award sequences within three actions and 128 nodes. Unknown
opponent hands and bank inventories are not supplied to this planner. Only a
known-legal root action may acquire resources; later actions are builds. It
replans after every real action. Twenty winning-turn fixtures cover the first
two acquisition paths, alongside privacy and legality tests.

`mc_mean` retains the original search budget and fresh confirmation samples but
selects a challenger on positive estimated improvement, without claiming that
its per-move LCB is positive. It is a separate experimental policy. Both policies
are selectable in the web player and available to `catan-sim`. The combined
`mc_tactical` research policy uses tactical own-player continuations; its first
cost check exceeded the two-second gate and it is not offered in the web selector.

```bash
uv run python -m catan_ai.search_eval --config experiments/tactical-dev.json --output runs/tactical-dev-v1.json
uv run python -m catan_ai.search_eval --config experiments/mc-mean-dev.json --output runs/mc-mean-dev-v1.json
uv run python scripts/calibrate_inference.py --output runs/inference-calibration-repeat.json
```

The evaluator now supports named candidates and per-player decision timing.
It retains Hoeffding reporting by default and offers explicitly approximate
paired seed-level normal inference for a separately preregistered held-out trial.
The approximate method requires at least 128 seeds, a variance floor, a matching
calibration result and disjoint DEV/held-out seeds. It estimates a held-out sample
size from DEV variability instead of treating 128 seeds as universally sufficient.
See `experiments/README.md` for thresholds, assumptions and results. The current
implementation passed 100 Python tests and 13 browser tests before default promotion; those are correctness
and integration checks, not proof of stronger play.

### Validated tactical default

On 155 reserved, previously unopened boards, rotating all four seats, Tactical won
228/620 games (36.77%) against three heuristic opponents. The paired original
heuristic control won 155/620 (25%). The observed gain is +11.77 percentage points;
the preregistered approximate one-sided 95% seed-cluster lower bound is +8.47 points.
All 1,240 candidate/control games completed, and policy-decision p95 was 2.36 ms
(excluding observation construction and HTTP overhead). The held-out gate passed.
Tactical is consequently the default for new web/API games. The original heuristic
remains separately selectable and unchanged as the experiment comparator.

Mean-search gained +6.25 points in DEV but failed its 2-second latency gate
(2.35-second search p95), so it remains experimental. The combined tactical rollout
policy failed its initial cost gate. These results establish improvement over this
heuristic on our engine; they do not establish competitive Colonist strength.
Full receipts, statistical assumptions and source lineage are recorded in
`experiments/README.md`. Post-promotion validation passes 101 Python tests and 13
browser tests. No external deployment was performed.
## Optional Cython engine

The Python rules and road-scoring sources also compile to native extensions:

```sh
uv sync --group native
uv run --group native python scripts/build_native.py build_ext --inplace
CATAN_ENGINE=cython uv run pytest -q
CATAN_ENGINE=cython uv run catan-server
```

`CATAN_ENGINE=python` is the default reference backend. Forced `cython` refuses
missing or stale builds; rebuild after changing engine or scoring sources.
`CATAN_ENGINE=auto` allows fallback to Python. Native binaries are local build
products, excluded from portable wheels. Experiment identities include the
selected backend and native binary hashes. Compare real search workloads with
`uv run python scripts/benchmark_native.py --output runs/native-check-NEW`.
