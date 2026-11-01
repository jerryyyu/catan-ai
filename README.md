# Catan AI

Unofficial research project; not affiliated with Catan GmbH or its publishers.
The UI uses original shapes and styling rather than commercial game assets.

See the [research notes](experiments/README.md#research-competitive-four-player-catan-ai)
for external Catan AI approaches, primary sources, evidence limits, and how they
inform the competitive-agent plan.

Current state and claim limits: [docs/CONTRACT.md](docs/CONTRACT.md). Plan:
[docs/PLAN.md](docs/PLAN.md). Receipt index: [docs/results/LADDER.md](docs/results/LADDER.md).

A small, reproducible research stack for a Catan-like four-player game:

- one authoritative, UI-free Python rules engine;
- heuristic, Tactical, strategic-style, search, and learned policies behind one bot interface;
- seeded headless simulation;
- paired, seat-rotated experiments with JSON receipts and confidence intervals;
- a React web table for solo play and private three-human/one-bot rooms.

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

## Run it

```bash
# terminal 1
uv sync --dev
uv run catan-server

# terminal 2
cd web
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api` to FastAPI on port
8000. A production web build is served by FastAPI when `web/dist` exists.

## Simulate and measure

```bash
uv run catan-sim --games 100 --seed 42
uv run catan-experiment --config experiments/base-actions-dev.json
uv run catan-sim --games 1 --seed 42 --policies mc heuristic heuristic heuristic
```

The first learned checkpoint is an unpromoted research baseline. To try the local
artifact in the web player's **Learned baseline (experimental)** option:

```bash
CATAN_MODEL_PATH=runs/learned-baseline-v1.json uv run catan-server
```

Missing or invalid models fall back to Tactical, with a notice in the web player.
Artifacts under `runs/` are local and ignored by Git. Generation, training,
prediction evaluation, gameplay results and reproduction commands are recorded
in [experiments/README.md](experiments/README.md).

Experiments write an immutable JSON result under `runs/` containing the config,
seed range, per-seat results, aggregate win rate, conservative seed-cluster 95%
interval, duration, source-file hashes, and the source revision when available.
Incomplete experiments report no aggregate win rate or interval and cannot pass.
A result receipt is evidence about that
exact comparison, not a deployment decision.

## Layout

```
src/catan_ai/engine.py       board, rules, legal actions, state transitions
src/catan_ai/bots.py         policy interface and baseline bots
src/catan_ai/sim.py          deterministic bot-v-bot game runner
src/catan_ai/experiments.py  paired seat-rotation experiment runner
src/catan_ai/api.py          per-player HTTP state and browser game endpoint
tests/                       engine, determinism, API, and experiment tests
web/                         React + TypeScript table
```

The server never sends opponents' resource cards. Seeds and complete state remain
available to local simulations but are absent from player-facing API views.
Development-card identities and hidden victory points are also withheld from other
players until the game ends (card identities remain private in the current view).

Run `uv run pytest -q` for rules, privacy, API, deterministic replay, and conservation
checks; run `npm run build` in `web/` for the production client build.
`npm run test:e2e` uses installed Google Chrome and a local API on port 8013 to
exercise setup and 45 human decisions plus controlled special-action fixtures.
Fixture endpoints exist only in the test server, never the production app.

Local pilot games are saved privately under `runs/pilots/` (override with
`CATAN_PILOT_DIR`). The browser remembers the latest match and resumes it on
refresh. Choose any of the four starting seats before creating a match; bots
place before you as needed, and hand, trading and replay views follow your seat.
Tactical and Heuristic matches also recover after a server restart by
replaying recorded actions and verifying bot decisions under the same source
version. Changed source or unsupported experimental-policy recovery returns a
clear error and preserves the record. Run one server process for local pilots.
Use **Download replay** for a human-view JSON record with observed events and
server action timings; opponents' hidden cards and the game seed are excluded.
Use **Review match** to step through recorded decisions on the board. Replay
navigation leaves the live game and its random state untouched; **Back to play**
returns to the current position. The replay endpoint accepts `?step=N`, where
zero is the initial empty board and each subsequent step applies one recorded
action, including bot actions.
The private server record includes the seed and full submitted actions and should
not be shared during play. Human timings measure action processing, not thinking
time; bot timings include observation, decision and application. These local
one-human/three-bot pilots support debugging, not a human-transfer strength claim.

Solo play also offers **Trained challenger (pilot)** against three fitted bots.
It requires a source-compatible `CATAN_PARAMETER_MODEL` on the server and records
its checkpoint SHA256 for restart recovery. All three opponents must load the
same checkpoint. This mode supports usability feedback; it does not measure
performance against three experienced human opponents.

Saved solo pilots can be checked without changing the record:

```sh
PYTHONPATH=/path/to/frozen/src CATAN_ENGINE=python \
  CATAN_PARAMETER_MODEL=/path/to/matching/model.json \
  uv run python scripts/audit_solo_pilot.py \
  --record /path/to/saved-game.json --output /path/to/new-audit.json
```

Use the match's frozen source and exact checkpoint. The audit replays bot
choices and checks legality, outcome, actor labels and recorded timings. It
reports a pilot result, not proof of human participation or experience, and
refuses to overwrite an existing output. Human action timings exclude thinking
time; unfinished turns can have incomplete timing totals.

For a private three-human match, open **Private three-human match** (or `/#room`),
choose the bot and its seat, and create a room. Tactical remains the default. Share one invitation with each
human, including yourself. Each invitation grants access to that seat; keep it
private. Players enter a display name, completed-match count, optional platform,
and consent to recording. Play starts when all three join. Refresh retains the
seat through the invitation URL. **Review match** shows that player's private
view at any recorded step without advancing the live game.

Room records use the same private storage directory, with `room_` filenames.
They retain source identity, participant self-reports, actions and bot compute
timings; only token hashes are stored. Run one server process. Restart recovery
requires matching source and reproduces the fixed bot decisions. The opt-in
**Trained challenger (pilot)** uses `planner_parameter`. Before starting the
server, set `CATAN_PARAMETER_MODEL` to a checkpoint bound to that exact server
source. A missing or incompatible checkpoint rejects room creation. The room
records its checkpoint SHA256; restart and offline audit require the identical
checkpoint, even if a replacement contains the same weights. Keep the frozen
server source and checkpoint together for replay. Source changes require a new
checkpoint binding and compatibility validation; do not overwrite old bindings.
The audit command needs the same `CATAN_PARAMETER_MODEL` environment variable.
These rooms are pilots: the UI does not randomize blinded candidate/control
assignments, and no experienced-human transfer result has yet been collected.

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

To audit one private three-human pilot record after play, run the offline
reader with the same source snapshot that served that room:

```sh
PYTHONPATH=runs/room-pilot-source-v1/src uv run python scripts/audit_human_pilot.py \
  --record runs/room-pilots-v1/room_ROOM_ID.json --output runs/pilot-audit-ROOM_ID.json
```

It validates legal replay, deterministic bot decisions, terminal metadata,
recorded timing values, consent and self-reported experience. Its report omits
names, invitations and private hands. A valid replay is not proof of actual
human participation: registration and actor labels can also come from automated
tests. Participant confirmation and the predeclared comparison remain separate
requirements. This audit does not infer full-turn latency from per-action times.
