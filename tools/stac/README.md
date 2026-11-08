# StacSettlers bridge

Lets the served Catan AI policy (`planner_offer_accept`) play classic 4-player StacSettlers games
inside STAC's own simulation harness (`soc.robot.stac.simulation.Simulation`: in-process server and
robot clients, no network), against STAC's agents: Dobre's POMCP-TS-CR and STAC's bundled JSettlers
1.1.16 robot. STAC tracks the game and stays authoritative; at each of our decisions the robot
serializes its own client-side copy of the game, the Python bridge rebuilds an engine `Game` /
`Observation`, asks the policy, and the robot sends the answer through the normal client API.

StacSettlers is GPLv3 (a JSettlers 1.1.16 fork; the MCTS jar is MIT). Nothing from its source is
copied here: the Java classes under `java/` only extend and call its classes, compiled against a
private build of the pinned commit plus a local match patch, both kept in `runs/external/`.

## Layout

| Path | Role |
|---|---|
| `java/catanai/stac/CatanAiStacFactory.java` | `SOCRobotFactory` named in a STAC config as `class:catanai.stac.CatanAiStacFactory`. |
| `java/catanai/stac/CatanAiStacBrain.java` | Extends STAC's `SOCRobotBrainImpl` (the JSettlers 1.1.16 brain); every decision hook asks the bridge, STAC's own logic is the counted fallback. |
| `java/catanai/stac/StacStateJson.java` | Bot-side snapshot, board and STAC's legal sets in the JSettlers bridge format. |
| `../jsettlers/java/catanai/jsettlers/PythonBridge.java`, `DecisionLog.java` | Reused unchanged (compiled into the STAC build): one bounded Python process per JVM, JSON-lines log. |
| `stacbridge/decide.py` | `StacBridge`: `jsbridge.decide.Bridge` with player offers disabled and a 10-point check. |
| `stacbridge/server.py`, `../../scripts/stac_bridge.py` | `jsbridge.server.serve` with `StacBridge`. |
| `fixtures/snapshots.jsonl` | Snapshots (+ replies) recorded from real STAC bridge games, every STAC decision type. |

The Python reconstruction, policy call and translation are `tools/jsettlers/jsbridge` (board
mapping, seat rotation, synthetic history, legality cross-check); the STAC serializer emits the same
snapshot format, translating the two codes that differ between JSettlers 1.1.16 and 2.x (desert hex
0 -> 6, knight card 0 -> 9).

## Commands

```bash
uv run python scripts/stac_build.py --config experiments/stac-smoke.json     # private verified build
uv run --with pytest --with numpy python -m pytest -q tests/test_stac_*.py
uv run python scripts/stac_match.py --config experiments/stac-smoke.json --out runs/stac-smoke-<stamp>
uv run python scripts/stac_match.py --config experiments/stac-determinism.json --out runs/stac-det-a
uv run python scripts/stac_match.py --config experiments/stac-determinism.json --out runs/stac-det-b
uv run python scripts/stac_match.py --compare runs/stac-det-a/receipt.json runs/stac-det-b/receipt.json
```

Prerequisites under `runs/external/` (never committed): the `ruflab/StacSettlers` checkout at
`c9e3b66f69e796b1cdd56a508ec32a6ceed1141d` (clean), the match patch named in the config with its
SHA-256, the 36 Maven dependency jars in `stac-deps/` (SHA-256 pinned in `scripts/stac_build.py`),
the JDK 17.0.21+9 and the Gson jar used by the JSettlers bridge.

## Build identity

`scripts/stac_build.py` stages everything a run loads into a new private directory inside the run
(it refuses an existing one, so no rebuild can change a class path a live JVM reads): the pinned
commit's `src/`, `lib/`, `pom.xml` and human action-type file are exported from the commit object
(`git archive`, not the working tree), the declared patch (exact SHA-256) is applied, the sources are
compiled with `javac --release 8`, and a second compile in a temporary directory must reproduce every
loaded `.class` byte for byte, with every resource equal to its pinned source. The `lib/` jars, Maven
jars, Gson jar and the action-type file are checked against pinned SHA-256s; our classes are built
against the staged STAC classes. `manifest.json` / `receipt.json` record the commit, patch SHA-256,
per-file hashes and digests; `verify_loaded` re-hashes all of it right before launch.

## The match patch (`runs/external/stac-catanai-v2.patch`)

Applied to the pinned commit; every change is opt-in or behaviour-neutral:

- **Build:** drops the `DeepCatan-0.0.1.jar` dependency (a git-lfs object GitHub no longer serves)
  and stubs the three neural-network seeding classes that needed it; NN-seeded agents fail fast.
- **Agent factory:** `Simulation` accepts `class:<fqcn>` as an agent type (a third-party
  `SOCRobotFactory`, loaded by reflection).
- **Seeding** (`soc.game.CatanAiSeed`, system property `stac.catanai.seed`; unset = upstream
  unseeded behaviour): per (base seed, game name, role) streams for the board layout and robot seat
  selection (robots are first put in name order, so connection order cannot matter), and one keyed
  source per event for the development deck, first player, dice roll *k*, steal *k* and forced
  discard *k*; JSettlers robot brains get (game, seat) streams and the robber strategy a (game,
  seat, turn, robber hex) stream. Derivation is FNV-1a 64 + SplitMix64, identical to the JSettlers2
  seeding patch (`scripts/jsettlers_smoke.derive_seed`).
- **No player trades** (`stac.catanai.noPlayerTrades=Y`): `SOCGame.canMakeTrade` refuses every
  player-to-player exchange and logs it; bank and port trades are unaffected.
- **Turn limit** (`stac.catanai.forceEndTurnSec`, default upstream 30 s): the server's robot
  inactivity limit, made configurable; every forced end is logged.
- **Accounting lines** (`catanai-*` on stdout): seed, setup (first player, seats, board, deck),
  observability flags per game, every dice roll with its index, every MCTS search (wall time,
  iterations, cap, whether the cap cut it), forced turn ends, blocked trades and the game end
  (winner, VP, turns).
- Two private brain fields used by initial placement are made protected for third-party brains.
- **MCTS time listener fix:** overrides `mcts.listeners.TimeListener` from `MCTS-1.0.jar` (MIT). Upstream,
  agents still running from an earlier time-capped search keep calling that search's listener, which
  then shuts down and notifies the *current* search, cutting it to a handful of iterations (seen in a
  first smoke as capped searches followed by searches of 17-800 iterations). The fixed listener fires
  once and waits for its own signal; the stopping rule (iterations or time) is unchanged. v1 of the
  patch (without this fix) was retired before any reported run.

## Rules audit: STAC (JSettlers 1.1.16) vs the engine

| Rule | STAC | Engine | Bridge handling |
|---|---|---|---|
| Board | Classic 19 hexes, 9 ports at fixed edges, shuffled terrain/numbers/port types | Standard base board | Mapped geometrically (`jsbridge.board`); refused unless the terrain mix, tokens and ports are standard |
| Setup | Snake order from a random first player; second settlement produces | Same, seat 0 first | Seats rotated so STAC's first player is engine seat 0 |
| Win | 10 VP (`getVpWinner`), checked on the owner's turn | 10 VP | Snapshot carries `vp_to_win`; `StacBridge` refuses anything but 10 |
| Development deck | 25: 14 knights, 5 VP, 2 each road building / year of plenty / monopoly | Same | Card codes translated (knight 0 -> 9) |
| Playing cards | One per turn, not the turn bought, any type before or after the roll; road building needs a road piece | Same (road building needs a legal road) | `canPlayKnight/RoadBuilding/Discovery/Monopoly` checked before sending; a refusal is a counted fallback |
| Robber | Any land hex but its current one (desert allowed) | Same | Engine hex vs STAC `canMoveRobber` cross-checked |
| Discard | More than 7 cards: discard half (rounded down) | Same | Exact discard set checked against the hand |
| Bank | **No finite resource supply** (production, year of plenty and bank trades never run out) | 19 cards per resource | Engine can only be *more* restrictive; opponents' placeholder hands come from STAC's totals |
| Bank/port trades | 4:1, 3:1 generic port, 2:1 resource port | Same | Engine rate vs STAC rate cross-checked per decision |
| Player trades | Off for these matches (see below) | Engine supports offers | Policy offers disabled; no trade answers exist |

## Hidden information

The snapshot is built from the robot client's own `SOCGame` copy, which is what the STAC server tells
that seat. The match runner refuses `FullyObservable`, `ObservableVP` or chat negotiation, and the
patched server logs all three per game (`catanai-observability`); a receipt with any game not logged
as hidden fails. In that mode the server sends other players' steals, discards and development-card
draws as UNKNOWN. Per opponent the snapshot carries pieces, `known` (resources the client saw it gain
or spend), `unknown`, `total`, development-card count, knights, public VP and whether it played a card
this turn; our own exact hand and cards go only in the `own` block. `jsbridge.state.check_hidden_boundary`
rejects anything else, and the tests run it on every recorded STAC snapshot, check that the engine
`Observation` exposes no opponent hand or card list, and check that `StacStateJson` writes no other
per-player key.

## Player trading is off

STAC's MCTS agents offer trades through chat negotiation that the JSettlers robots never answer
(their turns stall until the server forces them to end), so player trading is off in both directions:
`Trades=false` for STAC's JSettlers brains, `NO_TRADES` for the MCTS agents, `policy.offers = False` for
the served planner (it then ends the turn where it would have offered), and the patched server refuses
any player exchange and logs it. The receipt fails if STAC's results show an executed player trade.

## Decisions and fallbacks

| Decision | STAC hook | Bridge |
|---|---|---|
| Initial settlement (both rounds) | `placeFirstSettlement` / `placeSecondSettlement` | `setup_settlement`; `isPotentialSettlement` checked |
| Initial road | `placeInitRoad` | `setup_road`; must touch the new settlement |
| Roll or play a card first | `rollOrPlayKnightOrExpectDice` | `roll` |
| Turn actions | `getActionForPLAY1` | `actions`: build, buy, bank/port trade, play card, end turn |
| Free roads | `placeIfExpectPlacing` (PLACING_FREE_ROAD1/2) | `free_roads` |
| Robber, victim | `moveRobber`, `chooseRobberVictim` | `robber`, `steal` |
| Discard | `discard` | `discard` |

Any reply STAC would refuse (placement not potential, card not playable, victim not offered, discard
not held, more than 80 decisions or the same refused command 4 times in a turn) makes STAC's own brain
decide and is logged as `fallback` with its reason; bridge transport or exception failures are logged
as `cmd=error` (bridge errors) and also fall back. STAC's loop can enter an initial-placement state on
a stale turn flag (the state message arrives before the turn message); upstream then sends a placement
the server ignores. The bridge skips such a call without asking the policy and logs `out_of_turn_skip`.

## Timeouts

- Our bot: the bridge's per-request deadline (`bridge_timeout_ms`, default 30 s) and retirement rules
  are `PythonBridge`'s, unchanged from the JSettlers bridge.
- STAC MCTS agents: a declared per-search cap (`time_cap_ms`, passed as `MCTS_TIME_LIMIT_MS`; the
  search stops at 10k iterations or the cap). The cap is checked at the end of each iteration, so a
  search can overrun it by one iteration. Every search is logged with its iterations; a capped search
  counts as a cap hit.
- All robots: STAC's server ends a robot turn after `force_end_turn_sec` (30 s) without an action;
  every forced end is logged and counted (ours separately).

## Known limits

- STAC's MCTS agents are not reproducible even when seeded (4-thread tree parallelism and the MCTS
  jar's own random sources), so seeding pins the environment's luck (board, deck, first player, seats,
  dice by roll index, steals, forced discards), not their play.
- Seats come from the seeded shuffle, not a rotation; the receipt records seat and position counts.
- STAC's results files and the server's `catanai-end` line must agree on every winner, or the receipt
  fails.
