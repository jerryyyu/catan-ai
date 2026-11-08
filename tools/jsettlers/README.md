# JSettlers2 bridge

Lets the served Catan AI policy (`planner_offer_accept`) play classic 4-player games on a
[JSettlers2](https://github.com/jdmonin/JSettlers2) server as a third-party robot. JSettlers
tracks the game and stays authoritative; at each of our decisions the robot serializes its own
client-side copy of the game, a long-lived Python process rebuilds an engine `Game` /
`Observation`, asks the policy, and the robot sends the answer through the normal client API.

JSettlers2 is GPLv3. Nothing from its source is copied here: the Java classes under `java/`
only extend and call its classes, compiled against a local build in `runs/external/` (tag
`release-2.6.10`, commit `c5141056d8e3fff548817b3a1712eb66085e897b`).

## Layout

| Path | Role |
|---|---|
| `java/catanai/jsettlers/CatanAiClient.java` | `SOCRobotClient` subclass; classic 4-player only (no 6p/sea features). Counts public trade answers; logs game ends. |
| `java/catanai/jsettlers/CatanAiBrain.java` | `SOCRobotBrain` subclass; every decision hook asks the bridge. |
| `java/catanai/jsettlers/StateJson.java`, `BoardJson.java` | Bot-side snapshot and JSettlers' own legal sets, as JSON. |
| `java/catanai/jsettlers/PythonBridge.java` | One Python process per JVM, JSON lines on stdin/stdout. |
| `java/catanai/jsettlers/DecisionLog.java` | JSON-lines decision / fallback / game-end log. |
| `java/catanai/jsettlers/BoardDump.java` | Prints boards made by JSettlers' own generator (test fixture). |
| `jsbridge/board.py` | JSettlers v1 hex/node/edge coordinates <-> engine tile/vertex/edge indices. |
| `jsbridge/state.py` | Snapshot -> engine `Game` + `Observation` (seat rotation, synthetic history). |
| `jsbridge/decide.py` | Policy call, engine action -> JSettlers command, legality cross-check. |
| `jsbridge/server.py` | The stdin/stdout loop (`scripts/jsettlers_bridge.py` launches it). |
| `fixtures/boards.json` | 60 boards from JSettlers' generator. |
| `fixtures/snapshots.jsonl` | 38 snapshots (+ replies) recorded from real bridge games, every decision type. |

## Commands

```bash
uv run python scripts/jsettlers_build.py                 # javac -> runs/external/jsettlers-build/catanai-classes
uv run python scripts/jsettlers_build.py --dump-boards 60 > tools/jsettlers/fixtures/boards.json
uv run --with pytest --with numpy python -m pytest -q tests/test_jsettlers_*.py
uv run python scripts/jsettlers_smoke.py --config experiments/jsettlers-smoke.json
```

The smoke script starts `soc.server.SOCServer` with `jsettlers.bots.start3p=1,catanai.jsettlers.CatanAiClient`,
built-in robots, `percent3p=25` (one of our bot per game) and robot-only games
(`jsettlers.bots.botgames.*`), then writes `runs/jsettlers-smoke/<stamp>/receipt.json` from the
decision log. JSettlers properties are program arguments after the class name; the bridge's own
settings (`catanai.bridge.cmd`, `catanai.bridge.cwd`, `catanai.bridge.stderr`,
`catanai.decision.log`) are JVM `-D` flags.

## Run hygiene and identity

- **Sealed outputs.** `--out` (or the default stamped directory) must not exist; it is created
  exclusively before anything is built or launched. `build-manifest.json` and `manifest.json`
  (config bytes hash, source revision and dirty flag, verified build identity, bridge code hash,
  policy, served checkpoint `model_hash`, engine backend) are written with exclusive create before
  launch, and `receipt.json` is published the same way. A repeated command cannot mix runs or
  overwrite a receipt.
- **Verified binaries.** `jsettlers_build.build_manifest()` refuses to run unless the JSettlers
  checkout is at the pinned commit with no change under `src/`, a fresh `javac` of the pinned
  sources with the pinned JDK reproduces every loaded `.class` byte for byte (no missing or extra
  file), every other file on the class directory equals the checkout's `src/main/resources`
  copy, and the Gson jar has its pinned SHA-256. Our bridge classes are rebuilt into a clean
  directory. All of it is re-hashed right before launch (`verify_loaded`), and the config's
  declared tag/commit must equal the verified pin. `uv run python scripts/jsettlers_build.py --verify`
  runs the same check by hand.
- **Bounded bridge.** Each Java-to-Python request has a deadline (`catanai.bridge.timeout_ms`,
  default 30 s; config key `bridge_timeout_ms`). A missed deadline, a closed pipe or a garbled
  reply retires the transport: the Python process tree is destroyed, a `bridge_retired` record is
  logged, and every later request fails at once as a counted bridge error (with JSettlers'
  fallback so the server can finish the game). The bridge is not restarted, since a new process
  would lose per-game policy state; a run with a retired bridge fails its pass bar.
- **Process cleanup.** The server runs in its own process group (`start_new_session`); the
  group, including robot clients and the bridge, is killed on timeout and again after the
  server exits.

## Board mapping

JSettlers' classic board uses the v1 encoding: `0xAB`, two diagonal axes, centre hex `0x77`.
A hex centre is placed at `A*u1 + B*u2` (unit vectors at -60/+60 degrees); a node is the centroid
of the three hexes around it (`getAdjacentHexesToNode`). Of the twelve hex-lattice symmetries, the
one that puts every land hex on an engine tile centre is used, and nodes and edges follow by
position. Terrain, numbers, the robber hex and all nine ports (edge + type) are copied onto the
engine `Board`, and the result must be a standard base board (terrain mix, number tokens, 4 generic
and 5 specific coastal ports) or the bridge refuses it. The test maps 60 generator boards both
ways and checks every hex, node, edge and port.

## Seats

JSettlers picks a random first player; the engine's setup order is fixed `0,1,2,3,3,2,1,0` with
seat 0 first. Engine seat = `(JSettlers player number - first player) mod 4`, which also matches
JSettlers' clockwise turn order. The engine turn number is `SOCGame.getTurnCount() - 1`.

## Hidden information and opponents' hands

The snapshot is built from the robot client's own `SOCGame` copy, which is what the server tells
that seat. Per opponent it carries: pieces, `known` (the resources JSettlers' client has seen the
opponent gain and spend: dice production, builds, bank and player trades, discards and steals it
was party to), `unknown` (cards whose type it could not see, e.g. steals between two other
players), `total`, development-card count, played knights, public VP and whether it played a card
this turn. Our own exact hand and card list (with new/playable state) go only in a separate `own`
block. `state.check_hidden_boundary` rejects any snapshot with other per-player keys or with
`known + unknown != total`; tests run it on every recorded snapshot and check that the engine
`Observation` exposes no opponent hand or card list.

The served planner reads opponents' hands only through `card_planner.resource_estimates(view.events)`.
The bridge gives it a short synthetic history instead of a move list:

1. one `accept_trade` / `reject_trade` event per public answer each seat has given to an offer
   this game (feeds `AcceptanceOfferPlanner.acceptance_rates`), and one `buy_development` event per
   card bought from the deck (`25 - remaining`, feeds the planner's deck-exhausted check). These
   have `counts` all zero, so they leave every hand estimate at zero;
2. one `jsettlers_hands` event whose `resource_deltas` give each seat `known + unknown/5` per
   resource and whose `counts` are the true totals.

`resource_estimates` therefore returns exactly JSettlers' bot-side tracking, with the unseen cards
spread evenly over the five resources (tested on every fixture). Inside the engine `Game` the
opponents hold placeholder hands with the right totals (unknown cards on the most plentiful
resource); the `Observation` never exposes them, but they feed the engine's 19-card bank check (see
below). Unknown opponent development cards are placeholders the planner already treats as non-VP.

## Decisions

| Decision | JSettlers hook | Bridge |
|---|---|---|
| Initial settlement (both rounds) | `OpeningBuildStrategy.planInitialSettlements/planSecondSettlement` | `setup_settlement` |
| Initial road (both rounds) | `OpeningBuildStrategy.planInitRoad` | `setup_road` |
| Roll or play a card before rolling | `rollOrPlayKnightOrExpectDice` | `roll` (any card type, as in both rule sets) |
| Each main-phase action | `planAndDoActionForPLAY1` | `actions`: build road/settlement/city, buy card, bank or port trade (one card at the engine rate), knight, road building, year of plenty, monopoly, trade offer, end turn |
| Road Building's free roads | `placeIfExpectPlacing` (`PLACING_FREE_ROAD1/2`) | `free_roads` |
| Robber hex | `RobberStrategy.getBestRobberHex` | `robber` |
| Victim (only when JSettlers asks: more than one candidate) | `RobberStrategy.chooseRobberVictim` | `steal` |
| Discard | `DiscardStrategy.discard` | `discard` (engine's one-card steps run locally, sent as one set) |
| Offers made to us | `considerOffer` | `trade_response`: accept or reject, never counter |

Every bridged decision is logged with its JSettlers round-trip latency (snapshot, JSON, Python,
reply) and the policy's own think time. A **fallback** is a decision made by the inherited
JSettlers logic instead of the policy; it is logged with its reason and counted in the receipt.
Causes: the engine action names a target outside JSettlers' legal set (settlement/road/city
nodes and edges, port rates, robber hexes, victims), a one-card Year of Plenty, a discard set
JSettlers would refuse, a card JSettlers refused this turn, the same command refused four times in
a row, or more than 80 decisions in one turn (both end the turn). A **bridge error** is a
transport failure or Python exception; it also falls back so the game continues, and is counted
separately. Every decision also records any difference between the engine's legal set and
JSettlers' (`legal_mismatches`).

## Rules differences the bridge meets

| Topic | JSettlers classic 4p | Engine | Bridge handling |
|---|---|---|---|
| First player / setup order | Random first player; snake order from it | Seat 0 first, `0,1,2,3,3,2,1,0` | Seat rotation (above). |
| Trade offers | One offer to any subset of players; each may accept, reject, counter or ignore; open until cleared | One recipient, who must accept or reject | We offer to the single engine recipient. No answer within JSettlers' 5 s bot timeout ends the offer without a trade. A built-in robot's counter-offer arrives as a new offer to us and is answered as a `trade_response`. We never counter. |
| Offers per turn | Unlimited | Policy offers at most once per own turn | Unchanged policy behaviour. |
| Acceptance record | Accept/reject messages are public; some bots ignore offers | Every offer gets exactly one answer from its recipient | Acceptance rates count every public accept/reject by each seat to any offer; ignored offers count nothing. |
| Accepting | Requires holding the requested cards; executes at once | Same | None needed. |
| Development card timing | Not on the turn bought; one per turn; any type before or after the roll | Same | JSettlers' new/playable/kept states map to bought this turn / earlier. |
| Year of Plenty | Always two cards (no bank limit) | Two cards, or one if the 19-card bank holds only one | One-card answer is a fallback (never seen). |
| Road Building | Needs road pieces, not a legal edge; ends early if no edge | Needs a legal edge; ends early if none | Free roads asked one at a time; early end clears the brain's flags. |
| Victory point cards | Count toward 10 on your own turn | Same | None. |
| Resource bank | No finite supply in `SOCGame` | 19 per resource: production and bank trades limited by supply | Engine may omit a bank trade or Year of Plenty choice JSettlers would allow (a missed option, not a fallback). Opponent placeholder hands make the engine's supply approximate. |
| Bank / port trades | Any multiple in one request; undo allowed | One card at the best rate | One-card requests at the engine rate, checked against JSettlers' port flags. |
| Ports | 4 generic 3:1 + 5 resource 2:1 on fixed coastal edges, types shuffled | Same counts on coastal edges | Mapped by node; standard-board check. |
| Robber | Any other land hex incl. desert (option `RD` off by default) | Any other tile | JSettlers' legal hexes are cross-checked. |
| Steal | Victim prompt only when 2+ candidates; with one, the server steals | Explicit steal phase whenever there is a victim | Policy asked only when JSettlers asks. |
| Discard | Simultaneous; whole set at once | One card per action, seat by seat | Engine steps run locally on our copy, sent as one set. |
| Longest road / largest army | 5 roads / 3 knights; ties keep the holder | Same thresholds and tie rule | Holders copied from JSettlers, not recomputed. |
| Win | 10 VP (`VP` option default), only on own turn | 10 VP, checked for the turn owner | None. |
| Opponents' hidden cards | Client sees gains/spends; unseen steals become unknown | Engine events carry the same public/private split | Synthetic history (above). |

## Known limitations

- The server chooses our seat and the first player; the smoke receipt records both (and our
  position after the first player) per game but cannot rotate them.
- Latency is the JSettlers-side round trip, including snapshot serialization and JSON.
- `public_vp_client_copy` in the receipt is the bot's client copy when the game-over message
  arrives; robot-only games are destroyed immediately, so it can lag the final position. The
  winner comes from the server's current-player announcement sent just before `OVER`.
- Monopoly is translated and unit-tested but was not played in the recorded games.
