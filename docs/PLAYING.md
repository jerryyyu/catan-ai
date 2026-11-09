# Playing, pilots and audits

Operating notes for the local web player. Moved from the README (2026-11-08); content unchanged.

> Statements below that Tactical is the default predate 2026-11-03; the served bot (now `planner_pv2_tiles`) is listed in [POLICIES.md](POLICIES.md).


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

Solo play defaults to **`planner_pv2_tiles`** (served since 2026-11-08 by owner
promotion): the trained planner below with its own-turn build decisions chosen
by PV search, each candidate afterstate scored by a tile-token value transformer
served in NumPy. Its weights are a local file that is never committed. To serve
it, place the file and check it:

```sh
cp /path/to/value_tiles_v1.npz checkpoints/value_tiles_v1.npz
shasum -a 256 checkpoints/value_tiles_v1.npz   # must equal npz_sha256 in checkpoints/value_tiles_v1.meta.json
uv run python -c "from catan_ai.api import bots_info; print(bots_info())"
```

The last line must print `'default': 'planner_pv2_tiles'`. The server verifies
the file once per file version (SHA256 against the committed metadata, tokenizer
layout, and a parity probe against the exported torch logits). If NumPy or the
weights are missing or fail a check, new games use `planner_offer_accept` and
`GET /api/bots` gives `fallback_reason` "Tiles value weights missing or
unverified" with the loader's error in `fallback_detail` (paths cut to their
basename; the full error is in the server log). A game records the
weights' SHA256 and a restart refuses the game if they changed.

The fallback and previous default is the **Trained planner** (`planner_offer_accept`: the fitted
`planner_parameter` build search, with robber, steal, discard,
development-card and trade-answer moves chosen by one shared position score,
plus at most one player trade offer per turn, never to a seat on 8+ points and
steered away from players who keep refusing; served 2026-11-03 to 2026-11-08). Every served bot and what it measured: [docs/POLICIES.md](docs/POLICIES.md). The server loads `checkpoints/planner_parameter.json` unless
`CATAN_PARAMETER_MODEL` names another checkpoint. A game records the checkpoint's
SHA256 for restart recovery, and all three opponents must load the same
checkpoint. The checkpoint pins every source file and the engine backend. It is
bound to the default pure-Python engine, so under `CATAN_ENGINE=cython`, or with
any unbound source change, `GET /api/bots` reports it unavailable and new games
fall back to Tactical with a visible notice. After changing `src/catan_ai`, run

```sh
uv run python scripts/bind_parameter_checkpoint.py --write
```

It keeps the weights, replays the checkpoint's 8 seeded compatibility games and
writes the new binding only if every trace, outcome and final RNG state matches
the evaluated snapshot's games exactly. This is bounded regression evidence: it
catches changes that alter those games, but cannot prove that every future
action is unchanged. If they differ, the change alters how the promoted bot
plays and needs a new evaluation, not a rebind. CI enforces both
checks (`tests/test_served_checkpoint.py`). Solo games support usability
feedback; they do not measure performance against experienced humans.

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

## Auditing a three-human pilot record

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

## Learned baseline (experimental)

To try the unpromoted learned checkpoint in the web player's **Learned baseline (experimental)** option:

```bash
CATAN_MODEL_PATH=runs/learned-baseline-v1.json uv run catan-server
```

Missing or invalid models fall back to Tactical, with a notice in the web player.

`npm run test:e2e` uses installed Google Chrome and a local API on port 8013 to exercise setup and 45 human decisions plus controlled special-action fixtures. Fixture endpoints exist only in the test server, never the production app.
