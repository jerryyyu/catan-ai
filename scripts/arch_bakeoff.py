"""Architecture bake-off on the value-v2 shards (experiments/arch-bakeoff.json).

Four arms trained with identical data, optimizer, schedule, augmentation and epoch
selection, then gated once on the test games' rows:

- ``base``: the value-v2-100k-aug-b residual MLP (built by ``train_value_v2.build_model``).
- ``heads``: same trunk; 4-way softmax over the winner's seat relative to me
  (cross-entropy, weight 1, P(I win) = p[me]) plus turns-remaining/100 (Huber,
  weight 0.1, auxiliary only). Targets come from the game records.
- ``tiles``: tile-token transformer (19 tile tokens = tile + 6 corner vertices +
  6 side edges in counterclockwise order from the tile centre, 4 seat-relative
  player tokens, 1 global token).
- ``cnn``: hex-grid CNN on 26 board planes (tile 8 + vertex 14 + edge 4 placed by
  quantized board coordinates), 6 residual 3x3 blocks of 64 channels, average and
  max pooling, concatenated with the player and global features into a 256 head.

The augmentation (one of the 12 hex symmetries per row per batch) is applied to
the flat standardized vector before any tokenizing or gridding, so every arm sees
the same augmentation. The standardization is symmetrized over the 12 symmetries
(as in ``train_value_v2``) and applied per batch on the device; ``x`` stays float16
in host memory. The epoch is chosen by the validation BCE of P(I win) on the last
``validation_games`` train games, never on test.

    arch_bakeoff.py train --config experiments/arch-bakeoff.json --arm base \
        --data runs/value-v2-100k-shards --records runs --out runs/arch-bakeoff
    arch_bakeoff.py gate --config experiments/arch-bakeoff.json \
        --data runs/value-v2-100k-shards --records runs --out runs/arch-bakeoff

``--max-rows`` / ``--epochs`` are smoke-test overrides; any output they touch is
marked ``smoke: true``.

Array binding (Codex HOLD @b48f61d, P1; same rule as ``train_value_v2`` after PR #30):
the shards are read through the verified loader ``train_value_v2.shards`` (inventory,
NPZ SHA-256, split and seed blocks). ``train`` stores ``train_arrays_sha256`` (the hash
of the verified NPZ digests) and ``train_sealed_after_the_fact`` in each checkpoint;
``gate`` re-verifies the train shards and refuses a checkpoint whose sidecar identity or
recorded array hash differs. A checkpoint written before this binding carries only the
sidecar identity: ``gate`` accepts it but labels it (``train_arrays_bound_in_checkpoint:
false`` and an entry in ``identity_limitations``), as it labels shards whose NPZ hashes
were sealed after the fact (``legacy_seal``).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_guard import publish, refuse_existing, sha256  # noqa: E402
from train_value import auc  # noqa: E402
from train_value_v2 import DATA_KEYS, build_model, make_optimizer, shards, symmetry_index  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARMS = ('base', 'heads', 'tiles', 'cnn')

# encoder-v2 layout (verified against catan_ai.encoding.size() in ``layout()``)
N_TILES, N_VERTICES, N_EDGES, N_PLAYERS = 19, 54, 72, 4
TILE_W, VERTEX_W, EDGE_W, PLAYER_W, GLOBAL_W = 8, 14, 4, 27, 20
V0 = N_TILES * TILE_W                     # 152
E0 = V0 + N_VERTICES * VERTEX_W           # 908
P0 = E0 + N_EDGES * EDGE_W                # 1196
G0 = P0 + N_PLAYERS * PLAYER_W            # 1304
WIDTH = G0 + GLOBAL_W                     # 1324
TURN_COL = G0 + 5 + 5                     # min(turn, 400) / 400
SEAT_COLS = slice(TURN_COL + 1, TURN_COL + 5)  # absolute-seat one-hot
TOKEN_W = TILE_W + 6 * VERTEX_W + 6 * EDGE_W   # 116
CHANNELS = TILE_W + VERTEX_W + EDGE_W     # 26

TILES_D, TILES_HEADS, TILES_MLP, TILES_LAYERS, TILES_DROPOUT = 128, 4, 256, 4, .1
CNN_CH, CNN_BLOCKS, CNN_HEAD = 64, 6, 256
HEADS_CE_W, HEADS_TURN_W = 1., .1
ECE_BINS = 10
EVAL_BATCH = 16384


def layout():
    from catan_ai.encoding import size
    if size() != WIDTH or TURN_COL != WIDTH - 10 or SEAT_COLS != slice(WIDTH - 9, WIDTH - 5):
        raise ValueError('encoder-v2 layout differs from the constants in arch_bakeoff.py')
    from catan_ai.engine import Game
    board = Game.new(list('abcd'), 0).board
    if (len(board.tiles), len(board.vertices), len(board.edges)) != (N_TILES, N_VERTICES, N_EDGES):
        raise ValueError('board topology differs from the encoder layout')
    return board


def _centre(board, tile):
    return (sum(board.vertices[v][0] for v in tile.vertices) / len(tile.vertices),
            sum(board.vertices[v][1] for v in tile.vertices) / len(tile.vertices))


def tile_geometry(board):
    """Per tile: corner vertex ids counterclockwise by angle from the tile centre (starting
    at the smallest angle in [0, 360)), and side k = edge between corners k and k+1."""
    eindex = {tuple(sorted(e)): i for i, e in enumerate(board.edges)}
    out = []
    for tile in board.tiles:
        cx, cy = _centre(board, tile)

        def angle(v, cx=cx, cy=cy):
            x, y = board.vertices[v]
            return round(math.degrees(math.atan2(y - cy, x - cx)) % 360, 3) % 360
        corners = sorted(tile.vertices, key=angle)
        if len(corners) != 6:
            raise ValueError('tile without six corners')
        sides = [eindex[tuple(sorted((corners[k], corners[(k + 1) % 6])))] for k in range(6)]
        out.append((corners, sides))
    return out


def tile_index(board):
    """[19, 116] flat feature indices: tile block, 6 corner vertex blocks, 6 side edge blocks."""
    rows = []
    for t, (corners, sides) in enumerate(tile_geometry(board)):
        row = list(range(t * TILE_W, (t + 1) * TILE_W))
        for v in corners:
            row += range(V0 + v * VERTEX_W, V0 + (v + 1) * VERTEX_W)
        for e in sides:
            row += range(E0 + e * EDGE_W, E0 + (e + 1) * EDGE_W)
        rows.append(row)
    return np.asarray(rows, dtype=np.int64)


def grid_index(board):
    """Board planes: tile centres, vertices and edge midpoints quantized onto a 2D grid
    (x step sqrt(3)/4, y step 1/4, i.e. exact for this board; empty rows/columns removed).
    Returns ``src`` [26*H*W] of flat feature indices (``WIDTH`` = the zero pad) and (H, W)."""
    items = []  # (x, y, first feature, width, first channel)
    for t, tile in enumerate(board.tiles):
        items.append((*_centre(board, tile), t * TILE_W, TILE_W, 0))
    for v, (x, y) in enumerate(board.vertices):
        items.append((x, y, V0 + v * VERTEX_W, VERTEX_W, TILE_W))
    for e, (a, b) in enumerate(board.edges):
        (xa, ya), (xb, yb) = board.vertices[a], board.vertices[b]
        items.append(((xa + xb) / 2, (ya + yb) / 2, E0 + e * EDGE_W, EDGE_W, TILE_W + VERTEX_W))
    qx, qy = math.sqrt(3) / 4, .25
    cells = []
    for x, y, *_ in items:
        cx, cy = x / qx, y / qy
        if abs(cx - round(cx)) > .02 or abs(cy - round(cy)) > .02:
            raise ValueError('board coordinate off the quantization lattice')
        cells.append((round(cy), round(cx)))
    if len(set(cells)) != len(cells):
        raise ValueError('two board items share a grid cell')
    rows = {r: i for i, r in enumerate(sorted({r for r, _ in cells}))}
    cols = {c: i for i, c in enumerate(sorted({c for _, c in cells}))}
    h, w = len(rows), len(cols)
    src = np.full(CHANNELS * h * w, WIDTH, dtype=np.int64)
    for (r, c), (_, _, first, width, channel) in zip(cells, items):
        for j in range(width):
            src[(channel + j) * h * w + rows[r] * w + cols[c]] = first + j
    return src, (h, w)


# ------------------------------------------------------------------- models
def base_spec():
    cfg = json.loads((ROOT / 'experiments' / 'value-v2-100k-aug-b.json').read_text())['model']
    return {k: cfg[k] for k in ('arch', 'hidden', 'blocks', 'dropout')}


def make_net(torch, arm, board=None, size=None, phase=False):
    """``size`` overrides the tiles arm's {d, heads, mlp, layers, dropout} (value-arms T7); ``phase`` adds a
    learned vector to the global token for phase-1 rows (value-arms S1). Defaults build the served net
    exactly (same parameters and state-dict keys)."""
    nn = torch.nn
    board = board if board is not None else layout()
    sz = {'d': TILES_D, 'heads': TILES_HEADS, 'mlp': TILES_MLP, 'layers': TILES_LAYERS, 'dropout': TILES_DROPOUT,
          **(size or {})}
    if (size or phase) and arm != 'tiles':
        raise ValueError('size and phase apply to the tiles arm only')

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = build_model(torch, WIDTH, base_spec())

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class Heads(nn.Module):
        def __init__(self):
            super().__init__()
            spec = base_spec()
            self.net = build_model(torch, WIDTH, spec)
            self.net.out = nn.Sequential(nn.LayerNorm(spec['hidden']), nn.Linear(spec['hidden'], N_PLAYERS + 1))

        def forward(self, x):
            out = self.net(x)
            return out[:, :N_PLAYERS], out[:, N_PLAYERS]

    class Tiles(nn.Module):
        def __init__(self):
            super().__init__()
            d = sz['d']
            self.register_buffer('tidx', torch.from_numpy(tile_index(board)))
            self.register_buffer('types', torch.tensor([0] * N_TILES + [1] * N_PLAYERS + [2]))
            self.tile, self.player, self.glob = nn.Linear(TOKEN_W, d), nn.Linear(PLAYER_W, d), nn.Linear(GLOBAL_W, d)
            self.pos = nn.Parameter(torch.randn(N_TILES + N_PLAYERS + 1, d) * .02)
            self.kind = nn.Parameter(torch.randn(3, d) * .02)
            layer = nn.TransformerEncoderLayer(d, sz['heads'], sz['mlp'], sz['dropout'], batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, sz['layers'], enable_nested_tensor=False)
            self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, 1)
            if phase:
                self.phase_vec = nn.Parameter(torch.zeros(d))

        def forward(self, x, phase=None):
            n = len(x)
            g = self.glob(x[:, G0:])
            if phase is not None:
                if not hasattr(self, 'phase_vec'):
                    raise ValueError('this net was built without a phase vector')
                g = g + phase.to(g.dtype).unsqueeze(1) * self.phase_vec
            h = torch.cat([self.tile(x[:, self.tidx]), self.player(x[:, P0:G0].reshape(n, N_PLAYERS, PLAYER_W)),
                           g.unsqueeze(1)], 1) + self.pos + self.kind[self.types]
            return self.head(self.norm(self.encoder(h)[:, -1])).squeeze(-1)

    class Res(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.f = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(),
                                   nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))

        def forward(self, x):
            return torch.relu(x + self.f(x))

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            src, self.shape = grid_index(board)
            self.register_buffer('src', torch.from_numpy(src))
            self.stem = nn.Sequential(nn.Conv2d(CHANNELS, CNN_CH, 3, padding=1, bias=False), nn.BatchNorm2d(CNN_CH),
                                      nn.ReLU())
            self.blocks = nn.Sequential(*[Res(CNN_CH) for _ in range(CNN_BLOCKS)])
            self.head = nn.Sequential(nn.Linear(2 * CNN_CH + WIDTH - P0, CNN_HEAD), nn.ReLU(), nn.Linear(CNN_HEAD, 1))

        def planes(self, x):
            pad = torch.cat([x, x.new_zeros(len(x), 1)], 1)
            return pad[:, self.src].reshape(len(x), CHANNELS, *self.shape)

        def forward(self, x):
            h = self.blocks(self.stem(self.planes(x)))
            return self.head(torch.cat([h.mean((2, 3)), h.amax((2, 3)), x[:, P0:]], 1)).squeeze(-1)

    return {'base': Base, 'heads': Heads, 'tiles': Tiles, 'cnn': CNN}[arm]()


def log_p_win(torch, arm, out):
    """(log P(I win), log P(I lose)) per row."""
    if arm == 'heads':
        logp = torch.log_softmax(out[0], 1)
        return logp[:, 0], torch.logsumexp(logp[:, 1:], 1)
    return torch.nn.functional.logsigmoid(out), torch.nn.functional.logsigmoid(-out)


def bce_sum(torch, arm, out, y):
    win, lose = log_p_win(torch, arm, out)
    return -(y * win + (1 - y) * lose).sum()


# --------------------------------------------------------------------- data
def manifests(root, split):
    files = sorted(Path(root).glob(f'{split}-*.npz'))
    if not files:
        raise ValueError(f'no {split} shards in {root}')
    texts = [f.with_suffix('.json').read_text() for f in files]
    sources = {}
    for text in texts:
        sources.update({int(k): v for k, v in json.loads(text)['source_sha256'].items()})
    return hashlib.sha256(''.join(texts).encode()).hexdigest(), sources


def declared_split(cfg):
    m = re.search(r'split train (\d+)\+(\d+) / test (\d+)\+(\d+)', cfg['data']['same_as'])
    if not m:
        raise ValueError('cannot read the split from data.same_as')
    a, b, c, d = map(int, m.groups())
    return {'train': [a, b], 'test': [c, d]}


def check_data(cfg, data):
    """The shard directory must be complete and made under the declared data keys."""
    ref = json.loads((ROOT / 'experiments' / f"{cfg['data']['same_as'].split()[0]}.json").read_text())
    done = Path(data) / 'DONE.json'
    made = json.loads(done.read_text())['config'] if done.exists() else None
    if made is None or any(made.get(k) != ref.get(k) for k in DATA_KEYS):
        raise ValueError('shards were not completed under the declared data keys')
    if made['split'] != declared_split(cfg):
        raise ValueError('shard split differs from the declared split')


def subsample(data, max_rows, seed):
    if not max_rows or max_rows >= len(data['y']):
        return data
    keep = np.sort(np.random.default_rng(seed).choice(len(data['y']), max_rows, replace=False))
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in data.items()}


def _record(job):
    seed, roots, expected = job
    for root in roots:
        path = Path(root) / 'games' / f'{seed}.json'
        if path.exists():
            raw = path.read_bytes()
            break
    else:
        raise ValueError(f'no game record for seed {seed}')
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'record {seed} differs from the shard manifest hash')
    record = json.loads(raw)
    if record.get('seed') != seed:
        raise ValueError(f'record {path} is not seed {seed}')
    return seed, record['winner'], record['turns']


def load_records(records, cfg, seeds, sources):
    """{seed: (winner or None, final turn)}, one read per game, hash-checked against the shards."""
    roots = [Path(records) / Path(r).name for r in cfg['data']['game_records']]
    jobs = [(int(s), roots, sources[int(s)]) for s in seeds]
    if len(jobs) > 2000:
        from multiprocessing import Pool
        with Pool(min(16, os.cpu_count() or 1)) as pool:
            results = pool.map(_record, jobs, chunksize=256)
    else:
        results = [_record(j) for j in jobs]
    return {s: (w, t) for s, w, t in results}


def head_targets(data, recs):
    """Relative winner (-1 if the game was truncated) and turns-left/100 per row; checks
    that the relative winner is 0 exactly on the rows labelled y = 1."""
    x = data['x']
    onehot = x[:, SEAT_COLS].astype(np.float32)
    if not np.all(onehot.sum(1) == 1):
        raise ValueError('absolute-seat one-hot is not one-hot on every row')
    seat = onehot.argmax(1)
    turn_enc = x[:, TURN_COL].astype(np.float32)
    turn = np.rint(turn_enc * 400)
    games, inverse = np.unique(data['game'], return_inverse=True)
    winner = np.array([-1 if recs[int(g)][0] is None else int(recs[int(g)][0]) for g in games])[inverse]
    turns = np.array([recs[int(g)][1] for g in games], dtype=np.float32)[inverse]
    known = winner >= 0
    rel = np.where(known, (winner - seat) % 4, -1)
    y = data['y'].astype(int)
    if not np.array_equal(rel[known] == 0, y[known] == 1) or np.any(y[~known] == 1):
        bad = int(np.sum((rel[known] == 0) != (y[known] == 1)) + np.sum(y[~known] == 1))
        raise ValueError(f'winner/label consistency check failed on {bad} rows')
    left = np.maximum(0, turns - turn) / 100
    counts = {'rows': len(y), 'truncated_rows': int((~known).sum()),
              'truncated_games': int(sum(recs[int(g)][0] is None for g in games)), 'games': len(games),
              'turn_capped_rows': int((turn_enc >= 1).sum())}
    return rel.astype(np.int64), left.astype(np.float32), counts


# -------------------------------------------------------------------- train
def standardization(torch, x, rows, perms):
    """Mean/variance of the fit rows (float64, chunked), symmetrized over the 12 symmetries."""
    n, s, ss = 0, np.zeros(x.shape[1]), np.zeros(x.shape[1])
    for lo in range(0, len(rows), 100_000):
        c = x[rows[lo:lo + 100_000]].astype(np.float64)
        n, s, ss = n + len(c), s + c.sum(0), ss + (c * c).sum(0)
    mean = s / n
    var = (ss - n * mean * mean) / max(1, n - 1)
    mean, var = torch.tensor(mean, dtype=torch.float32), torch.tensor(var, dtype=torch.float32)
    mean = torch.stack([mean[p] for p in perms]).mean(0)
    var = torch.stack([var[p] for p in perms]).mean(0)
    return mean, var.clamp_min(0).sqrt().clamp_min(1e-6)


def device_of(torch, name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def to_batch(torch, x, rows, mean, sd, device):
    return (torch.from_numpy(x[rows]).to(device).float() - mean) / sd


def score(torch, net, arm, x, rows, mean, sd, device):
    """P(I win) (float64) and, for heads, the 4-way log-probabilities, without augmentation."""
    net.eval()
    p, logp = [], []
    with torch.no_grad():
        for lo in range(0, len(rows), EVAL_BATCH):
            out = net(to_batch(torch, x, rows[lo:lo + EVAL_BATCH], mean, sd, device))
            p.append(log_p_win(torch, arm, out)[0].exp().cpu().double().numpy())
            if arm == 'heads':
                logp.append(torch.log_softmax(out[0], 1).cpu().double().numpy())
    return np.concatenate(p), (np.concatenate(logp) if logp else None)


def train(args, cfg):
    import torch
    common, arm = cfg['common'], args.arm
    epochs = args.epochs or common['epochs']
    torch.manual_seed(common['seed'])
    device = device_of(torch, args.device)
    board = layout()
    verified = shards(args.data, 'train')
    data = subsample(verified, args.max_rows, common['seed'])
    sources = manifests(args.data, 'train')[1]
    lo, n = declared_split(cfg)['train']
    if data['game'].min() < lo or data['game'].max() >= lo + n:
        raise ValueError('train shards hold games outside the declared train split')
    cut = lo + n - common['validation_games']
    fit_rows = np.flatnonzero(data['game'] < cut)
    val_rows = np.flatnonzero(data['game'] >= cut)
    if not len(fit_rows) or not len(val_rows):
        raise ValueError('empty fit or validation rows')
    x, y = data['x'], torch.from_numpy(data['y'].astype(np.float32))
    extra = {}
    if arm == 'heads':
        recs = load_records(args.records, cfg, np.unique(data['game']), sources)
        rel, left, counts = head_targets(data, recs)
        extra = {'counts': counts}
        rel, left = torch.from_numpy(rel), torch.from_numpy(left)
    perms_cpu = symmetry_index(torch)
    mean, sd = standardization(torch, x, fit_rows, perms_cpu)
    perms, dmean, dsd = perms_cpu.to(device), mean.to(device), sd.to(device)
    net = make_net(torch, arm, board).to(device)
    params = sum(p.numel() for p in net.parameters())
    batch = common['batch']
    steps = epochs * ((len(fit_rows) + batch - 1) // batch)
    opt, sched = make_optimizer(torch, net, common, steps)
    gen = torch.Generator().manual_seed(common['seed'])
    huber = torch.nn.HuberLoss(reduction='sum')
    yv = data['y'][val_rows].astype(np.float64)
    best, history, started = None, [], time.monotonic()
    print(json.dumps({'arm': arm, 'device': str(device), 'params': params, 'fit_rows': len(fit_rows),
                      'val_rows': len(val_rows), 'epochs': epochs, **extra}), flush=True)
    for epoch in range(epochs):
        net.train()
        sums = {'loss': 0., 'bce': 0., 'ce': 0., 'huber': 0.}
        for idx in torch.randperm(len(fit_rows), generator=gen).split(batch):
            rows = np.sort(fit_rows[idx.numpy()])
            xb = to_batch(torch, x, rows, dmean, dsd, device)
            pick = torch.randint(0, len(perms_cpu), (len(rows),), generator=gen).to(device)
            xb = torch.gather(xb, 1, perms[pick])
            rt = torch.from_numpy(rows)
            yb = y[rt].to(device)
            out = net(xb)
            if arm == 'heads':
                r, t = rel[rt].to(device), left[rt].to(device)
                m = r >= 0
                ce = torch.nn.functional.cross_entropy(out[0][m], r[m], reduction='sum')
                hu = huber(out[1][m], t[m])
                loss = (HEADS_CE_W * ce + HEADS_TURN_W * hu) / max(1, int(m.sum()))
                sums['ce'] += ce.item()
                sums['huber'] += hu.item()
            else:
                loss = bce_sum(torch, arm, out, yb) / len(rows)
            with torch.no_grad():
                sums['bce'] += bce_sum(torch, arm, out, yb).item()
            sums['loss'] += loss.item() * len(rows)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
        p, _ = score(torch, net, arm, x, val_rows, dmean, dsd, device)
        val_bce = log_loss(p, yv)
        entry = {'epoch': epoch, 'train_loss': sums['loss'] / len(fit_rows), 'train_bce': sums['bce'] / len(fit_rows),
                 'val_bce': val_bce, 'val_auc': fast_auc(p, yv), 'seconds': time.monotonic() - started}
        if arm == 'heads':
            entry.update({'train_ce': sums['ce'] / len(fit_rows), 'train_huber': sums['huber'] / len(fit_rows)})
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if best is None or val_bce < best[0]:
            best = (val_bce, epoch, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    seconds = time.monotonic() - started
    smoke = bool(args.max_rows or args.epochs)
    path = args.out / f'{arm}.pt'
    publish(path, lambda tmp: torch.save({
        'arm': arm, 'state': best[2], 'mean': mean, 'sd': sd, 'epoch': best[1], 'val_bce': best[0], 'config': cfg,
        'train_identity': verified['identity'], 'train_arrays_sha256': verified['arrays_sha256'],
        'train_sealed_after_the_fact': verified['sealed_after_the_fact'], 'params': params, 'train_seconds': seconds, 'smoke': smoke,
        'max_rows': args.max_rows, 'epochs': epochs}, tmp))
    return {'arm': arm, 'fit_rows': len(fit_rows), 'val_rows': len(val_rows),
            'fit_games': int(len(np.unique(data['game'][fit_rows]))),
            'val_games': int(len(np.unique(data['game'][val_rows]))),
            'chosen_epoch': best[1], 'chosen_val_bce': best[0], 'history': history, 'params': params,
            'train_seconds': seconds, 'device': str(device), 'epochs': epochs, 'max_rows': args.max_rows,
            'smoke': smoke, 'train_identity': verified['identity'], 'train_arrays_sha256': verified['arrays_sha256'],
            'train_sealed_after_the_fact': verified['sealed_after_the_fact'], 'model_sha256': sha256(path), **extra}


# --------------------------------------------------------------------- gate
def fast_auc(scores, labels):
    """Mann-Whitney AUC with midranks for ties (equals ``train_value.auc``)."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels) > 0
    pos = int(labels.sum())
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    midrank = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((midrank[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def log_loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins=ECE_BINS):
    which = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.
    for b in range(bins):
        m = which == b
        if m.any():
            total += abs(y[m].mean() - p[m].mean()) * m.sum() / len(p)
    return float(total)


def gate(args, cfg):
    import torch
    device = device_of(torch, args.device)
    board = layout()
    arms = [a for a in ARMS if (args.out / f'{a}.pt').exists()]
    if 'base' not in arms:
        raise ValueError('the gate needs the base checkpoint')
    trained = shards(args.data, 'train')
    train_binding = {k: trained[k] for k in ('identity', 'arrays_sha256', 'sealed_after_the_fact')}
    del trained
    test = subsample(shards(args.data, 'test'), args.max_rows, cfg['common']['seed'])
    lo, n = declared_split(cfg)['test']
    if test['game'].min() < lo or test['game'].max() >= lo + n:
        raise ValueError('test shards hold games outside the declared test split')
    rows = np.arange(len(test['y']))
    y = test['y'].astype(np.float64)
    result, preds, smoke = {}, {}, bool(args.max_rows)
    heads_targets = None
    if 'heads' in arms and args.records:
        recs = load_records(args.records, cfg, np.unique(test['game']), manifests(args.data, 'test')[1])
        heads_targets = head_targets(test, recs)
    for arm in arms:
        path = args.out / f'{arm}.pt'
        saved = torch.load(path, weights_only=False, map_location='cpu')
        if saved['config'] != cfg or saved['train_identity'] != train_binding['identity'] or saved['arm'] != arm:
            raise ValueError(f'{arm}: checkpoint provenance does not match this declaration and training data')
        bound = 'train_arrays_sha256' in saved
        if bound and saved['train_arrays_sha256'] != train_binding['arrays_sha256']:
            raise ValueError(f'{arm}: checkpoint was trained on different shard arrays')
        smoke = smoke or saved['smoke']
        net = make_net(torch, arm, board)
        net.load_state_dict(saved['state'])
        net.to(device)
        p, logp = score(torch, net, arm, test['x'], rows, saved['mean'].to(device), saved['sd'].to(device), device)
        preds[arm] = p
        point = fast_auc(p, y)
        reference = auc(p.tolist(), y.astype(int).tolist())
        if abs(point - reference) > 1e-9:
            raise ValueError('fast AUC disagrees with train_value.auc')
        result[arm] = {'auc': point, 'log_loss': log_loss(p, y), 'ece10': ece(p, y), 'params': saved['params'],
                       'train_seconds': saved['train_seconds'], 'chosen_epoch': saved['epoch'],
                       'chosen_val_bce': saved['val_bce'], 'smoke': saved['smoke'], 'model_sha256': sha256(path),
                       'train_arrays_bound_in_checkpoint': bound}
        if arm == 'heads' and heads_targets is not None:
            rel, _, counts = heads_targets
            m = rel >= 0
            result[arm]['four_way_log_loss'] = float(-np.mean(logp[m, rel[m]]))
            result[arm]['four_way_rows'] = int(m.sum())
            result[arm]['target_counts'] = counts
    by_game = {}
    for i, g in enumerate(test['game']):
        by_game.setdefault(int(g), []).append(i)
    ids = sorted(by_game)
    by_game = {g: np.asarray(v) for g, v in by_game.items()}
    others = [a for a in arms if a != 'base']
    rng = random.Random(cfg_gate(cfg)['seed'])
    diffs = {a: [] for a in others}
    for _ in range(cfg_gate(cfg)['resamples']):
        sample = np.concatenate([by_game[rng.choice(ids)] for _ in ids])
        b = fast_auc(preds['base'][sample], y[sample])
        for a in others:
            diffs[a].append(fast_auc(preds[a][sample], y[sample]) - b)
    bar = cfg_gate(cfg)['min_diff']
    for a in others:
        d = sorted(diffs[a])
        diff = result[a]['auc'] - result['base']['auc']
        lcb, ucb = d[int(.025 * len(d))], d[int(.975 * len(d)) - 1]
        result[a]['vs_base'] = {'auc_diff': diff, 'ci95': [lcb, ucb], 'pass': bool(diff >= bar and lcb > 0)}
    sealed = {'train': train_binding['sealed_after_the_fact'], 'test': test['sealed_after_the_fact']}
    limitations = [f'{a}: checkpoint predates array binding (sidecar identity only; its training arrays are '
                   'checked only through the current shard verification, not against the checkpoint)'
                   for a in arms if not result[a]['train_arrays_bound_in_checkpoint']]
    limitations += [f'{split} shards: NPZ hashes sealed after the fact ({info}), not by the producer'
                    for split, info in sealed.items() if info is not None]
    return {'arms': result, 'test_rows': len(y), 'test_games': len(ids), 'test_identity': test['identity'],
            'test_arrays_sha256': test['arrays_sha256'], 'train_identity': train_binding['identity'],
            'train_arrays_sha256': train_binding['arrays_sha256'], 'sealed_after_the_fact': sealed,
            'legacy_seal': any(v is not None for v in sealed.values()), 'identity_limitations': limitations, 'bar': {'auc_diff_min': bar, 'lcb_gt': 0., **cfg_gate(cfg)},
            'device': str(device), 'max_rows': args.max_rows, 'smoke': smoke,
            'heads_four_way': heads_targets is not None}


def cfg_gate(cfg):
    """The gate constants as written in the declaration's ``gate.primary`` sentence."""
    text = cfg['gate']['primary']
    m = re.search(r'>= \+(0\.\d+) .*\((\d[\d,]*) resamples, seed (\d+)\)', text)
    if not m:
        raise ValueError('cannot read the gate constants from gate.primary')
    return {'min_diff': float(m.group(1)), 'resamples': int(m.group(2).replace(',', '')), 'seed': int(m.group(3))}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train', 'gate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--records', type=Path, help='runs dir holding the game-record roots (heads targets)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='auto', help='auto (mps if available, else cpu), cpu or mps')
    parser.add_argument('--max-rows', type=int, default=None, help='smoke only')
    parser.add_argument('--epochs', type=int, default=None, help='smoke only')
    args = parser.parse_args(argv)
    if args.command == 'train' and args.arm is None:
        parser.error('train needs --arm')
    if args.command == 'train' and args.arm == 'heads' and args.records is None:
        parser.error('the heads arm needs --records')
    if args.command == 'gate' and args.epochs:
        parser.error('--epochs applies to train only')
    cfg = json.loads(args.config.read_text())
    check_data(cfg, args.data)
    if args.command == 'gate':
        cfg_gate(cfg)  # only the bakeoff gate reads these; a train-only declaration (tiles-onpolicy) words its own gate
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / (f'{args.arm}-train.json' if args.command == 'train' else 'gate.json')
    refuse_existing(output, *((args.out / f'{args.arm}.pt',) if args.command == 'train' else ()))
    started = time.monotonic()
    result = (train if args.command == 'train' else gate)(args, cfg)
    result.update({'command': args.command, 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                   'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'seconds': time.monotonic() - started})
    publish(output, lambda tmp: tmp.write_text(json.dumps(result, indent=1)))
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}, indent=1))
    return result


if __name__ == '__main__':
    main()
