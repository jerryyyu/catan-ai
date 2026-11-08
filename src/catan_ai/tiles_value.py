"""NumPy serving of the arch-bakeoff ``tiles`` value transformer (encoder-v2 rows -> logit).

Reproduces ``make_net(torch, 'tiles', board)`` from ``scripts/arch_bakeoff.py`` without
torch: standardize the 1,324 features with the checkpoint's ``mean``/``sd``; build 19
tile tokens (tile block + its 6 corner vertex blocks + its 6 side edge blocks, corners
counterclockwise by angle from the tile centre), 4 player tokens and 1 global token;
per-type linear projections plus learned position and type embeddings; 4 pre-norm
encoder layers (d 128, 4 heads, MLP 256, ReLU); LayerNorm on the global token and a
linear head giving the logit of P(I win).

Weights come from ``<stem>.npz`` written by ``scripts/export_tiles_npz.py``; the loader
refuses a file whose SHA-256 differs from ``<stem>.meta.json``, refuses a tokenizer
index that differs from the trained one, and checks its logits against the torch logits
stored with the export (parity probe) before use.
"""
import hashlib
import json
import math
from pathlib import Path

N_TILES, N_VERTICES, N_EDGES, N_PLAYERS = 19, 54, 72, 4
TILE_W, VERTEX_W, EDGE_W, PLAYER_W, GLOBAL_W = 8, 14, 4, 27, 20
V0 = N_TILES * TILE_W
E0 = V0 + N_VERTICES * VERTEX_W
P0 = E0 + N_EDGES * EDGE_W
G0 = P0 + N_PLAYERS * PLAYER_W
WIDTH = G0 + GLOBAL_W
TOKEN_W = TILE_W + 6 * VERTEX_W + 6 * EDGE_W
D, HEADS, LAYERS = 128, 4, 4
LN_EPS = 1e-5
PARITY_ATOL = 1e-4

TILES_V1 = Path(__file__).resolve().parents[2] / 'checkpoints' / 'value_tiles_v1'


def board():
    from .encoding import size
    from .engine import Game
    if size() != WIDTH:
        raise ValueError('encoder-v2 width differs from the tiles tokenizer layout')
    b = Game.new(list('abcd'), 0).board
    if (len(b.tiles), len(b.vertices), len(b.edges)) != (N_TILES, N_VERTICES, N_EDGES):
        raise ValueError('board topology differs from the encoder layout')
    return b


def tile_geometry(b):
    """Per tile: corner vertex ids counterclockwise by angle from the tile centre (smallest
    angle in [0, 360) first) and side k = the edge between corners k and k+1."""
    eindex = {tuple(sorted(e)): i for i, e in enumerate(b.edges)}
    out = []
    for tile in b.tiles:
        cx = sum(b.vertices[v][0] for v in tile.vertices) / len(tile.vertices)
        cy = sum(b.vertices[v][1] for v in tile.vertices) / len(tile.vertices)

        def angle(v, cx=cx, cy=cy):
            x, y = b.vertices[v]
            return round(math.degrees(math.atan2(y - cy, x - cx)) % 360, 3) % 360
        corners = sorted(tile.vertices, key=angle)
        if len(corners) != 6:
            raise ValueError('tile without six corners')
        sides = [eindex[tuple(sorted((corners[k], corners[(k + 1) % 6])))] for k in range(6)]
        out.append((corners, sides))
    return out


def tile_index(b=None):
    """[19][116] flat feature indices of each tile token."""
    rows = []
    for t, (corners, sides) in enumerate(tile_geometry(b if b is not None else board())):
        row = list(range(t * TILE_W, (t + 1) * TILE_W))
        for v in corners:
            row += range(V0 + v * VERTEX_W, V0 + (v + 1) * VERTEX_W)
        for e in sides:
            row += range(E0 + e * EDGE_W, E0 + (e + 1) * EDGE_W)
        rows.append(row)
    return rows


def tokenize(np, x, tidx):
    """Standardized rows [n, 1324] -> (tile [n, 19, 116], player [n, 4, 27], global [n, 20])."""
    n = len(x)
    return x[:, tidx], x[:, P0:G0].reshape(n, N_PLAYERS, PLAYER_W), x[:, G0:]


class TilesValueNet:
    """NumPy forward pass of the exported tiles transformer (float64)."""

    def __init__(self, stem=TILES_V1):
        import numpy as np
        self.np = np
        stem = str(stem)
        meta = json.loads(Path(f'{stem}.meta.json').read_text())
        raw = Path(f'{stem}.npz').read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta['npz_sha256']:
            raise ValueError('tiles value weights do not match their metadata')
        data = np.load(f'{stem}.npz')
        self.meta = meta
        self.w = {k[2:]: data[k].astype(np.float64) for k in data.files if k.startswith('w:')}
        self.mean, self.sd = data['mean'].astype(np.float64), data['sd'].astype(np.float64)
        self.tidx = np.asarray(tile_index(), dtype=np.int64)
        if not np.array_equal(self.tidx, data['tidx']):
            raise ValueError('tile tokenizer index differs from the trained one')
        types = np.asarray([0] * N_TILES + [1] * N_PLAYERS + [2])
        if not np.array_equal(types, data['types']):
            raise ValueError('token types differ from the trained ones')
        self.embed = self.w['pos'] + self.w['kind'][types]
        got = self.logits(data['parity_x'])
        diff = float(np.max(np.abs(got - data['parity_logit'])))
        if not diff < PARITY_ATOL:
            raise ValueError(f'numpy forward pass disagrees with the exported torch logits ({diff:.3g})')
        self.parity_max_abs_diff = diff

    def _ln(self, h, name):
        np = self.np
        mu = h.mean(-1, keepdims=True)
        var = ((h - mu) ** 2).mean(-1, keepdims=True)
        return (h - mu) / np.sqrt(var + LN_EPS) * self.w[f'{name}.weight'] + self.w[f'{name}.bias']

    def _lin(self, h, name):
        return h @ self.w[f'{name}.weight'].T + self.w[f'{name}.bias']

    def _attention(self, h, p):
        np = self.np
        n, t, _ = h.shape
        hd = D // HEADS
        qkv = h @ self.w[f'{p}.in_proj_weight'].T + self.w[f'{p}.in_proj_bias']
        q, k, v = (qkv[..., i * D:(i + 1) * D].reshape(n, t, HEADS, hd).transpose(0, 2, 1, 3) for i in range(3))
        s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
        s = np.exp(s - s.max(-1, keepdims=True))
        a = (s / s.sum(-1, keepdims=True)) @ v
        return self._lin(a.transpose(0, 2, 1, 3).reshape(n, t, D), f'{p}.out_proj')

    def standardize(self, rows):
        return (self.np.asarray(rows, dtype=self.np.float64) - self.mean) / self.sd

    def forward_standardized(self, x):
        np = self.np
        tile, player, glob = tokenize(np, x, self.tidx)
        h = np.concatenate([self._lin(tile, 'tile'), self._lin(player, 'player'),
                            self._lin(glob, 'glob')[:, None]], 1) + self.embed
        for i in range(LAYERS):
            p = f'encoder.layers.{i}'
            h = h + self._attention(self._ln(h, f'{p}.norm1'), f'{p}.self_attn')
            f = np.maximum(self._lin(self._ln(h, f'{p}.norm2'), f'{p}.linear1'), 0.)
            h = h + self._lin(f, f'{p}.linear2')
        return self._lin(self._ln(h[:, -1], 'norm'), 'head')[:, 0]

    def logits(self, rows):
        return self.forward_standardized(self.standardize(rows))
