"""PV search v2: the served search proposes, a learned afterstate value decides.

Shared by the data generator (``scripts/pv2_data.py``) and the bot, so training
and play see the same candidates and the same features (model audit fix 1/4):

- ``candidate_sequences``: "end the turn now" plus, for each first move the served
  turn search tried, its best continuation by the served score; the ``k`` best by
  that score are kept. Sequences are built on the public projection exactly as
  the served search does.
- ``afterstate_encoding``: replays a sequence on the card-counted world built from
  the chooser's view and encodes the resulting position with encoder v2. A
  development purchase is averaged over the standard deck prior (14/5/2/2/2), so
  it cannot reveal the hidden deck.

``PVSearchPlanner`` (``planner_pv2``) plays the candidate whose afterstate the
frozen value MLP rates highest; every other decision is served. It is registered
only once a model passes its predeclared within-decision gate.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .encoding import encode
from .observation import Observation
from .tactical import public_projection
from .unified_planner import AcceptanceOfferPlanner

DECK_PRIOR = (('knight', 14), ('victory_point', 5), ('road_building', 2), ('year_of_plenty', 2), ('monopoly', 2))
REPLACEABLE = ('build_city', 'build_settlement', 'build_road', 'bank_trade', 'buy_development', 'end_turn', 'offer_trade')


def served_sequences(planner, view, seat):
    """The served turn search, keeping the best-scoring sequence for each first move."""
    legal = view.legal_actions(seat)
    root = public_projection(view)
    if planner.buy_after_root and sum(e['action']['type'] == 'buy_development' for e in view.events) < 25:
        root.deck = ['unknown']
    best = {}
    frontier = [(root, [], legal)]
    nodes = 0
    for depth in range(3):
        successors = []
        for game, path, actions in frontier:
            for action in actions:
                kind = action['type']
                build = kind in {'build_city', 'build_settlement', 'build_road'}
                acquire = kind == 'bank_trade' or kind == 'play_development' and action['card'] in {'year_of_plenty', 'road_building'}
                buy = kind == 'buy_development'
                if not (build or buy and (depth == 0 or planner.buy_after_root) or depth == 0 and acquire):
                    continue
                if nodes >= 128:
                    break
                nodes += 1
                child = deepcopy(game, {id(game.board): game.board})
                if buy:
                    child.players[seat].pay('development')
                else:
                    child.apply(seat, action)
                score = planner._leaf(child, seat) + (planner.purchase_value if buy else 0) - .01 * (depth + 1)
                seq = path + [action]
                key = json.dumps(seq[0], sort_keys=True)
                if key not in best or score > best[key][0]:
                    best[key] = (score, seq)
                if not buy and child.winner is None:
                    successors.append((score, child, seq))
        successors.sort(key=lambda row: row[0], reverse=True)
        frontier = [(game, seq, game.legal_actions(seat)) for _, game, seq in successors[:8]]
    return sorted(best.values(), key=lambda t: -t[0])


def candidate_sequences(planner, view, seat, k):
    """[] (end the turn now) followed by the ``k`` best served sequences by first move."""
    return [[]] + [seq for _, seq in served_sequences(planner, view, seat)[:k]]


def _replay(world, seat, seq):
    for action in seq:
        if action['type'] == 'buy_development':
            # The card-counted world keeps no deck (its contents are hidden), so the
            # engine's legality check would refuse every purchase; check what the
            # chooser can know: phase and affordability.
            if world.phase != 'actions' or not world.players[seat].can_afford('development'):
                return None
            world.players[seat].pay('development')
            return world, True  # the purchase ends the sequence (served search never continues past it)
        if action not in world.legal_actions(seat):
            return None
        world.apply(seat, action)
    return world, False


def afterstate_encoding(planner, view, seat, seq):
    """Encoder-v2 features of the position after ``seq``, from the chooser's information only.

    Returns ``None`` if the sequence is not legal on the card-counted world, and
    ``(features, won)`` otherwise, where ``won`` marks an immediate win.
    """
    replayed = _replay(planner._world(view), seat, seq)
    if replayed is None:
        return None
    world, bought = replayed
    if world.winner == seat:
        return None, True
    if not bought:
        return encode(Observation.from_game(world, seat, include_events=False), world), False
    total = sum(n for _, n in DECK_PRIOR)
    mixed = None
    for card, n in DECK_PRIOR:
        variant = deepcopy(world, {id(world.board): world.board})
        variant.players[seat].development.append((card, view.turn))
        x = encode(Observation.from_game(variant, seat, include_events=False), variant)
        mixed = [n / total * v for v in x] if mixed is None else [m + n / total * v for m, v in zip(mixed, x)]
    return mixed, False


def load_value_mlp(path, sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError('value MLP checkpoint does not match its frozen hash')
    data = json.loads(raw)
    layers = []
    names = sorted({k.split('.')[0] for k in data['state'] if k.endswith('.weight')}, key=int)
    for n in names:
        layers.append((data['state'][f'{n}.weight'], data['state'][f'{n}.bias']))
    return {'mean': data['mean'], 'sd': data['sd'], 'layers': layers}


def mlp_logit(model, x):
    h = [(v - m) / s for v, m, s in zip(x, model['mean'], model['sd'])]
    for i, (w, b) in enumerate(model['layers']):
        h = [bias + sum(wi * hi for wi, hi in zip(row, h)) for row, bias in zip(w, b)]
        if i < len(model['layers']) - 1:
            h = [v if v > 0 else 0. for v in h]
    return h[0]


class PVSearchPlanner(AcceptanceOfferPlanner):
    """``planner_offer_accept`` whose own-turn build decisions are made by the afterstate value."""

    K = 4
    CHECKPOINT = None
    CHECKPOINT_SHA256 = None

    def __init__(self, rng=None):
        super().__init__(rng)
        self.value_mlp = load_value_mlp(self.CHECKPOINT, self.CHECKPOINT_SHA256)

    def _logits(self, rows):
        return [mlp_logit(self.value_mlp, x) for x in rows]


class ValueNetV2:
    """Numpy forward pass for the exported encoder-v2 state-value nets (``mlp`` or ``resmlp``).

    Loads ``<name>.npz`` and ``<name>.meta.json``; refuses a weights file whose
    SHA-256 differs from the metadata's, and checks itself against the torch logits
    stored with the export (parity probe) before use.
    """

    def __init__(self, stem):
        import numpy as np
        self.np = np
        meta = json.loads(Path(f'{stem}.meta.json').read_text())
        raw = Path(f'{stem}.npz').read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta['npz_sha256']:
            raise ValueError('value net weights do not match their metadata')
        data = np.load(f'{stem}.npz')
        self.arch, self.meta = meta['arch'], meta
        self.w = {k[2:]: data[k].astype(np.float64) for k in data.files if k.startswith('w:')}
        self.mean, self.sd = data['mean'].astype(np.float64), data['sd'].astype(np.float64)
        got = self.logits(data['parity_x'])
        if not np.allclose(got, data['parity_logit'], atol=1e-4):
            raise ValueError('numpy forward pass disagrees with the exported torch logits')

    def _ln(self, h, name):
        mu = h.mean(-1, keepdims=True)
        var = h.var(-1, keepdims=True)
        return (h - mu) / self.np.sqrt(var + 1e-5) * self.w[f'{name}.weight'] + self.w[f'{name}.bias']

    def _lin(self, h, name):
        return h @ self.w[f'{name}.weight'].T + self.w[f'{name}.bias']

    def logits(self, rows):
        np = self.np
        h = (np.asarray(rows, dtype=np.float64) - self.mean) / self.sd
        if self.arch.get('arch', 'mlp') == 'mlp':
            names = sorted({k.split('.')[0] for k in self.w if k.endswith('.weight')}, key=int)
            for i, n in enumerate(names):
                h = self._lin(h, n)
                if i < len(names) - 1:
                    h = np.maximum(h, 0.)
            return h[:, 0]
        h = self._lin(h, 'inp')
        for b in range(self.arch['blocks']):
            f = self._ln(h, f'blocks.{b}.f.0')
            f = np.maximum(self._lin(f, f'blocks.{b}.f.1'), 0.)
            h = h + self._lin(f, f'blocks.{b}.f.4')
        return self._lin(self._ln(h, 'out.0'), 'out.1')[:, 0]


VALUE_V2_B = Path(__file__).resolve().parents[2] / 'checkpoints' / 'value_v2_resmlp_b_v1'


class PVSearchV2BPlanner(PVSearchPlanner):
    """PV search with the 100k encoder-v2 state-value net, arm B (``value-v2-100k-aug-b``):
    test AUC .772; pv2b within-decision cross-eval +.0045 [LCB +.0008]. Needs numpy."""

    def __init__(self, rng=None):
        AcceptanceOfferPlanner.__init__(self, rng)
        self.net = ValueNetV2(VALUE_V2_B)

    def _logits(self, rows):
        return [float(v) for v in self.net.logits(rows)] if rows else []

    def choose(self, view, seat):
        served = super().choose(view, seat)
        if (view.phase != 'actions' or served['type'] not in REPLACEABLE
                or not any(a['type'] == 'end_turn' for a in view.actions)):
            return served
        scored = []
        for seq in candidate_sequences(self, view, seat, self.K):
            encoded = afterstate_encoding(self, view, seat, seq)
            if encoded is not None:
                scored.append((seq, *encoded))
        logits = self._logits([x for _, x, won in scored if not won]) if scored else []
        best, pick, it = -math.inf, None, iter(logits)
        for seq, _, won in scored:
            value = math.inf if won else next(it)
            if value > best:
                best, pick = value, seq
        if pick is None or not pick:
            return served if served['type'] in ('end_turn', 'offer_trade') else {'type': 'end_turn'}
        return pick[0]


class PVSearchV2BFixPlanner(PVSearchV2BPlanner):
    """``planner_pv2_b`` with the red-team integration fix (board #1, red team #2, step 1).

    The parked arm dropped served trade offers (the served ``choose`` had already marked
    the turn as offered) and ended turns with a bare ``end_turn`` that skipped the served
    end-of-turn logic. Here: a served offer is kept; PV only replaces served builds, bank
    trades, purchases and ends of turn; when PV chooses to end the turn, the served
    end-of-turn path runs (development-card fallback when a card is playable, then the
    offer step). Same value net, candidates and features.
    """

    def choose(self, view, seat):
        before = self.last_offer_turn
        served = AcceptanceOfferPlanner.choose(self, view, seat)
        if (view.phase != 'actions' or served['type'] in ('offer_trade', 'play_development')
                or served['type'] not in REPLACEABLE or not any(a['type'] == 'end_turn' for a in view.actions)):
            return served
        scored = []
        for seq in candidate_sequences(self, view, seat, self.K):
            encoded = afterstate_encoding(self, view, seat, seq)
            if encoded is not None:
                scored.append((seq, *encoded))
        logits = self._logits([x for _, x, won in scored if not won]) if scored else []
        best, pick, it = -math.inf, None, iter(logits)
        for seq, _, won in scored:
            value = math.inf if won else next(it)
            if value > best:
                best, pick = value, seq
        self.last_offer_turn = before
        if pick is None:
            return served
        if not pick:  # end the turn: the served end-of-turn path, not a bare end_turn
            if served['type'] == 'end_turn':
                return served
            legal = view.legal_actions(seat)
            action = (self._fallback(view, seat) if any(a['type'] == 'play_development' for a in legal)
                      else {'type': 'end_turn'})
            return self._end_of_turn_offer(view, seat, action)
        return pick[0]


def afterstate_variants(planner, view, seat, seq):
    """Like ``afterstate_encoding`` but a purchase returns one encoding per card type with
    its deck-prior weight, so the caller can average predictions instead of features.

    Returns ``None`` (not replayable), ``'won'`` (immediate win) or ``[(weight, x), ...]``.
    """
    replayed = _replay(planner._world(view), seat, seq)
    if replayed is None:
        return None
    world, bought = replayed
    if world.winner == seat:
        return 'won'
    if not bought:
        return [(1., encode(Observation.from_game(world, seat, include_events=False), world))]
    total = sum(n for _, n in DECK_PRIOR)
    out = []
    for card, n in DECK_PRIOR:
        variant = deepcopy(world, {id(world.board): world.board})
        variant.players[seat].development.append((card, view.turn))
        out.append((n / total, encode(Observation.from_game(variant, seat, include_events=False), variant)))
    return out


class PVSearchS1Planner(PVSearchV2BFixPlanner):
    """``planner_pv2_b_fix`` plus two search fixes from red team #2 (board #1):

    - commit to the picked sequence: the rest of the chosen sequence is replayed on
      later calls in the same turn while each step stays legal (re-plan otherwise);
      when it is exhausted the turn ends through the served end-of-turn path, which
      is exactly what the data and the within-decision gate assumed;
    - a development purchase is scored by averaging predicted win probabilities over
      the deck prior, not by averaging features.
    """

    NET = VALUE_V2_B

    def __init__(self, rng=None):
        AcceptanceOfferPlanner.__init__(self, rng)
        self.net = ValueNetV2(self.NET)
        self._plan = None

    def _end_path(self, view, seat):
        legal = view.legal_actions(seat)
        action = (self._fallback(view, seat) if any(a['type'] == 'play_development' for a in legal)
                  else {'type': 'end_turn'})
        return self._end_of_turn_offer(view, seat, action)

    def choose(self, view, seat):
        if view.phase == 'actions' and self._plan is not None and self._plan[0] == (view.turn, seat):
            remaining = self._plan[1]
            if not remaining:
                self._plan = None
                return self._end_path(view, seat)
            step = remaining[0]
            if step in view.legal_actions(seat):
                self._plan = ((view.turn, seat), remaining[1:])
                return step
            self._plan = None  # no longer legal: re-plan below
        elif self._plan is not None and self._plan[0] != (view.turn, seat):
            self._plan = None
        before = self.last_offer_turn
        served = AcceptanceOfferPlanner.choose(self, view, seat)
        if (view.phase != 'actions' or served['type'] in ('offer_trade', 'play_development')
                or served['type'] not in REPLACEABLE or not any(a['type'] == 'end_turn' for a in view.actions)):
            return served
        options, rows = [], []
        for seq in candidate_sequences(self, view, seat, self.K):
            variants = afterstate_variants(self, view, seat, seq)
            if variants is None:
                continue
            if variants == 'won':
                options.append((seq, None))
                continue
            options.append((seq, (len(rows), [w for w, _ in variants])))
            rows.extend(x for _, x in variants)
        logits = self._logits(rows) if rows else []
        best, pick = -math.inf, None
        for seq, spec in options:
            if spec is None:
                value = math.inf
            else:
                start, weights = spec
                p = sum(w / (1 + math.exp(-logits[start + i])) for i, w in enumerate(weights))
                value = math.log(max(p, 1e-12) / max(1 - p, 1e-12))
            if value > best:
                best, pick = value, seq
        self.last_offer_turn = before
        if pick is None:
            return served
        if not pick:
            return served if served['type'] == 'end_turn' else self._end_path(view, seat)
        self._plan = ((view.turn, seat), pick[1:])
        return pick[0]


class PVSearchS1M1Planner(PVSearchS1Planner):
    """``planner_pv2_s1`` with the fine-tuned value net ``value-v2-m1`` (local weights)."""

    NET = Path(__file__).resolve().parents[2] / 'checkpoints' / 'value_v2_m1_v1'


class PVSearchTilesPlanner(PVSearchV2BFixPlanner):
    """``planner_pv2_b_fix`` with its value scorer replaced by the arch-bakeoff ``tiles``
    transformer (``runs/arch-bakeoff-v1``; test AUC .7859 vs .7721 base; within-decision
    cross-eval +.0062), served in NumPy (``tiles_value.TilesValueNet``). Same candidates,
    afterstate features and integration; only ``_logits`` changes. Needs numpy, not torch."""

    def __init__(self, rng=None):
        from .tiles_value import TILES_V1, TilesValueNet
        AcceptanceOfferPlanner.__init__(self, rng)
        self.net = TilesValueNet(TILES_V1)

    def _logits(self, rows):
        return [float(v) for v in self.net.logits(rows)] if rows else []
