"""Corrected scoring of development-card purchases for the PV planners (``planner_pv2_tiles_buyfix``).

The transformer red team (board #1 comment 6751886556) found three Low defects in how
``pv_search.afterstate_encoding`` scores a candidate sequence that ends in a purchase:

* D1: the win check runs before the hypothetical card is added, so buying the victory-point card that
  reaches 10 points is scored by the value net (on a 10-point state it never saw) instead of as a win;
* D3: the card types are mixed by averaging *features* over the deck prior, producing fractional card
  counts never seen in training, instead of averaging the predicted win probabilities;
* D4: the deck prior is always the full 14/5/2/2/2 deck, ignoring the chooser's own cards and every card
  already played publicly.

This module scores purchases with all three corrected, from the chooser's information only: the cards it
holds, its own purchases, and public play events. Everything else (candidates, non-purchase afterstates,
end-of-turn handling) is ``planner_pv2_tiles``'s, so the two planners differ only on purchase scoring.
"""
from copy import deepcopy
import math

from .encoding import encode
from .observation import Observation
from .pv_search import (DECK_PRIOR, PVSearchTilesPlanner, REPLACEABLE, _replay, candidate_sequences)
from .unified_planner import AcceptanceOfferPlanner

WIN_POINTS = 10


def deck_prior(view, seat):
    """Remaining-deck counts as the chooser can know them: the full deck minus its own cards (held or
    played) and cards other players have played publicly. Falls back to the full prior if nothing remains."""
    remaining = dict(DECK_PRIOR)
    own = view.players[seat].development or ()
    for card in own:
        remaining[card] = remaining.get(card, 0) - 1
    for event in view.events:
        action = event.get('action') or {}
        if action.get('type') == 'play_development' and action.get('card') in remaining:
            remaining[action['card']] -= 1  # own plays are no longer held, so they are not double counted
    out = tuple((card, remaining[card]) for card, _ in DECK_PRIOR if remaining.get(card, 0) > 0)
    return out or DECK_PRIOR


def purchase_variants(world, seat, view):
    """[(weight, won, features)] for a purchase afterstate: one per possible card, won when it reaches 10."""
    prior = deck_prior(view, seat)
    total = sum(n for _, n in prior)
    out = []
    for card, n in prior:
        variant = deepcopy(world, {id(world.board): world.board})
        variant.players[seat].development.append((card, view.turn))
        won = variant.players[seat].points >= WIN_POINTS
        x = None if won else encode(Observation.from_game(variant, seat, include_events=False), variant)
        out.append((n / total, won, x))
    return out


class PVSearchTilesBuyFixPlanner(PVSearchTilesPlanner):
    """``planner_pv2_tiles`` with purchases scored as the mean win probability over the chooser's deck
    estimate (a winning victory-point draw counts as a win)."""

    def _pick(self, view, seat, fixed=True):
        """The sequence the PV comparison picks; ``fixed=False`` reproduces ``planner_pv2_tiles``."""
        if not fixed:
            return super_pick(self, view, seat)
        options, rows = [], []
        for seq in candidate_sequences(self, view, seat, self.K):
            replayed = _replay(self._world(view), seat, seq)
            if replayed is None:
                continue
            world, bought = replayed
            if world.winner == seat:
                options.append((seq, None))
                continue
            if not bought:
                options.append((seq, [(1.0, False, len(rows))]))
                rows.append(encode(Observation.from_game(world, seat, include_events=False), world))
                continue
            spec = []
            for weight, won, x in purchase_variants(world, seat, view):
                if won:
                    spec.append((weight, True, None))
                else:
                    spec.append((weight, False, len(rows)))
                    rows.append(x)
            options.append((seq, spec))
        logits = self._logits(rows) if rows else []
        best, pick = -math.inf, None
        for seq, spec in options:
            if spec is None:
                value = math.inf
            elif len(spec) == 1 and spec[0][0] == 1.0 and not spec[0][1]:
                value = logits[spec[0][2]]  # not a purchase: the raw logit, exactly as planner_pv2_tiles
            else:
                p = sum(w * (1.0 if won else 1 / (1 + math.exp(-logits[i]))) for w, won, i in spec)
                value = math.inf if p >= 1.0 else math.log(max(p, 1e-12) / max(1 - p, 1e-12))
            if value > best:
                best, pick = value, seq
        return pick

    def choose(self, view, seat):
        before = self.last_offer_turn
        served = AcceptanceOfferPlanner.choose(self, view, seat)
        if (view.phase != 'actions' or served['type'] in ('offer_trade', 'play_development')
                or served['type'] not in REPLACEABLE or not any(a['type'] == 'end_turn' for a in view.actions)):
            return served
        pick = self._pick(view, seat)
        self.last_offer_turn = before
        if pick is None:
            return served
        if not pick:  # end the turn: the served end-of-turn path, as planner_pv2_tiles does
            if served['type'] == 'end_turn':
                return served
            legal = view.legal_actions(seat)
            action = (self._fallback(view, seat) if any(a['type'] == 'play_development' for a in legal)
                      else {'type': 'end_turn'})
            return self._end_of_turn_offer(view, seat, action)
        return pick[0]


def super_pick(planner, view, seat):
    """``planner_pv2_tiles``' own pick (feature-averaged purchases, full deck prior), for comparisons."""
    from .pv_search import afterstate_encoding
    scored = []
    for seq in candidate_sequences(planner, view, seat, planner.K):
        encoded = afterstate_encoding(planner, view, seat, seq)
        if encoded is not None:
            scored.append((seq, *encoded))
    logits = planner._logits([x for _, x, won in scored if not won]) if scored else []
    best, pick, it = -math.inf, None, iter(logits)
    for seq, _, won in scored:
        value = math.inf if won else next(it)
        if value > best:
            best, pick = value, seq
    return pick
