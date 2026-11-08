"""Ask the served policy for one JSettlers decision and translate the answer.

A request is a snapshot (see ``state.py``). The reply is a JSettlers command:

========================  =============================================================
engine action             reply ``cmd``
========================  =============================================================
build_settlement          ``settlement`` (``node``)
build_road                ``road`` (``edge``)
build_city                ``city`` (``node``)
buy_development           ``buy_dev``
bank_trade                ``bank`` (``give``/``get`` resource-code -> count)
play_development          ``play_dev`` (``card``; ``resource`` or ``pick`` for monopoly / YoP)
end_turn / roll           ``end_turn`` / ``roll``
move_robber / steal       ``robber`` (``hex``) / ``steal`` (``victim`` player number)
discard (repeated)        ``discard`` (``resources``)
accept/reject_trade       ``accept`` / ``reject``
offer_trade               ``offer`` (``to`` player number, ``give``, ``get``)
========================  =============================================================

When the engine action is impossible under JSettlers' own legal set (sent in
``js_legal``) the reply is ``{"cmd": "fallback", "reason": ...}``; the Java brain then
uses its own JSettlers choice and counts the fallback.
"""
from __future__ import annotations

import time
import zlib

from catan_ai.bots import make_policy
from catan_ai.engine import Resource
from catan_ai.observation import Observation

from .board import RESOURCE_CODE
from .state import DEV_CODES, reconstruct

POLICY = 'planner_offer_accept'


def _codes(bundle: dict) -> dict[str, int]:
    return {str(RESOURCE_CODE[Resource(r)]): int(n) for r, n in bundle.items() if n}


class Bridge:
    """Long-lived per-process state: one policy instance per JSettlers game."""

    def __init__(self, policy: str = POLICY, seed: int = 0):
        self.policy_name = policy
        self.seed = seed
        self.policies: dict[str, object] = {}

    def policy_for(self, game: str):
        if game not in self.policies:
            seed = (self.seed * 1_000_003 + zlib.crc32(game.encode())) % (2 ** 31)
            self.policies[game] = make_policy(self.policy_name, seed)
        return self.policies[game]

    def end_game(self, game: str) -> None:
        self.policies.pop(game, None)

    def decide(self, snap: dict) -> dict:
        start = time.perf_counter()
        game, obs, ctx = reconstruct(snap)
        policy = self.policy_for(snap['game'])
        seat = ctx['seat']
        mismatches = legal_mismatches(snap, obs, ctx)
        if snap['decision'] == 'discard':
            reply = self._discard(game, policy, seat, ctx)
        else:
            action = policy.choose(obs, seat)
            reply = translate(action, snap, game, ctx)
        reply['engine_legal'] = len(obs.actions)
        if mismatches:
            reply['legal_mismatches'] = mismatches
        reply['think_ms'] = round(1000 * (time.perf_counter() - start), 2)
        return reply

    @staticmethod
    def _discard(game, policy, seat, ctx):
        chosen = {}
        while game.phase == 'discard' and seat in game.discard_remaining:
            obs = Observation.from_game(game, seat, include_events=False)
            obs = _with_events(obs, ctx)
            action = policy.choose(obs, seat)
            if action.get('type') != 'discard':
                return {'cmd': 'fallback', 'reason': f'discard step returned {action.get("type")}'}
            chosen[action['resource']] = chosen.get(action['resource'], 0) + 1
            game.apply(seat, action)
        return {'cmd': 'discard', 'resources': _codes(chosen), 'action': {'type': 'discard', 'resources': chosen}}


def _with_events(obs, ctx):
    from dataclasses import replace
    return replace(obs, events=ctx['events']) if 'events' in ctx else obs


def translate(action: dict, snap: dict, game, ctx) -> dict:
    m, seats = ctx['board_map'], ctx['seats']
    v2n, e2c = m.vertex_to_node, m.engine_to_edge
    legal = snap.get('js_legal', {})
    kind = action.get('type')
    out = {'action': action}

    def fallback(reason):
        return {'cmd': 'fallback', 'reason': reason, 'action': action}

    if kind in ('build_settlement', 'build_city'):
        node = v2n[action['vertex']]
        key = 'settlements' if kind == 'build_settlement' else 'cities'
        if key in legal and node not in legal[key]:
            return fallback(f'{kind} at node {node:#x} not legal in JSettlers')
        out.update(cmd='settlement' if kind == 'build_settlement' else 'city', node=node)
    elif kind == 'build_road':
        edge = e2c[tuple(sorted(action['edge']))]
        if 'roads' in legal and edge not in legal['roads']:
            return fallback(f'road at edge {edge:#x} not legal in JSettlers')
        out.update(cmd='road', edge=edge)
    elif kind == 'buy_development':
        out.update(cmd='buy_dev')
    elif kind == 'bank_trade':
        give = Resource(action['give'])
        rate = game.trade_rate(ctx['seat'], give)
        js_rate = legal.get('rates', {}).get(str(RESOURCE_CODE[give]))
        if js_rate is not None and js_rate != rate:
            return fallback(f'bank rate {rate} for {give.value} but JSettlers rate {js_rate}')
        out.update(cmd='bank', give={str(RESOURCE_CODE[give]): rate},
                   get={str(RESOURCE_CODE[Resource(action['receive'])]): 1})
    elif kind == 'play_development':
        card = action['card']
        out.update(cmd='play_dev', card=card, card_type=DEV_CODES[card])
        if card == 'monopoly':
            out['resource'] = RESOURCE_CODE[Resource(action['resource'])]
        elif card == 'year_of_plenty':
            pick = {}
            for r in action['resources']:
                code = str(RESOURCE_CODE[Resource(r)])
                pick[code] = pick.get(code, 0) + 1
            if sum(pick.values()) != 2:
                return fallback('year of plenty must pick two resources in JSettlers')
            out['pick'] = pick
    elif kind == 'end_turn':
        out.update(cmd='end_turn')
    elif kind == 'roll':
        out.update(cmd='roll')
    elif kind == 'move_robber':
        hex_ = m.tile_to_hex[action['tile']]
        if 'robber_hexes' in legal and hex_ not in legal['robber_hexes']:
            return fallback(f'robber hex {hex_:#x} not legal in JSettlers')
        out.update(cmd='robber', hex=hex_)
    elif kind == 'steal':
        victim = seats.js(action['victim'])
        if 'victims' in legal and victim not in legal['victims']:
            return fallback(f'victim {victim} not offered by JSettlers')
        out.update(cmd='steal', victim=victim)
    elif kind in ('accept_trade', 'reject_trade'):
        out.update(cmd='accept' if kind == 'accept_trade' else 'reject')
    elif kind == 'offer_trade':
        out.update(cmd='offer', to=seats.js(action['recipient']), give=_codes(action['give']),
                   get=_codes(action['receive']))
    else:
        return fallback(f'unsupported engine action {kind}')
    return out


def legal_mismatches(snap: dict, obs, ctx) -> list[str]:
    """Differences between engine legal actions and JSettlers' legal sets for this decision."""
    legal = snap.get('js_legal')
    if not legal:
        return []
    m, seats = ctx['board_map'], ctx['seats']
    v2n, e2c = m.vertex_to_node, m.engine_to_edge
    actions = obs.actions
    out = []

    def compare(name, engine, js):
        engine, js = set(engine), set(js)
        if engine != js:
            out.append(f'{name}: engine-only {sorted(engine - js)} js-only {sorted(js - engine)}')

    decision = snap['decision']
    if decision in ('setup_settlement',) and 'settlements' in legal:
        compare('settlements', (v2n[a['vertex']] for a in actions), legal['settlements'])
    if decision in ('setup_road', 'free_roads') and 'roads' in legal:
        js_roads = legal['roads']
        if decision == 'setup_road':
            # JSettlers lists every potential road; setup requires the new settlement's edges.
            pending = obs.pending_setup_vertex
            js_roads = [e for e in js_roads if pending in m.edge_to_engine[e]]
        compare('roads', (e2c[tuple(a['edge'])] for a in actions if a['type'] == 'build_road'), js_roads)
    if decision == 'actions':
        if 'settlements' in legal and any(a['type'] == 'build_settlement' for a in actions):
            compare('settlements', (v2n[a['vertex']] for a in actions if a['type'] == 'build_settlement'),
                    legal['settlements'])
        if 'roads' in legal and any(a['type'] == 'build_road' for a in actions):
            compare('roads', (e2c[tuple(a['edge'])] for a in actions if a['type'] == 'build_road'),
                    legal['roads'])
        if 'cities' in legal and any(a['type'] == 'build_city' for a in actions):
            compare('cities', (v2n[a['vertex']] for a in actions if a['type'] == 'build_city'),
                    legal['cities'])
        if 'rates' in legal:
            rates = {str(RESOURCE_CODE[r]): n for r, n in obs.rates.items()}
            if rates != {k: int(v) for k, v in legal['rates'].items()}:
                out.append(f'rates: engine {rates} js {legal["rates"]}')
    if decision == 'robber' and 'robber_hexes' in legal:
        compare('robber_hexes', (m.tile_to_hex[a['tile']] for a in actions), legal['robber_hexes'])
    if decision == 'steal' and 'victims' in legal:
        compare('victims', (seats.js(a['victim']) for a in actions), legal['victims'])
    return out
