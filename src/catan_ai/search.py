"""Root Monte Carlo improvement with fresh, bounded confirmation samples."""

from copy import deepcopy
from dataclasses import dataclass, asdict
import math
import random
import time

from .belief import BeliefFailure, BeliefTracker
from .bots import HeuristicBot
from .observation import Observation


@dataclass(frozen=True)
class SearchConfig:
    candidates: int = 5
    selection_samples: int = 4
    confirmation_samples: int = 32
    rollout_decisions: int = 1500
    particles: int = 128
    alpha: float = 0.05
    selection_rule: str = 'lcb'
    tactical: bool = False
    root_tactical: bool = False
    sequence_candidates: bool = False

    def __post_init__(self):
        if min(self.candidates, self.selection_samples, self.confirmation_samples,
               self.rollout_decisions, self.particles) < 1 or not 0 < self.alpha < 1:
            raise ValueError('positive budgets and alpha in (0, 1) required')
        if self.selection_rule not in {'lcb', 'mean'}:
            raise ValueError('selection_rule must be lcb or mean')


def improvement_lcb(differences: list[float], alpha: float) -> float:
    """One-sided Hoeffding bound for independent paired differences in [-1, 1]."""
    if not differences:
        return -1.0
    return max(-1.0, sum(differences) / len(differences)
               - math.sqrt(2 * math.log(1 / alpha) / len(differences)))


class MonteCarloBot:
    needs_history = True
    def __init__(self, seed: int, config: SearchConfig | None = None):
        self.config = config or SearchConfig()
        self.rng = random.Random(seed)
        self.baseline = HeuristicBot(random.Random(seed))
        if self.config.tactical or self.config.root_tactical:
            from .tactical import TacticalBot
            self.baseline = TacticalBot(random.Random(seed))
        self.belief = BeliefTracker(seed + 1000003, self.config.particles)
        self.diagnostics = []

    def _candidates(self, view, baseline):
        groups = {}
        for action in view.actions:
            key = (action['type'], action.get('card'))
            groups.setdefault(key, []).append(action)
        # Keep the baseline, then spread the budget across action families.
        candidates = [baseline]
        if self.config.sequence_candidates:
            from .tactical import turn_plan
            plan = turn_plan(view)
            if len(plan) > 1 and len(candidates) < self.config.candidates:
                candidates.append(plan)
        for actions in groups.values():
            useful_trade = (self.baseline._useful_trade(view, view.seat, actions)
                            if actions[0]['type'] == 'bank_trade' else None)
            actions.sort(key=lambda a: -1 if a == useful_trade
                         else -self.baseline._vertex_score(view, a['vertex'])
                         if 'vertex' in a else -self.baseline._road_score(view, view.seat, a['edge'])
                         if 'edge' in a else 0)
        while groups and len(candidates) < self.config.candidates:
            for key in list(groups):
                action = groups[key].pop(0)
                if action not in candidates:
                    candidates.append(action)
                if not groups[key]:
                    del groups[key]
                if len(candidates) >= self.config.candidates:
                    break
        return candidates

    def _rollout(self, world, action, seat, seed):
        game = deepcopy(world, {id(world.board): world.board})
        policies = [HeuristicBot(random.Random(seed * 7 + s)) for s in range(4)]
        if self.config.tactical:
            from .tactical import TacticalBot
            policies[seat] = TacticalBot(random.Random(seed * 7 + seat))
        policies[seat].last_offer_turn = self.baseline.last_offer_turn
        for step in action if isinstance(action, list) else [action]:
            game.apply(seat, step)
            if game.winner is not None:
                break
        for _ in range(self.config.rollout_decisions):
            if game.winner is not None:
                return float(game.winner == seat)
            actor = game.current
            game.apply(actor, policies[actor].choose(Observation.for_policy(game, actor, policies[actor]), actor))
        return float(game.winner == seat) if game.winner is not None else None

    def choose(self, view: Observation, seat: int) -> dict:
        if seat != view.seat:
            raise ValueError('search requires the acting player observation')
        started = time.perf_counter()
        baseline = self.baseline.choose(view, seat)
        record = {'turn': view.turn, 'phase': view.phase, 'baseline': deepcopy(baseline),
                  'chosen': deepcopy(baseline), 'reason': 'unsupported_phase', 'rollouts': 0,
                  'truncated': 0, 'values': [], 'confirmation_differences': [], 'lcb': None,
                  'config': asdict(self.config)}
        chosen = baseline
        try:
            if view.phase != 'actions':
                return chosen
            candidates = self._candidates(view, baseline)
            if len(candidates) < 2:
                record['reason'] = 'single_candidate'
                return chosen
            self.belief.update(view)
            values = [[] for _ in candidates]
            for _ in range(self.config.selection_samples):
                seed = self.rng.getrandbits(63)
                world = self.belief.sample(view, seed)
                for i, action in enumerate(candidates):
                    value = self._rollout(world, action, seat, seed)
                    record['rollouts'] += 1
                    record['truncated'] += value is None
                    values[i].append(value)
            record['values'] = [{'action': deepcopy(a), 'wins': sum(v for v in vs if v is not None),
                                 'completed': sum(v is not None for v in vs), 'samples': len(vs)}
                                for a, vs in zip(candidates, values)]
            if record['truncated']:
                record['reason'] = 'selection_truncated'
                return chosen
            challenger = max(range(len(candidates)), key=lambda i: sum(values[i]))
            if challenger == 0:
                record['reason'] = 'baseline_selected'
                return chosen
            differences = []
            for _ in range(self.config.confirmation_samples):
                seed = self.rng.getrandbits(63)
                world = self.belief.sample(view, seed)
                a = self._rollout(world, candidates[challenger], seat, seed)
                b = self._rollout(world, baseline, seat, seed)
                record['rollouts'] += 2
                record['truncated'] += (a is None) + (b is None)
                if a is not None and b is not None:
                    differences.append(a - b)
            record['confirmation_differences'] = differences
            if record['truncated']:
                record['reason'] = 'confirmation_truncated'
                return chosen
            record['lcb'] = improvement_lcb(differences, self.config.alpha)
            record['reason'] = 'not_confirmed'
            accepted = (record['lcb'] > 0 if self.config.selection_rule == 'lcb'
                        else sum(differences) > 0)
            if accepted:
                chosen = candidates[challenger]
                if isinstance(chosen, list):
                    chosen = chosen[0]
                record['reason'] = 'confirmed' if self.config.selection_rule == 'lcb' else 'mean_selected'
            return deepcopy(chosen)
        except BeliefFailure as exc:
            record['reason'] = 'belief_failure'
            record['detail'] = str(exc)
            return baseline
        finally:
            record['chosen'] = deepcopy(chosen)
            record['seconds'] = time.perf_counter() - started
            self.diagnostics.append(record)
