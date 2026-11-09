"""StacSettlers adapter over the JSettlers bridge (``tools/jsettlers/jsbridge``).

The STAC robot (``catanai.stac.CatanAiStacBrain``) sends snapshots in the JSettlers bridge format,
so reconstruction, policy calls and translation are ``jsbridge``'s. This adapter adds only what a
STAC match needs:

* Player trading follows the match mode. Off (the default): STAC's server blocks player exchanges,
  so each game's policy has its end-of-turn offers switched off (``policy.offers = False``: the
  planner then ends the turn where it would have offered). A policy without that switch is refused
  (fail closed), and an ``offer_trade`` that still reaches translation becomes a counted fallback.
  On (``player_trades=True``, STAC chat negotiation): the policy keeps its offers and answers, and
  the Java brain carries them in STAC's chat trade format.
* The engine plays to 10 points; a snapshot with another win threshold is refused.
"""
from __future__ import annotations

from jsbridge.decide import POLICY, Bridge
from jsbridge.state import SnapshotError

VP_TO_WIN = 10


class TradingSwitchMissing(RuntimeError):
    pass


class StacBridge(Bridge):
    """``jsbridge.decide.Bridge``; player-trade offers disabled per game unless ``player_trades``."""

    def __init__(self, policy: str = POLICY, seed: int = 0, player_trades: bool = False):
        super().__init__(policy, seed)
        self.player_trades = player_trades
        self.offers_suppressed = 0

    def policy_for(self, game: str):
        fresh = game not in self.policies
        policy = super().policy_for(game)
        if fresh and not self.player_trades:
            if not hasattr(policy, 'offers'):
                raise TradingSwitchMissing(f'policy {self.policy_name} has no offers switch; '
                                           'cannot disable player trading')
            policy.offers = False
        return policy

    def decide(self, snap: dict) -> dict:
        vp = snap.get('vp_to_win', VP_TO_WIN)
        if int(vp) != VP_TO_WIN:
            raise SnapshotError(f'STAC game plays to {vp} points; the engine plays to {VP_TO_WIN}')
        reply = super().decide(snap)
        if reply.get('cmd') == 'offer' and not self.player_trades:
            self.offers_suppressed += 1
            reply = {'cmd': 'fallback', 'reason': 'player trading is disabled for this match',
                     'action': reply.get('action'), 'suppressed_offer': True,
                     **{k: reply[k] for k in ('engine_legal', 'think_ms', 'legal_mismatches') if k in reply}}
        return reply
