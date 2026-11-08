"""StacSettlers adapter over the JSettlers bridge (``tools/jsettlers/jsbridge``).

The STAC robot (``catanai.stac.CatanAiStacBrain``) sends snapshots in the JSettlers bridge format,
so reconstruction, policy calls and translation are ``jsbridge``'s. This adapter adds only what a
STAC match needs:

* Player trading is off for the match (STAC game option NT plus the patched server's block), so each
  game's policy has its end-of-turn offers switched off (``policy.offers = False``: the planner then
  ends the turn where it would have offered). A policy without that switch is refused (fail closed),
  and an ``offer_trade`` that still reaches translation becomes a counted fallback.
* The engine plays to 10 points; a snapshot with another win threshold is refused.
"""
from __future__ import annotations

from jsbridge.decide import POLICY, Bridge
from jsbridge.state import SnapshotError

VP_TO_WIN = 10


class TradingSwitchMissing(RuntimeError):
    pass


class StacBridge(Bridge):
    """``jsbridge.decide.Bridge`` with player-trade offers disabled per game."""

    def __init__(self, policy: str = POLICY, seed: int = 0):
        super().__init__(policy, seed)
        self.offers_suppressed = 0

    def policy_for(self, game: str):
        fresh = game not in self.policies
        policy = super().policy_for(game)
        if fresh:
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
        if reply.get('cmd') == 'offer':
            self.offers_suppressed += 1
            reply = {'cmd': 'fallback', 'reason': 'player trading is disabled for this match',
                     'action': reply.get('action'), 'suppressed_offer': True,
                     **{k: reply[k] for k in ('engine_legal', 'think_ms', 'legal_mismatches') if k in reply}}
        return reply
