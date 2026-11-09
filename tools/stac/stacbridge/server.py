"""Python side of the StacSettlers bridge: ``jsbridge.server.serve`` with the STAC adapter."""
from __future__ import annotations

import argparse
import sys

from jsbridge.server import serve

from .decide import POLICY, StacBridge


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', default=POLICY)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log', help='append every request/reply as JSON lines')
    parser.add_argument('--player-trades', action='store_true',
                        help='keep the policy\'s trade offers (STAC chat-negotiation matches)')
    args = parser.parse_args(argv)
    log = open(args.log, 'a') if args.log else None
    try:
        serve(sys.stdin, sys.stdout, StacBridge(args.policy, args.seed, player_trades=args.player_trades), log)
    finally:
        if log is not None:
            log.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
