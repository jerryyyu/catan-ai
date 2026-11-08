"""Line-oriented JSON bridge between the JSettlers robot (Java) and the served policy.

One request per stdin line, one reply per stdout line, same ``id``. Kinds:
``decide`` (a snapshot), ``end_game`` (drop the game's policy), ``ping``.
Any exception becomes ``{"cmd": "error"}`` so the Java side can fall back and
count a bridge error; the process keeps serving.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

from .decide import POLICY, Bridge


def serve(stdin, stdout, bridge: Bridge, log=None) -> None:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        received = time.time()
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            reply = {'id': None, 'cmd': 'error', 'error': f'bad json: {exc}'}
            request = None
        else:
            kind = request.get('kind')
            try:
                if kind == 'decide':
                    reply = bridge.decide(request)
                elif kind == 'end_game':
                    bridge.end_game(request['game'])
                    reply = {'cmd': 'ok'}
                elif kind == 'ping':
                    reply = {'cmd': 'pong', 'policy': bridge.policy_name}
                else:
                    reply = {'cmd': 'error', 'error': f'unknown kind {kind!r}'}
            except Exception as exc:  # noqa: BLE001 - every failure is reported, never hidden
                reply = {'cmd': 'error', 'error': f'{type(exc).__name__}: {exc}',
                         'traceback': traceback.format_exc(limit=6)}
            reply['id'] = request.get('id')
        stdout.write(json.dumps(reply, separators=(',', ':')) + '\n')
        stdout.flush()
        if log is not None:
            log.write(json.dumps({'t': received, 'request': request, 'reply': reply},
                                 separators=(',', ':')) + '\n')
            log.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', default=POLICY)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log', help='append every request/reply as JSON lines')
    args = parser.parse_args(argv)
    log = open(args.log, 'a') if args.log else None
    try:
        serve(sys.stdin, sys.stdout, Bridge(args.policy, args.seed), log)
    finally:
        if log is not None:
            log.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
