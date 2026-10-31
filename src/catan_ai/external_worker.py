"""Local IPC worker for the pinned external policy; accepts trusted local inputs."""
import pickle
import socket
import struct
import sys
import time


def read_message(connection, deadline=None):
    def receive(count):
        result = bytearray()
        while len(result) < count:
            if deadline is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError('response deadline exceeded')
                connection.settimeout(remaining)
            block = connection.recv(count - len(result))
            if not block:
                raise EOFError('worker connection closed')
            result.extend(block)
        return bytes(result)
    length = struct.unpack('!I', receive(4))[0]
    if length > 16 * 1024 * 1024:
        raise ValueError('IPC message exceeds size limit')
    return pickle.loads(receive(length))


def send_message(connection, message):
    data = pickle.dumps(message, protocol=5)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError('IPC message exceeds size limit')
    connection.sendall(struct.pack('!I', len(data)) + data)


def verify_install():
    import hashlib
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    import subprocess
    import catanatron
    root = Path(__file__).resolve().parents[2]
    checkout = root / 'runs/external/catanatron'
    pin = 'ecf931181b9a65bb4116a2153fb78c16f1438e00'
    head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if head != pin:
        raise RuntimeError('external checkout does not match pinned commit')
    subprocess.run(['git', '-C', str(checkout), 'diff', '--exit-code', 'HEAD', '--',
                    'catanatron/catanatron', 'pyproject.toml'], check=True, capture_output=True)
    installed_root = Path(catanatron.__file__).parent
    source_root = checkout / 'catanatron/catanatron'
    hashes = {}
    for source in sorted(source_root.rglob('*.py')):
        relative = source.relative_to(source_root)
        installed = installed_root / relative
        if installed.read_bytes() != source.read_bytes():
            raise RuntimeError(f'external installed source mismatch: {relative}')
        hashes[str(relative)] = hashlib.sha256(installed.read_bytes()).hexdigest()
    versions = {'catanatron': '3.3.0', 'z3-solver': '4.15.4.0', 'click': '8.5.0', 'markdown-it-py': '4.2.0',
                'mdurl': '0.1.2', 'networkx': '3.7', 'pygments': '2.21.0', 'rich': '15.0.0'}
    for name, expected in versions.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f'external dependency version mismatch: {name}')
    return {'commit': pin, 'versions': versions, 'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
            'adaptation': 'depth2-guided-solver-v13',
            'package_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def main():
    from .recovery_belief import RecoveryBeliefTracker
    from .catanatron_adapter import from_observation, decision_to_local, choose_trade_response
    from catanatron.players.minimax import AlphaBetaPlayer
    import catanatron.players.minimax as minimax
    from catanatron.models.player import Color
    minimax.MAX_SEARCH_TIME_SECS = 1.5
    connection = socket.socket(fileno=int(sys.argv[1]))
    tracker = RecoveryBeliefTracker(int(sys.argv[2]), particles=512)
    identity = None
    while True:
        try:
            if identity is None:
                identity = verify_install()
            view, seed = read_message(connection)
        except EOFError:
            return
        try:
            external, mapping = from_observation(view, tracker, seed)
            if view.phase == 'trade_response':
                action = choose_trade_response(external, view)
            else:
                chosen = AlphaBetaPlayer(list(Color)[view.seat]).decide(external, external.playable_actions)
                action = decision_to_local(chosen, view, mapping)
            send_message(connection, {'action': action, 'identity': identity,
                                      'belief': {'solver_attempts': tracker.solver_attempts,
                                                 'solver_recoveries': tracker.solver_recoveries,
                                                 'solver_seconds': tracker.solver_seconds}})
        except Exception as exc:
            send_message(connection, {'error': type(exc).__name__ + ': ' + str(exc)})


if __name__ == '__main__':
    main()
