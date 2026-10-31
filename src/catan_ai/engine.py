"""Select a rules backend while preserving the public engine interface."""
import hashlib
import importlib
import json
import os
from pathlib import Path


def _select_backend():
    requested = os.environ.get('CATAN_ENGINE', 'python')
    if requested not in {'python', 'cython', 'auto'}:
        raise ValueError('CATAN_ENGINE must be python, cython, or auto')
    if requested != 'python':
        root = Path(__file__).parent
        try:
            manifest = json.loads((root / '_native_manifest.json').read_text())
            if hashlib.sha256((root / '_engine_py.py').read_bytes()).hexdigest() != manifest['source_sha256']:
                raise RuntimeError('compiled engine source is stale; rebuild native engine')
            binary = root / manifest['binary']
            if binary.parent != root or hashlib.sha256(binary.read_bytes()).hexdigest() != manifest['binary_sha256']:
                raise RuntimeError('compiled binary does not match its build')
            if hashlib.sha256((root / 'scoring.py').read_bytes()).hexdigest() != manifest['scoring_source_sha256']:
                raise RuntimeError('compiled scoring source is stale; rebuild native engine')
            scoring_binary = root / manifest['scoring_binary']
            if scoring_binary.parent != root or hashlib.sha256(scoring_binary.read_bytes()).hexdigest() != manifest['scoring_binary_sha256']:
                raise RuntimeError('compiled scoring binary does not match its build')
            module = importlib.import_module('._engine_cy', __package__)
            if Path(module.__file__).resolve() != binary.resolve():
                raise RuntimeError('loaded binary differs from manifest')
            return module, 'cython'
        except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
            if requested == 'cython':
                raise RuntimeError('Cython engine unavailable: ' + str(exc)) from exc
    return importlib.import_module('._engine_py', __package__), 'python'


_implementation, BACKEND = _select_backend()
NATIVE_BINARY = _implementation.__file__ if BACKEND == 'cython' else None
globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith('_')})
