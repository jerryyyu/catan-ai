"""Build the optional native backend from the authoritative Python rules."""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import Cython
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


class NativeBuild(build_ext):
    def run(self):
        super().run()
        if not self.inplace:
            raise RuntimeError('use build_ext --inplace for the optional local backend')
        binary = Path(self.get_ext_fullpath('catan_ai._engine_cy')).resolve()
        source = ROOT / 'src/catan_ai/_engine_py.py'
        manifest = {
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'binary': binary.name,
            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'cython': Cython.__version__, 'python': sys.version, 'host': platform.node(),
            'directives': {'language_level': 3, 'annotation_typing': False, 'binding': True},
        }
        scoring_binary = Path(self.get_ext_fullpath('catan_ai._scoring_cy')).resolve()
        manifest['scoring_source_sha256'] = hashlib.sha256((source.parent / 'scoring.py').read_bytes()).hexdigest()
        manifest['scoring_binary'] = scoring_binary.name
        manifest['scoring_binary_sha256'] = hashlib.sha256(scoring_binary.read_bytes()).hexdigest()
        (source.parent / '_native_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


setup(
    name='catan-ai-native', packages=['catan_ai'], package_dir={'': 'src'},
    ext_modules=cythonize(
        [Extension('catan_ai._engine_cy', ['src/catan_ai/_engine_py.py']),
         Extension('catan_ai._scoring_cy', ['src/catan_ai/scoring.py'])],
        build_dir='build/cython',
        compiler_directives={'language_level': 3, 'annotation_typing': False, 'binding': True},
    ),
    cmdclass={'build_ext': NativeBuild},
)
