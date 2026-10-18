"""Environment leaf: configure_runtime -> project-local caches and offline model use.

Called once by the Python entry; no training, downloads or feature imports occur.
"""
import os
import sys

from .paths import CACHE, LOGS, OUTPUTS, SOX, TEMP


def configure_runtime():
    """Initialize the training process without relying on shell activation or Conda."""
    values = {'HF_HOME': CACHE / 'huggingface', 'HF_HUB_CACHE': CACHE / 'huggingface/hub',
              'TORCH_HOME': CACHE / 'torch', 'NUMBA_CACHE_DIR': CACHE / 'numba',
              'TEMP': TEMP, 'TMP': TEMP}
    os.environ.update({key: str(value) for key, value in values.items()})
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    os.environ['PATH'] = str(SOX) + os.pathsep + os.environ.get('PATH', '')
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    for path in (LOGS, OUTPUTS, TEMP, *values.values()):
        path.mkdir(parents=True, exist_ok=True)
