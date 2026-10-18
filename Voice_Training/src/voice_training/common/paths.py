"""Pure training address book. root -> manifests/config -> cache/data/output/logs.

Only constants are defined; all repository modules consume these paths.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL_MANIFEST = ROOT / 'deployment/models.json'
TEMPLATE_CONFIG = ROOT / 'configs/training.example.json'
CHARACTER_CONFIGS = ROOT / 'configs/characters'
CHARACTER_DATA = ROOT / 'data/characters'
LOGS = ROOT / 'logs'
TRAINING_LOCK = LOGS / 'training.lock'
ERROR_LOG = LOGS / 'training-error.log'
CACHE = ROOT / 'cache'
TEMP = ROOT / 'temp'
OUTPUTS = ROOT / 'outputs'
SOX = ROOT / 'tools/sox/sox-14.4.2'
