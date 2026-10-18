"""Pure address book: root -> config/manifest -> logs/output/runtime paths.

Imported by shared helpers and application entries; no I/O or model loading.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/runtime.json'
MODEL_MANIFEST = ROOT / 'deployment/models.json'
OUTPUTS = ROOT / 'outputs'
LOGS = ROOT / 'logs'
SAMPLES = ROOT / 'samples'
SETTINGS = LOGS / 'desktop-settings.json'
WORKER_LOG = LOGS / 'desktop-worker.log'
ERROR_LOG = LOGS / 'desktop-error.log'
WORKER = ROOT / 'desktop_worker.py'
IPC_NAME = 'VoiceCloneDesktop_' + hashlib.sha256(str(ROOT).encode('utf-8')).hexdigest()[:12]
CACHE = ROOT / 'cache'
TEMP = ROOT / 'temp'
SOX = ROOT / 'tools/sox/sox-14.4.2'

README = ROOT / 'README.md'
DIAGNOSTIC_REPORT = LOGS / 'diagnostics.json'
LANGUAGES = [('中文','Chinese'),('自动识别','Auto'),('英语','English'),('日语','Japanese'),('韩语','Korean'),('法语','French'),('德语','German'),('西班牙语','Spanish'),('俄语','Russian'),('葡萄牙语','Portuguese'),('意大利语','Italian')]
