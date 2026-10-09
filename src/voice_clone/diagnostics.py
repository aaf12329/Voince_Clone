"""Export an allowlisted report without recordings, texts, raw logs or usernames."""
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from . import __version__
from .project import ROOT, LOGS, write_json


def read_object(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        return value if isinstance(value, dict) else {}
    except (ValueError, OSError):
        return {}


def create_report(root=ROOT):
    root = Path(root)
    hardware = read_object(root / 'logs/hardware-report.json')
    state = read_object(root / 'logs/deploy-state.json')
    runtime = read_object(root / 'logs/runtime-validation.json')
    versions = {}
    for package in ('torch', 'torchaudio', 'qwen-tts', 'transformers', 'PySide6', 'soundfile'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'application_version': __version__, 'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'os': platform.platform(), 'python': platform.python_version(), 'packages': versions,
            'hardware': {key: hardware.get(key) for key in ('cpu', 'logical_cores', 'ram_gib', 'eligible')},
            'gpus': [{key: gpu.get(key) for key in ('index', 'name', 'vram_mib', 'free_mib', 'driver', 'compute_capability')}
                     for gpu in hardware.get('gpus', []) if isinstance(gpu, dict)],
            'deployment_phase': state.get('phase'),
            'model_file_count': len(state.get('model_signatures', {})),
            'runtime': {key: runtime.get(key) for key in ('gpu', 'capability', 'dtype', 'inference_verified', 'cuda', 'torch')},
            'environment_exists': (root / '.venv/Scripts/python.exe').is_file(),
            'includes': '仅版本与硬件状态，不含录音、朗读文字、原始日志或个人路径。'}


def export_report(destination=None, root=ROOT):
    destination = Path(destination) if destination else Path(root) / 'logs/diagnostics.json'
    write_json(destination, create_report(root))
    return destination
