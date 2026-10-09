"""Project-relative paths and atomic local JSON storage."""
import json
import os
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / 'outputs'
LOGS = ROOT / 'logs'
SETTINGS = LOGS / 'desktop-settings.json'
LANGUAGES = [('中文', 'Chinese'), ('自动识别', 'Auto'), ('英语', 'English'), ('日语', 'Japanese'),
             ('韩语', 'Korean'), ('法语', 'French'), ('德语', 'German'), ('西班牙语', 'Spanish'),
             ('俄语', 'Russian'), ('葡萄牙语', 'Portuguese'), ('意大利语', 'Italian')]


def ensure_directories():
    for directory in (OUTPUTS, LOGS, ROOT / 'temp'):
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    """Do not truncate an existing settings file on interrupted writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_settings():
    try:
        data = json.loads(SETTINGS.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            return {}
        return {key: value for key, value in data.items() if isinstance(value, str)}
    except (OSError, ValueError):
        return {}
