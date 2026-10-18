"""Shared I/O entry points: read_json/write_json, append_text and save_wave.

Call tree: entries/runtime -> these leaves -> JSON/pathlib or soundfile.
Contracts: JSON -> dict/list; generated waveform -> validated output Path.
"""
import json
import os
import uuid

from .paths import OUTPUTS, SETTINGS
from .audio import write_result


def read_json(path):
    """Read UTF-8 JSON with optional BOM for the application callers."""
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    """Atomically persist JSON, called by desktop settings handling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def append_text(path, text):
    """Append UTF-8 diagnostic text through one shared text entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(text)


def save_wave(wave, rate):
    """Publish generated audio through the shared binary audio writer."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    destination = OUTPUTS / (uuid.uuid4().hex + '.wav')
    write_result(destination, wave, rate)
    return destination


def read_settings(path=None):
    """Read optional local settings while rejecting unexpected schema values."""
    try:
        value = read_json(path or SETTINGS)
        return {key: item for key, item in value.items() if isinstance(item, str)} if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}
