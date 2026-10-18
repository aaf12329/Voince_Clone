"""Shared training I/O and bounded relative path resolution.

Structure: paths -> JSON/text I/O -> hashes/profile records.
Call tree: stage/CLI -> inside_project/read_json/write_json/append_jsonl;
source identity -> load_profile -> file_hash. No feature module is imported.
"""
import hashlib
import json
import os
import uuid
from pathlib import Path, PureWindowsPath

from .paths import ROOT


def inside_project(relative):
    """Resolve a project-relative path and reject drive paths, traversal and escaped links."""
    value = Path(relative)
    if value.is_absolute() or PureWindowsPath(str(relative)).is_absolute():
        raise ValueError('路径必须相对于训练项目，不能写盘符或绝对路径。')
    result = (ROOT / value).resolve()
    if not result.is_relative_to(ROOT.resolve()):
        raise ValueError('路径不能超出训练项目目录。')
    return result


def read_json(path):
    """Read one structured UTF-8 file for configuration/manifest callers."""
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    """Read manifest rows through the structured I/O entry, retaining line diagnostics."""
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f'数据清单第 {number} 行不是有效 JSON。') from exc
    return rows


def write_json(path, value):
    """Publish JSON atomically, used by every training stage."""
    path = Path(path)
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


def append_jsonl(path, value):
    """Append a structured optimizer record through the shared JSON I/O module."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def append_text(path, text):
    """Append diagnostic text for the CLI error boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(text)


def file_hash(path):
    """Compute SHA256 for source identity and checkpoint callers."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_profile(config, identifier=None):
    """Validate one role recording/transcript against its imported content hash."""
    profiles = read_json(inside_project(config['profiles']))
    identifier = identifier or config['reference_id']
    if identifier not in profiles:
        raise ValueError('数据清单不存在录音编号：' + identifier)
    profile = dict(profiles[identifier])
    audio = inside_project(profile['audio'])
    if not audio.is_file() or file_hash(audio) != profile['sha256']:
        raise ValueError('录音已改变或缺失，请重新导入数据，不能混用旧台词和训练缓存。')
    if not isinstance(profile.get('text'), str) or not profile['text'].strip():
        raise ValueError('每条录音必须提供准确的完整台词。')
    profile.update(id=identifier, audio_path=str(audio))
    return profile
