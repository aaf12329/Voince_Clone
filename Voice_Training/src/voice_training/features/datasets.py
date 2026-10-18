"""Role dataset import. import_character -> validate rows/audio -> profiles/config JSON.

Input: project-relative JSONL, each row {id,audio,text,split[,reviewed]}.
Output: {config,profiles,train_samples,validation_samples}, all paths project-relative.
Original recordings are read only; generated role metadata is excluded from Git.
"""
import re
from pathlib import Path, PureWindowsPath

from ..common.io import file_hash, inside_project, read_json, read_jsonl, write_json
from ..common.paths import CHARACTER_CONFIGS, CHARACTER_DATA, ROOT, TEMPLATE_CONFIG


def validate_row(row, manifest, seen, maximum):
    """Validate one row and derive a hashed profile; called by the import loop."""
    import soundfile as sf
    if not isinstance(row, dict):
        raise ValueError('每行必须是包含 id、audio、text、split 的 JSON 对象。')
    identifier = row.get('id', '')
    if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', identifier):
        raise ValueError('录音 id 必须是安全的英文/数字标识。')
    if identifier in seen:
        raise ValueError('数据清单包含重复编号：' + identifier)
    seen.add(identifier)
    text = row.get('text')
    if not isinstance(text, str) or not text.strip() or re.search(r'[{}@]', text):
        raise ValueError(identifier + ' 必须提供准确台词，并先处理模板占位符。')
    relative = row.get('audio', '')
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or PureWindowsPath(relative).is_absolute():
        raise ValueError(identifier + ' 的 audio 必须相对于 JSONL 所在目录。')
    audio = (manifest.parent / relative).resolve()
    if not audio.is_relative_to(ROOT.resolve()) or not audio.is_file():
        raise ValueError(identifier + ' 的录音不存在或位于训练项目之外。')
    info = sf.info(audio)
    if not 0.25 <= info.duration <= maximum:
        raise ValueError(f'{identifier} 时长必须在 0.25～{maximum} 秒之间；不会自动截断未对齐录音。')
    split = row.get('split', 'train')
    if split not in ('train', 'validation'):
        raise ValueError('split 只能填写 train 或 validation。')
    return identifier, {'audio': audio.relative_to(ROOT).as_posix(), 'text': text.strip(),
                        'sha256': file_hash(audio), 'duration_seconds': round(info.duration, 6),
                        'manual_listening_verified': row.get('reviewed') is True}, split


def import_character(name, manifest_relative, language='Chinese', reference_id=None):
    """Create a new isolated role configuration; refuse to overwrite an existing role."""
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('角色 Name 必须以英文字母开头，只能包含英文、数字、下划线和连字符。')
    config_path = CHARACTER_CONFIGS / (name + '.json')
    profile_path = CHARACTER_DATA / name / 'profiles.json'
    if config_path.exists() or profile_path.exists():
        raise ValueError('角色配置已存在，请使用新的角色名，避免覆盖旧训练结果。')
    config = read_json(TEMPLATE_CONFIG)
    if language not in ('Chinese','English','Japanese','Korean','German','French','Russian','Portuguese','Spanish','Italian'):
        raise ValueError('训练语言必须明确指定为支持的语言，不能使用 Auto。')
    manifest = inside_project(manifest_relative)
    profiles, train_ids, validation_ids, seen = {}, [], [], set()
    hashes = set()
    for row in read_jsonl(manifest):
        identifier, profile, split = validate_row(row, manifest, seen, config['max_audio_seconds'])
        if profile['sha256'] in hashes:
            raise ValueError('同一录音重复出现，可能造成训练/验证数据泄漏：' + identifier)
        hashes.add(profile['sha256'])
        profiles[identifier] = profile
        (train_ids if split == 'train' else validation_ids).append(identifier)
    if len(train_ids) < 2 or not validation_ids:
        raise ValueError('至少需要两条训练录音和一条独立验证录音；用 split 标明。')
    reference_id = reference_id or train_ids[0]
    if reference_id not in train_ids:
        raise ValueError('参考录音必须来自训练集，不能使用验证集录音。')
    config.update(speaker_name=name, language=language, reference_id=reference_id,
                  validation_ids=validation_ids, profiles=profile_path.relative_to(ROOT).as_posix(),
                  prepared_directory=f'data/prepared/{name}', teacher_cache=f'cache/teacher/{name}',
                  checkpoint_directory=f'checkpoints/{name}', export_directory=f'models/characters/{name}')
    write_json(profile_path, profiles)
    write_json(config_path, config)
    return {'config': config_path.relative_to(ROOT).as_posix(),
            'profiles': profile_path.relative_to(ROOT).as_posix(),
            'train_samples': len(train_ids), 'validation_samples': len(validation_ids)}
