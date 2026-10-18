"""common/state.py: fingerprint, load_training_config, source_identity, manifest.

Call flow: CLI orchestration -> these functions -> shared common leaves/Qwen/PyTorch.
Data: project-relative configuration and checked manifests -> prepared data, caches,
adapter checkpoints or an exported character model. No sibling repository is imported.
"""
import hashlib
import json
import re

from .io import file_hash, inside_project, load_profile, read_json
from .paths import MODEL_MANIFEST


def fingerprint(value):
    """Hash canonical JSON to identify a model and dataset version."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def load_training_config(config_relative='configs/training.example.json'):
    """Resolve model IDs and validate role settings and project-relative paths."""
    cfg = read_json(inside_project(config_relative))
    models = read_json(MODEL_MANIFEST)['models']
    mapping = {item['id']: item['directory'] for item in models}
    cfg['student_model'] = mapping[cfg['student_model_id']]
    cfg['teacher_model'] = mapping[cfg['teacher_model_id']]
    if cfg.get('language') not in ('Chinese','English','Japanese','Korean','German','French','Russian','Portuguese','Spanish','Italian'):
        raise ValueError('训练语言必须明确指定为 Qwen 支持的语言，不能使用 Auto。')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', cfg['speaker_name']):
        raise ValueError('speaker_name 必须为安全的角色英文标识。')
    for key in ('teacher_model', 'student_model', 'profiles', 'prepared_directory', 'teacher_cache',
                'checkpoint_directory', 'export_directory'):
        inside_project(cfg[key])
    for key in ('epochs', 'gradient_accumulation', 'lora_rank', 'lora_alpha',
                'subcode_chunk_frames', 'checkpoint_every_steps'):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= 256:
            raise ValueError(f'{key} 必须是 1～256 的整数。')
    if not 0 < cfg['learning_rate'] <= 0.001 or not 0 <= cfg['kd_weight'] <= 1:
        raise ValueError('学习率或蒸馏权重不正确。')
    if not 0 < cfg['temperature'] <= 10 or not 0 < cfg['subcode_weight'] <= 1:
        raise ValueError('蒸馏温度或副音频单元权重不正确。')
    return cfg


def source_identity(cfg):
    """Verify model compatibility, recording hashes and train/validation membership."""
    student = inside_project(cfg['student_model'])
    teacher = inside_project(cfg['teacher_model'])
    print('核对学生、老师权重与数据版本…', flush=True)
    identities = {}
    for name, path in (('student', student), ('teacher', teacher)):
        if not (path / 'model.safetensors').is_file():
            raise ValueError(f'{name} 权重不存在：{path}')
        identities[name] = {file: file_hash(path / file) for file in
                            ('model.safetensors', 'config.json', 'vocab.json', 'merges.txt')}
    if any(identities['student'][name] != identities['teacher'][name] for name in ('vocab.json', 'merges.txt')):
        raise ValueError('老师和学生的文本编码词表不同，不能直接对齐蒸馏。')
    student_cfg = read_json(student / 'config.json')
    teacher_cfg = read_json(teacher / 'config.json')
    for key in ('vocab_size', 'num_code_groups', 'codec_eos_token_id', 'codec_bos_id',
                'codec_pad_id', 'codec_language_id', 'codec_think_id', 'codec_think_bos_id', 'codec_think_eos_id'):
        if student_cfg['talker_config'][key] != teacher_cfg['talker_config'][key]:
            raise ValueError('老师和学生的语音编码配置不同：' + key)
    profiles = read_json(inside_project(cfg['profiles']))
    selected, excluded = {}, {}
    for identifier, profile in sorted(profiles.items()):
        if re.search(r'[{}@]', profile['text']):
            excluded[identifier] = '台词含未处理的模板占位符，需人工核对'
        elif profile['duration_seconds'] > cfg['max_audio_seconds']:
            excluded[identifier] = '超过本轮完整录音长度上限；不做未经对齐的截断'
        else:
            selected[identifier] = load_profile(cfg, identifier)
    validation = set(cfg['validation_ids'])
    if not validation.issubset(selected) or cfg['reference_id'] not in selected or cfg['reference_id'] in validation:
        raise ValueError('验证集编号或训练参考录音不符合配置。')
    if len(selected) - len(validation) < 2:
        raise ValueError('有效训练录音不足。')
    identity = {'models': identities, 'profiles_sha256': file_hash(inside_project(cfg['profiles'])),
                'audio_sha256': {key: value['sha256'] for key, value in selected.items()},
                'validation_ids': sorted(validation), 'reference_id': cfg['reference_id'],
                'format': 'nonstream-' + cfg['language'].lower() + '-16codec-explicit-nexttoken-v1'}
    return selected, excluded, identity


def manifest(cfg):
    """Load prepared data only when its model, data and artifact hashes still match."""
    path = inside_project(cfg['prepared_directory']) / 'manifest.json'
    if not path.is_file():
        raise ValueError('请先运行 train.py prepare。')
    value = read_json(path)
    _, _, current = source_identity(cfg)
    if fingerprint(current) != value['signature']:
        raise ValueError('原始数据或模型改变，已拒绝混用旧训练缓存。')
    for row in value['samples']:
        path = inside_project(cfg['prepared_directory']) / row['sample']
        if file_hash(path) != row['sample_sha256']:
            raise ValueError('准备后的音频编码文件已改变：' + row['id'])
        audio = inside_project(cfg['prepared_directory']) / row['audio']
        if file_hash(audio) != row.get('audio_sha256'):
            raise ValueError('准备后的 WAV 文件已改变：' + row['id'])
    return value
