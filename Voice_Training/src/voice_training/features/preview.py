"""Export preview: validate_export -> local model -> custom voice generation -> WAV.

Input: role configuration and new text. Output: {path,speaker,duration_seconds}.
No reference recording or base-model fallback is used for an exported character.
"""
from ..common.io import file_hash, inside_project, read_json


def validate_export(directory, speaker):
    """Reject untrained, altered or mismatched exports before GPU model loading."""
    metadata = read_json(directory / 'distillation.json')
    config = read_json(directory / 'config.json')
    if metadata.get('checkpoint_step', 0) < 1 or metadata.get('speaker') != speaker:
        raise ValueError('导出模型没有有效的角色训练记录。')
    if config.get('tts_model_type') != 'custom_voice':
        raise ValueError('该目录不是训练后导出的角色模型。')
    if file_hash(directory / 'model.safetensors') != metadata.get('model_sha256'):
        raise ValueError('导出模型权重校验失败，请重新导出或检查文件。')
    return metadata


def generate(config, text, output_relative):
    """Speak with one exported role model for the CLI entry."""
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    if not text.strip() or len(text) > 1500:
        raise ValueError('朗读文字必须为 1～1500 字。')
    model_path = inside_project(config['export_directory'])
    validate_export(model_path, config['speaker_name'])
    if not torch.cuda.is_available():
        raise RuntimeError('需要可用的 NVIDIA CUDA 显卡，请运行 Check.ps1。')
    model = Qwen3TTSModel.from_pretrained(str(model_path), device_map='cuda:0', dtype=torch.bfloat16,
                                        attn_implementation='sdpa', local_files_only=True)
    with torch.inference_mode():
        waves, rate = model.generate_custom_voice(text=text, language=config['language'],
            speaker=config['speaker_name'], non_streaming_mode=True, max_new_tokens=1024)
    wave = np.asarray(waves[0], dtype='float32')
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('试听结果无效，未写入音频。')
    destination = inside_project(output_relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, wave, rate, subtype='PCM_16')
    return {'path': destination.relative_to(inside_project('.')).as_posix(),
            'speaker': config['speaker_name'], 'duration_seconds': round(len(wave) / rate, 3)}
