"""features/preparation.py: prepare.

Call flow: CLI orchestration -> these functions -> shared common leaves/Qwen/PyTorch.
Data: project-relative configuration and checked manifests -> prepared data, caches,
adapter checkpoints or an exported character model. No sibling repository is imported.
"""

from ..common.io import file_hash, inside_project, read_json, write_json
from ..common.state import fingerprint, source_identity
from ..common.tensor_ops import atomic_torch_save, release_gpu


def prepare(cfg):
    """Validate recordings and cache aligned 24 kHz audio and codec tensors."""
    import numpy as np
    import soundfile as sf
    import librosa
    from qwen_tts import Qwen3TTSTokenizer
    from transformers import Qwen2TokenizerFast
    from qwen_tts.core.models import Qwen3TTSProcessor
    from ..common.lora import text_ids
    selected, excluded, identity = source_identity(cfg)
    directory = inside_project(cfg['prepared_directory'])
    manifest_path = directory / 'manifest.json'
    signature = fingerprint(identity)
    if manifest_path.is_file():
        old = read_json(manifest_path)
        if old['signature'] == signature and all(
                (directory / row['sample']).is_file() and file_hash(directory / row['sample']) == row['sample_sha256']
                and (directory / row['audio']).is_file()
                and file_hash(directory / row['audio']) == row.get('audio_sha256')
                for row in old['samples']):
            print('复用已校验的编码数据。', flush=True)
            return old
        raise ValueError('原始数据版本已改变。请配置新的 prepared_directory，避免混用训练数据。')
    directory.mkdir(parents=True, exist_ok=True)
    student = inside_project(cfg['student_model'])
    processor = Qwen3TTSProcessor(Qwen2TokenizerFast.from_pretrained(str(student), local_files_only=True,
                                                                   fix_mistral_regex=True))
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(student / 'speech_tokenizer'),
                                                 device_map='cuda:0', local_files_only=True)
    samples = []
    for number, (identifier, profile) in enumerate(selected.items(), 1):
        wav, sr = sf.read(profile['audio_path'], dtype='float32', always_2d=True)
        wav = wav.mean(axis=1)
        if sr != 24000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
        if not np.isfinite(wav).all() or np.max(np.abs(wav)) < 1e-5:
            raise ValueError(f'{identifier} 音频无效。')
        destination = directory / 'audio' / (identifier + '.wav')
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, wav, 24000, subtype='PCM_16')
        # Encode the same PCM waveform used later for speaker extraction.
        codes = tokenizer.encode(str(destination)).audio_codes[0].detach().cpu().long()
        if codes.ndim != 2 or codes.shape[1] != 16 or not len(codes) or codes.min() < 0 or codes.max() >= 2048:
            raise ValueError(f'{identifier} 音频编码形状或范围错误。')
        role, content = text_ids(processor, profile['text'])
        sample = {'codes': codes, 'role_ids': role.cpu(), 'content_ids': content.cpu()}
        sample_name = 'samples/' + identifier + '.pt'
        atomic_torch_save(directory / sample_name, sample)
        samples.append({'id': identifier, 'text': profile['text'], 'sample': sample_name,
                        'sample_sha256': file_hash(directory / sample_name),
                        'audio': 'audio/' + identifier + '.wav', 'frames': len(codes),
                        'audio_sha256': file_hash(destination),
                        'duration_seconds': round(len(wav) / 24000, 3),
                        'split': 'validation' if identifier in cfg['validation_ids'] else 'train',
                        'manual_listening_verified': profile['manual_listening_verified']})
        print(f'编码 {number}/{len(selected)} {identifier}: {len(codes)} 帧', flush=True)
    del tokenizer, processor
    release_gpu()
    manifest = {'signature': signature, 'source_identity': identity, 'samples': samples,
                'excluded': excluded, 'training_ids': [v['id'] for v in samples if v['split'] == 'train'],
                'validation_ids': cfg['validation_ids'], 'reference_id': cfg['reference_id'],
                'total_seconds': round(sum(v['duration_seconds'] for v in samples), 3),
                'manual_transcript_review_complete': False}
    write_json(manifest_path, manifest)
    return manifest
