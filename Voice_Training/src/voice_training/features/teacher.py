"""features/teacher.py: teacher_cache.

Call flow: CLI orchestration -> these functions -> shared common leaves/Qwen/PyTorch.
Data: project-relative configuration and checked manifests -> prepared data, caches,
adapter checkpoints or an exported character model. No sibling repository is imported.
"""
import time

from ..common.io import file_hash, inside_project, read_json, write_json
from ..common.tensor_ops import atomic_torch_save, torch_load, release_gpu, load_tts


def teacher_cache(cfg, data):
    """Persist frozen teacher logits tied to the prepared dataset signature."""
    import soundfile as sf
    import torch
    from ..common.lora import codec_distributions
    directory = inside_project(cfg['teacher_cache'])
    directory.mkdir(parents=True, exist_ok=True)
    cache_manifest = directory / 'manifest.json'
    if cache_manifest.is_file():
        cache = read_json(cache_manifest)
        if cache['signature'] != data['signature']:
            raise ValueError('老师缓存与数据不一致。请为本轮配置新 teacher_cache。')
    else:
        cache = {'signature': data['signature'], 'files': {}, 'teacher_frozen': True,
                 'all_codebooks': 16, 'conditioning': 'fixed_training_reference_cn_chinese_nonstream'}
    pending = [row for row in data['samples'] if row['id'] not in cache['files'] or
               not (directory / (row['id'] + '.pt')).is_file() or
               file_hash(directory / (row['id'] + '.pt')) != cache['files'][row['id']]]
    if not pending:
        print('复用已校验的 1.7B 老师分布。', flush=True)
        return cache
    wrapper = load_tts(inside_project(cfg['teacher_model']))
    wrapper.model.eval().requires_grad_(False)
    prepared = inside_project(cfg['prepared_directory'])
    ref_wav, sr = sf.read(str(prepared / 'audio' / (cfg['reference_id'] + '.wav')), dtype='float32')
    with torch.no_grad():
        speaker = wrapper.model.extract_speaker_embedding(ref_wav, sr).detach()
        for number, row in enumerate(pending, 1):
            started = time.perf_counter()
            sample = torch_load(prepared / row['sample'])
            first, sub, _, _ = codec_distributions(wrapper.model, sample, speaker, cfg['subcode_chunk_frames'], cfg['language'].lower())
            target = directory / (row['id'] + '.pt')
            atomic_torch_save(target, {'first': first.cpu().to(torch.float16),
                                       'sub': sub.cpu().to(torch.float16), 'signature': data['signature']})
            cache['files'][row['id']] = file_hash(target)
            write_json(cache_manifest, cache)
            print(f'老师分布 {number}/{len(pending)} {row["id"]}，{time.perf_counter()-started:.1f} 秒', flush=True)
            del first, sub
    del wrapper, speaker
    release_gpu()
    return cache
