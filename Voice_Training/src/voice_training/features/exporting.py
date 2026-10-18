"""features/exporting.py: export.

Call flow: CLI orchestration -> these functions -> shared common leaves/Qwen/PyTorch.
Data: project-relative configuration and checked manifests -> prepared data, caches,
adapter checkpoints or an exported character model. No sibling repository is imported.
"""
import shutil
import uuid

from ..common.io import file_hash, inside_project, read_json, write_json
from ..common.state import manifest
from ..common.tensor_ops import torch_load, release_gpu, load_tts


def export(cfg):
    """Verify the best checkpoint and atomically publish a merged character model."""
    import torch
    from safetensors.torch import save_file
    from ..common.lora import attach_lora, load_adapter, merge_lora
    directory = inside_project(cfg['checkpoint_directory'])
    checkpoint = directory / 'best.pt'
    saved = torch_load(checkpoint)
    if saved['training_config'] != cfg:
        raise ValueError('导出配置与训练检查点不一致。')
    data = manifest(cfg)
    if saved['data_signature'] != data['signature']:
        raise ValueError('导出检查点与当前数据不一致。')
    if saved['step'] == 0:
        print('训练后的验证损失没有超过底座，暂不导出为可用角色模型。检查点保留。', flush=True)
        return None
    checkpoint_hash = file_hash(checkpoint)
    destination = inside_project(cfg['export_directory'])
    if destination.exists():
        metadata = destination / 'distillation.json'
        if metadata.is_file() and read_json(metadata)['checkpoint_sha256'] == checkpoint_hash:
            print('本检查点已经导出。', flush=True)
            return destination
        raise ValueError('导出目录已有其他版本，请修改 export_directory；不覆盖已有模型。')
    wrapper = load_tts(inside_project(cfg['student_model']))
    attach_lora(wrapper.model, cfg['lora_rank'], cfg['lora_alpha'])
    load_adapter(wrapper.model, saved['adapters'])
    merge_lora(wrapper.model)
    with torch.no_grad():
        wrapper.model.talker.get_input_embeddings().weight[3000].copy_(saved['speaker'].to('cuda:0'))
    staging = destination.with_name(destination.name + '.staging-' + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    source = inside_project(cfg['student_model'])
    for path in source.rglob('*'):
        if path.is_file() and path.relative_to(source).as_posix() != 'model.safetensors':
            target = staging / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    model_config = read_json(source / 'config.json')
    model_config['tts_model_type'] = 'custom_voice'
    model_config['talker_config']['spk_id'] = {cfg['speaker_name']: 3000}
    model_config['talker_config']['spk_is_dialect'] = {cfg['speaker_name']: False}
    write_json(staging / 'config.json', model_config)
    state = {key: value.detach().cpu().contiguous() for key, value in wrapper.model.state_dict().items()
             if not key.startswith('speaker_encoder.')}
    save_file(state, str(staging / 'model.safetensors'))
    write_json(staging / 'distillation.json', {'checkpoint_sha256': checkpoint_hash,
               'checkpoint_step': saved['step'], 'data_signature': saved['data_signature'],
               'model_sha256': file_hash(staging / 'model.safetensors'),
               'validation': saved['evaluation'], 'speaker': cfg['speaker_name'], 'quality_verified': False,
               'experimental': True, 'method': '16-codebook KL + real-recording CE, merged LoRA'})
    staging.rename(destination)
    del wrapper, state, saved
    release_gpu()
    print(f'已导出实验角色模型：{destination}', flush=True)
    return destination
