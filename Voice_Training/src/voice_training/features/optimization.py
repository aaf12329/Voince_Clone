"""features/optimization.py: sample_loss, evaluate, train.

Call flow: CLI orchestration -> these functions -> shared common leaves/Qwen/PyTorch.
Data: project-relative configuration and checked manifests -> prepared data, caches,
adapter checkpoints or an exported character model. No sibling repository is imported.
"""
import json
import math
import random
import time

from ..common.io import append_jsonl, inside_project, read_json, write_json
from ..common.state import fingerprint
from ..common.tensor_ops import atomic_torch_save, torch_load, release_gpu, load_tts


def sample_loss(cfg, wrapper, sample, cached, speaker):
    """Evaluate one encoded sample against real codes and frozen teacher logits."""
    from ..common.lora import codec_distributions, distribution_loss
    first, sub, labels, sub_labels = codec_distributions(wrapper.model, sample, speaker, cfg['subcode_chunk_frames'], cfg['language'].lower())
    args = (cfg['temperature'], cfg['kd_weight'])
    main, ce, kd = distribution_loss(first, cached['first'], labels, *args)
    extra, sub_ce, sub_kd = distribution_loss(sub, cached['sub'], sub_labels, *args)
    loss = main + cfg['subcode_weight'] * extra
    metrics = {name: float(value.detach()) for name, value in
               (('loss', loss), ('ce', ce), ('kd', kd), ('sub_ce', sub_ce), ('sub_kd', sub_kd))}
    return loss, metrics


def evaluate(cfg, wrapper, data, speaker):
    """Aggregate held-out sample metrics with gradients disabled."""
    import torch
    rows = [row for row in data['samples'] if row['split'] == 'validation']
    values = []
    was_training = wrapper.model.training
    wrapper.model.eval()
    try:
        with torch.no_grad():
            for row in rows:
                sample = torch_load(inside_project(cfg['prepared_directory']) / row['sample'])
                cached = torch_load(inside_project(cfg['teacher_cache']) / (row['id'] + '.pt'))
                _, item = sample_loss(cfg, wrapper, sample, cached, speaker)
                values.append({'id': row['id'], **item})
        return {'per_utterance': values, 'mean': {key: sum(v[key] for v in values) / len(values)
                                                  for key in ('loss', 'ce', 'kd', 'sub_ce', 'sub_kd')}}
    finally:
        wrapper.model.train(was_training)


def train(cfg, data, resume=False, max_steps=None):
    """Optimize adapters and speaker embeddings with validated cache and resume state."""
    import soundfile as sf
    import torch
    from ..common.lora import adapter_state, attach_lora, load_adapter
    torch.manual_seed(cfg['seed'])
    directory = inside_project(cfg['checkpoint_directory'])
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / 'latest.pt'
    signature = fingerprint({'data': data['signature'], 'training': cfg})
    if checkpoint.is_file() and not resume:
        raise ValueError('已有检查点。请追加 --resume 或配置新的 checkpoint_directory。')
    saved = torch_load(checkpoint) if checkpoint.is_file() else None
    if saved is not None:
        if saved['signature'] != signature:
            raise ValueError('检查点与模型、数据或训练配置不一致。')
        if saved['epoch'] >= cfg['epochs'] and (directory / 'report.json').is_file():
            print(f'本轮已完成 {saved["step"]} 步；保留原训练报告，不重复训练。', flush=True)
            return read_json(directory / 'report.json')
    wrapper = load_tts(inside_project(cfg['student_model']))
    reference = inside_project(cfg['prepared_directory']) / 'audio' / (cfg['reference_id'] + '.wav')
    wave, sr = sf.read(str(reference), dtype='float32')
    with torch.no_grad():
        speaker = wrapper.model.extract_speaker_embedding(wave, sr).detach().clone()
    layers = attach_lora(wrapper.model, cfg['lora_rank'], cfg['lora_alpha'])
    parameters = [p for p in wrapper.model.parameters() if p.requires_grad]
    wrapper.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    wrapper.model.config.use_cache = False
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=0.01)
    epoch, cursor, step = 0, 0, 0
    baseline, best_loss = None, float('inf')
    if saved is not None:
        load_adapter(wrapper.model, saved['adapters'])
        optimizer.load_state_dict(saved['optimizer'])
        speaker = saved['speaker'].to('cuda:0', dtype=wrapper.model.talker.get_input_embeddings().weight.dtype)
        epoch, cursor, step = saved['epoch'], saved['cursor'], saved['step']
        baseline, best_loss = saved['baseline'], saved['best_loss']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        del saved
    total_params = sum(p.numel() for p in parameters)
    print(f'学生 LoRA：{len(layers)} 层、{total_params:,} 个可训练参数；从第 {step} 步继续。', flush=True)

    def save(status, evaluation=None, best=False):
        """Checkpoint optimizer, adapters, speaker and resume cursor atomically."""
        state = {'signature': signature, 'data_signature': data['signature'], 'training_config': cfg,
                 'adapters': adapter_state(wrapper.model), 'optimizer': optimizer.state_dict(),
                 'speaker': speaker.cpu(), 'epoch': epoch, 'cursor': cursor, 'step': step,
                 'baseline': baseline, 'best_loss': best_loss, 'evaluation': evaluation,
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(), 'status': status}
        atomic_torch_save(directory / ('best.pt' if best else 'latest.pt'), state)
        if not best:
            write_json(directory / 'state.json', {'status': status, 'step': step, 'epoch': epoch, 'cursor': cursor,
                'trainable_parameters': total_params, 'lora_layers': len(layers), 'data_signature': data['signature'],
                'baseline': baseline, 'evaluation': evaluation, 'teacher_frozen': True,
                'distillation_started': step > 0, 'quality_verified': False})

    if baseline is None:
        print('计算独立验证集的训练前指标…', flush=True)
        baseline = evaluate(cfg, wrapper, data, speaker)
        best_loss = baseline['mean']['loss']
        save('initialized', baseline)
        save('initialized', baseline, best=True)
    rows = [row for row in data['samples'] if row['split'] == 'train']
    initial_adapters = adapter_state(wrapper.model)
    losses = []
    log_path = directory / 'training.jsonl'
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        while epoch < cfg['epochs'] and (max_steps is None or step < max_steps):
            order = list(rows)
            random.Random(cfg['seed'] + epoch).shuffle(order)
            wrapper.model.train()
            while cursor < len(order) and (max_steps is None or step < max_steps):
                group = order[cursor:cursor + cfg['gradient_accumulation']]
                metrics = []
                for row in group:
                    sample = torch_load(inside_project(cfg['prepared_directory']) / row['sample'])
                    cached = torch_load(inside_project(cfg['teacher_cache']) / (row['id'] + '.pt'))
                    if cached['signature'] != data['signature']:
                        raise ValueError('老师缓存版本不一致。')
                    loss, item = sample_loss(cfg, wrapper, sample, cached, speaker)
                    if not torch.isfinite(loss):
                        raise RuntimeError('训练损失不是有限值，已停止。')
                    (loss / len(group)).backward()
                    metrics.append(item)
                    del loss, sample, cached
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                cursor += len(group)
                step += 1
                item = {'step': step, 'epoch': epoch + 1, 'ids': [r['id'] for r in group],
                        **{key: sum(m[key] for m in metrics) / len(metrics) for key in metrics[0]},
                        'gradient_norm': float(norm), 'elapsed_seconds': round(time.perf_counter()-started, 2),
                        'peak_allocated_gib': round(torch.cuda.max_memory_allocated()/1024**3, 3)}
                losses.append(item)
                append_jsonl(log_path, item)
                print(f'训练步 {step}：loss={item["loss"]:.4f} CE={item["ce"]:.4f} '
                      f'KD={item["kd"]:.4f}，显存 {item["peak_allocated_gib"]:.2f} GiB', flush=True)
                if step % cfg['checkpoint_every_steps'] == 0:
                    save('training')
            if cursor >= len(order):
                epoch += 1
                cursor = 0
            current = evaluate(cfg, wrapper, data, speaker)
            if current['mean']['loss'] < best_loss:
                best_loss = current['mean']['loss']
                save('evaluated', current, best=True)
            save('complete' if epoch >= cfg['epochs'] else 'paused_at_step_limit', current)
        final = evaluate(cfg, wrapper, data, speaker)
        current_adapters = adapter_state(wrapper.model)
        delta = math.sqrt(sum(float((current_adapters[key] - value).double().square().sum())
                              for key, value in initial_adapters.items()))
        report = {'status': 'complete' if epoch >= cfg['epochs'] else 'paused_at_step_limit',
                  'optimizer_steps': step, 'completed_epochs': epoch, 'train_samples': len(rows),
                  'validation_samples': len(cfg['validation_ids']), 'trainable_parameters': total_params,
                  'adapter_change_l2_this_session': delta, 'baseline': baseline, 'final': final,
                  'teacher': 'Qwen3-TTS-12Hz-1.7B-Base (frozen)', 'student': 'Qwen3-TTS-12Hz-0.6B-Base + LoRA',
                  'method': 'all-16-codebook teacher-forced KL distillation + real-recording CE',
                  'elapsed_seconds_this_session': round(time.perf_counter()-started, 2),
                  'peak_allocated_gib': round(torch.cuda.max_memory_allocated()/1024**3, 3),
                  'quality_verified': False, 'manual_transcript_review_complete': False,
                  'losses_this_session': losses}
        write_json(directory / 'report.json', report)
        print(json.dumps({key: value for key, value in report.items() if key not in
                          ('losses_this_session', 'baseline', 'final')}, ensure_ascii=False, indent=2), flush=True)
    except (KeyboardInterrupt, Exception):
        optimizer.zero_grad(set_to_none=True)
        save('interrupted')
        raise
    finally:
        wrapper = optimizer = parameters = speaker = None
        release_gpu()
    return report
