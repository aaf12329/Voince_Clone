"""common/tensor_ops.py: atomic_torch_save, torch_load, release_gpu, load_tts.

Call flow: CLI orchestration -> these functions -> shared common leaves/Qwen/PyTorch.
Data: project-relative configuration and checked manifests -> prepared data, caches,
adapter checkpoints or an exported character model. No sibling repository is imported.
"""
import gc
import uuid
from pathlib import Path



def atomic_torch_save(path, value):
    """Persist a tensor checkpoint through a flushed temporary file and rename."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')
    try:
        with temporary.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def torch_load(path):
    """Read tensor-only checkpoints on CPU without arbitrary pickle objects."""
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def release_gpu():
    """Collect released Python objects and free unused CUDA allocations."""
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def load_tts(path):
    """Load a local Qwen model with BF16 on a supported CUDA device."""
    import torch
    from qwen_tts import Qwen3TTSModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('这一版训练需要支持 BF16 的 CUDA GPU。')
    print(f'加载模型：{path.name}', flush=True)
    return Qwen3TTSModel.from_pretrained(str(path), device_map='cuda:0', dtype=torch.bfloat16,
                                        attn_implementation='sdpa', local_files_only=True)
