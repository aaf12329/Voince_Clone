"""Shared Qwen adapter leaf used by desktop/web entries.

Call tree: load_local_model -> manifest path/Qwen.from_pretrained;
synthesize -> validation/Qwen.generate_voice_clone -> waveform and sample rate.
No application feature or UI module is imported here.
"""
from .runtime import get_model_path


def load_local_model():
    """Load the downloaded Base model with CUDA, using local files only."""
    import torch
    from qwen_tts import Qwen3TTSModel
    path = get_model_path()
    if not (path / 'model.safetensors').is_file() or not (path / 'config.json').is_file():
        raise FileNotFoundError('本地模型不完整，请先运行项目根目录的 Setup.ps1。')
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到可用 NVIDIA 显卡，请检查显卡驱动并运行 Check.ps1。')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return Qwen3TTSModel.from_pretrained(str(path), device_map='cuda:0', dtype=dtype,
                                        attn_implementation='sdpa', local_files_only=True)


def synthesize(model, reference, transcript, text, language):
    """Generate audio for the UI caller, translating GPU OOM into an actionable error."""
    import torch
    if not transcript.strip() or not text.strip() or len(text.strip()) > 1500:
        raise ValueError('参考录音原文和朗读文字不能为空，朗读文字最多 1500 字。')
    try:
        with torch.inference_mode():
            waves, rate = model.generate_voice_clone(text=text.strip(), language=language,
                ref_audio=reference, ref_text=transcript.strip(), x_vector_only_mode=False, max_new_tokens=2048)
        return waves[0], rate
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError('显存不足，请关闭占用显卡的软件或缩短朗读文字。') from exc
