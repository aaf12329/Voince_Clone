"""Shared Qwen aggregation adapter: request validation -> reference audio ->
load_local_model (once) -> prompt (once/request) -> chunk generation -> write_result.
This explicit adapter coordinates shared audio/model leaves; no UI module is imported.
"""
import re
import time
import uuid
from .audio import load_reference, write_result
from .paths import OUTPUTS, LANGUAGES
from .model import load_local_model


def split_text(text, maximum=200):
    """Bound prompt length while preserving sentence punctuation and word boundaries."""
    chunks = []
    remaining = text.strip()
    while len(remaining) > maximum:
        candidate = remaining[:maximum]
        boundaries = [match.end() for match in re.finditer(r'[。！？!?；;\n]|[.!?](?=\s)', candidate)]
        cut = boundaries[-1] if boundaries else candidate.rfind(' ') + 1
        if cut < maximum // 3:
            cut = maximum
        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


class InferenceEngine:
    def __init__(self):
        """Initialize the instance state and resources owned by this component."""
        self.model = None
        self.torch = None

    def generate(self, request, notify):
        """Validate the request and produce audio through the inference adapter."""
        if not isinstance(request, dict):
            raise ValueError('生成请求无效，请重新启动窗口。')
        if any(not isinstance(request.get(key), str) for key in ('reference', 'text', 'transcript', 'language')):
            raise ValueError('请填写参考录音、录音原文和朗读文字。')
        text, transcript = request['text'].strip(), request['transcript'].strip()
        if not text or not transcript or len(text) > 1500:
            raise ValueError('请检查录音原文与朗读文字（最多 1500 字）。')
        if request['language'] not in {code for _, code in LANGUAGES}:
            raise ValueError('请选择支持的朗读语言。')
        started = time.perf_counter()
        notify('stage', message='正在检查参考录音…')
        reference_audio, _ = load_reference(request['reference'])
        if self.model is None:
            notify('stage', message='正在加载本地模型，首次使用请稍候…')
            import torch
            self.torch = torch
            self.model = load_local_model()
        import numpy as np
        chunks = split_text(text)
        parts = []
        rate = None
        with self.torch.inference_mode():
            notify('stage', message='正在准备参考声音…')
            prompt = self.model.create_voice_clone_prompt(ref_audio=reference_audio,
                                                        ref_text=transcript, x_vector_only_mode=False)
            for index, chunk in enumerate(chunks, 1):
                notify('stage', message=f'正在生成语音… {index} / {len(chunks)} 段')
                waves, current_rate = self.model.generate_voice_clone(
                    text=chunk, language=request['language'], voice_clone_prompt=prompt, max_new_tokens=2048,
                )
                if not waves or current_rate <= 0:
                    raise ValueError('模型未返回有效音频，请缩短文字或更换录音。')
                if rate is not None and current_rate != rate:
                    raise ValueError('模型输出采样率不一致，本次结果未保存。')
                rate = current_rate
                wave = np.asarray(waves[0])
                if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
                    raise ValueError('模型输出包含无效音频，本次结果未保存。')
                if index > 1:
                    parts.append(np.zeros(int(rate * 0.15), dtype='float32'))
                parts.append(wave)
        waveform = np.concatenate(parts)
        destination = OUTPUTS / (time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6] + '.wav')
        write_result(destination, waveform, rate)
        return {'path': str(destination), 'elapsed': round(time.perf_counter() - started, 1),
                'duration': round(len(waveform) / rate, 2), 'segments': len(chunks)}

    def describe_error(self, exc):
        """Turn model failures into actionable user messages and clear an OOM cache."""
        if self.torch is not None and isinstance(exc, self.torch.cuda.OutOfMemoryError):
            self.torch.cuda.empty_cache()
            return '显存不足。请关闭占用显卡的软件，并缩短朗读文字后重试。'
        return str(exc) or type(exc).__name__
