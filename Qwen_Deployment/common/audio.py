"""Shared audio leaf. reference_info -> load_reference validates recording bytes;
write_result validates generated mono audio before atomic WAV publication.
Callers: desktop UI, shared inference adapter and web output I/O.
"""
from pathlib import Path

MAX_REFERENCE_SECONDS = 120


def reference_info(path):
    """Inspect the recording metadata and reject unsupported input."""
    import soundfile as sf
    path = Path(path)
    if not path.is_file():
        raise ValueError('参考录音不存在，请重新选择。')
    try:
        info = sf.info(path)
    except (OSError, RuntimeError) as exc:
        raise ValueError('无法读取录音。请使用可播放的 WAV、MP3、FLAC 或 OGG 文件。') from exc
    if info.samplerate <= 0 or info.frames <= 0:
        raise ValueError('录音没有可用音频。')
    duration = info.frames / info.samplerate
    if duration < 0.25 or duration > MAX_REFERENCE_SECONDS:
        raise ValueError(f'参考录音需要在 0.25–{MAX_REFERENCE_SECONDS} 秒内，建议选择 5–15 秒的单人录音。')
    if info.channels > 8:
        raise ValueError('参考录音声道过多，请转换为单声道或双声道录音。')
    return {'duration': duration, 'samplerate': info.samplerate, 'channels': info.channels}


def load_reference(path):
    """Read no more than the validated frame count; reject NaN and silence."""
    import numpy as np
    import soundfile as sf
    info = reference_info(path)
    with sf.SoundFile(path) as source:
        if source.frames > MAX_REFERENCE_SECONDS * source.samplerate:
            raise ValueError('参考录音已被修改，请重新选择。')
        data = source.read(frames=source.frames, dtype='float32', always_2d=True)
        rate = source.samplerate
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError('参考录音损坏或包含无效音频。')
    mono = data.mean(axis=1)
    if float(np.max(np.abs(mono))) < 1e-5:
        raise ValueError('参考录音是静音，请选择有人声的录音。')
    return (mono, rate), info


def write_result(path, waveform, rate):
    """Publish WAV only after checking the model result and completing the write."""
    import numpy as np
    import soundfile as sf
    waveform = np.asarray(waveform)
    if rate <= 0 or waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
        raise ValueError('模型未返回有效音频，请重试或缩短朗读内容。')
    if float(np.max(np.abs(waveform))) < 1e-5:
        raise ValueError('模型生成了静音，请更换参考录音后重试。')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.wav.part')
    try:
        sf.write(temporary, waveform, rate, subtype='PCM_16', format='WAV')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
