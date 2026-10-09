"""Local Qwen3-TTS Base voice cloning interface."""
import os
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import gradio as gr
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from voice_clone.audio import load_reference, write_result

MODEL_PATH = Path(os.environ.get('VOICE_CLONE_MODEL_DIR', str(ROOT / 'models' / 'Qwen3-TTS-12Hz-1.7B-Base')))
OUTPUTS = ROOT / 'outputs'
OUTPUTS.mkdir(exist_ok=True)


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到可用的 NVIDIA 显卡，请查看 logs/server.err.log。')
    print(f'Loading Qwen3-TTS on {torch.cuda.get_device_name(0)}', flush=True)
    return Qwen3TTSModel.from_pretrained(
        str(MODEL_PATH), device_map='cuda:0', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16,
        attn_implementation='sdpa', local_files_only=True,
    )


def make_demo(model):
    def generate(reference, reference_text, text, language):
        if not reference:
            raise gr.Error('请上传或录制一段参考音频。')
        if not reference_text or not reference_text.strip():
            raise gr.Error('请填写参考音频中实际说出的文字。')
        if not text or not text.strip():
            raise gr.Error('请填写需要朗读的新文字。')
        if len(text.strip()) > 1500:
            raise gr.Error('请将长文本分段，每次不超过 1500 个字符。')
        started = time.perf_counter()
        try:
            reference_audio, _ = load_reference(reference)
            with torch.inference_mode():
                wavs, sample_rate = model.generate_voice_clone(
                    text=text.strip(), language=language,
                    ref_audio=reference_audio, ref_text=reference_text.strip(),
                    x_vector_only_mode=False, max_new_tokens=2048,
                )
            filename = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6] + '.wav'
            destination = OUTPUTS / filename
            write_result(destination, wavs[0], sample_rate)
            elapsed = time.perf_counter() - started
            return str(destination), str(destination), f'完成，用时 {elapsed:.1f} 秒。音频已保存到 {OUTPUTS}。'
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise gr.Error('显存不足。请关闭占用显卡的程序，并缩短本次朗读的文字。')
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title='Voice Clone · Qwen3-TTS', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# 声音克隆\nQwen3-TTS 1.7B Base · 本机运行')
        gr.Markdown('上传清晰的单人说话录音，填写录音中的原文，再输入新的朗读文字。建议参考录音约 5–15 秒。')
        with gr.Row():
            with gr.Column():
                reference = gr.Audio(label='参考声音', type='filepath', format='wav', sources=['upload', 'microphone'])
                reference_text = gr.Textbox(label='参考录音原文', lines=3, placeholder='逐字填写参考音频中说出的内容。')
            with gr.Column():
                text = gr.Textbox(label='需要朗读的新文字', lines=6, placeholder='输入希望这个声音说出的新内容。')
                language = gr.Dropdown(label='朗读语言', choices=[('中文', 'Chinese'), ('自动识别', 'Auto'), ('英语', 'English'), ('日语', 'Japanese'), ('韩语', 'Korean'), ('法语', 'French'), ('德语', 'German'), ('西班牙语', 'Spanish'), ('俄语', 'Russian'), ('葡萄牙语', 'Portuguese'), ('意大利语', 'Italian')], value='Chinese')
                button = gr.Button('生成语音', variant='primary')
        audio = gr.Audio(label='生成结果', type='filepath')
        download = gr.File(label='下载 WAV 音频')
        status = gr.Textbox(label='状态', interactive=False)
        button.click(generate, [reference, reference_text, text, language], [audio, download, status], concurrency_limit=1)
        gr.Markdown(f'生成文件保存在 `{OUTPUTS}`。关闭网页不会停止服务；停止服务可释放显存。')
    return demo


if __name__ == '__main__':
    model = load_model()
    demo = make_demo(model)
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name='127.0.0.1', server_port=7860, share=False,
        allowed_paths=[str(OUTPUTS)], show_error=True,
    )
