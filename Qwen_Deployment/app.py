"""Web entry: main -> load_model/make_demo -> shared synthesis and audio I/O.

The entry owns UI orchestration. Input: recording/transcript/text/language;
output: saved waveform paths and status. Paths are read from common's address book.
"""
import time
import gradio as gr

from common.io import read_json, save_wave
from common.model import load_local_model, synthesize
from common.paths import CONFIG, OUTPUTS
from common.runtime import configure_runtime

configure_runtime()
WEB_CONFIG = read_json(CONFIG)


def load_model():
    """Compatibility entry used by the web startup and validation tools."""
    return load_local_model()


def make_demo(model):
    """Build the web controls and bind the shared inference callback."""
    def generate(reference, reference_text, text, language):
        """Validate the request and produce audio through the inference adapter."""
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
            wave, sample_rate = synthesize(model, reference, reference_text, text, language)
            destination = save_wave(wave, sample_rate)
            elapsed = time.perf_counter() - started
            return str(destination), str(destination), f'完成，用时 {elapsed:.1f} 秒。音频已保存到 outputs/。'
        except (ValueError, RuntimeError) as exc:
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
        gr.Markdown('生成文件保存在 `outputs/`。关闭网页不会停止服务；停止服务可释放显存。')
    return demo


if __name__ == '__main__':
    model = load_model()
    demo = make_demo(model)
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=WEB_CONFIG['web_host'], server_port=WEB_CONFIG['web_port'], share=False,
        allowed_paths=[str(OUTPUTS)], show_error=True,
    )
