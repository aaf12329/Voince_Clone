"""Runtime leaves: configure_runtime -> local process environment/directories;
get_model_path -> shared JSON I/O -> manifest-relative model path.
No download or graphical action is performed by this module.
"""
import os
from pathlib import Path, PureWindowsPath

from .io import read_json
from .paths import CACHE, CONFIG, LOGS, MODEL_MANIFEST, OUTPUTS, ROOT, SOX, TEMP


def configure_runtime():
    """Configure project-local caches and offline inference for application entries."""
    values = {'HF_HOME': CACHE / 'huggingface', 'HF_HUB_CACHE': CACHE / 'huggingface/hub',
              'TORCH_HOME': CACHE / 'torch', 'NUMBA_CACHE_DIR': CACHE / 'numba',
              'GRADIO_TEMP_DIR': TEMP / 'gradio', 'TEMP': TEMP, 'TMP': TEMP}
    os.environ.update({name: str(value) for name, value in values.items()})
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      GRADIO_ANALYTICS_ENABLED='False', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    os.environ['PATH'] = str(SOX) + os.pathsep + os.environ.get('PATH', '')
    for directory in (LOGS, OUTPUTS, TEMP, *values.values()):
        directory.mkdir(parents=True, exist_ok=True)


def get_model_path():
    """Read the selected model directory from deployment's single model manifest."""
    selection = read_json(CONFIG)['model_id']
    manifest = read_json(MODEL_MANIFEST)
    model = next((item for item in manifest['models'] if item['id'] == selection), None)
    if model is None:
        raise ValueError('configs/runtime.json 的 model_id 未出现在模型下载清单中。')
    relative = model['directory']
    if Path(relative).is_absolute() or PureWindowsPath(relative).is_absolute():
        raise ValueError('模型目录必须使用项目相对路径。')
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('模型目录不能超出项目。')
    return path
