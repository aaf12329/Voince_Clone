"""Deployment check. main -> metadata/venv checks -> CUDA/audio probe -> JSON report.

Input: repository dependency manifest. Output: logs/environment-check.json.
No model training, model download, or graphical application is started.
"""
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / 'deployment/dependencies.json'
REPORT = ROOT / 'logs/environment-check.json'


def main():
    """Run environment probes and report checked package and GPU status."""
    config = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    expected_prefix = (ROOT / config['venv']['directory']).resolve()
    if Path(sys.prefix).resolve() != expected_prefix or sys.prefix == sys.base_prefix:
        raise RuntimeError('Use this project\'s .venv/Scripts/python.exe; run Setup.ps1 first.')
    differences = []
    for item in config['packages']:
        try:
            actual = importlib.metadata.version(item['name'])
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != item['version']:
            differences.append({'name': item['name'], 'expected': item['version'], 'actual': actual})
    if differences:
        raise RuntimeError('Dependency versions differ; rerun Setup.ps1: ' + json.dumps(differences))
    import torch
    import torchaudio
    from qwen_tts import Qwen3TTSModel
    if not torch.cuda.is_available():
        raise RuntimeError('A compatible NVIDIA GPU driver is required; PyTorch CUDA is unavailable.')
    value = torch.ones((2, 2), device='cuda')
    assert (value @ value).sum().item() == 8
    assert torchaudio.functional.resample(torch.zeros(1, 2400), 24000, 16000).shape[-1] == 1600
    library = (Path(torch.__file__).parent / 'lib').resolve()
    if not library.is_relative_to(expected_prefix):
        raise RuntimeError('Torch DLLs unexpectedly resolve outside this project environment.')
    if config['project_kind'] == 'voice_clone':
        from PySide6 import QtCore, QtMultimedia, QtWidgets
        assert QtWidgets.QApplication and QtMultimedia.QMediaPlayer and QtCore.__version__
    report = {'status': 'ready', 'python': sys.version.split()[0], 'venv': config['venv']['directory'],
              'packages_verified': len(config['packages']), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
              'torch_libraries': library.relative_to(ROOT).as_posix(), 'qwen_import': Qwen3TTSModel.__name__}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
