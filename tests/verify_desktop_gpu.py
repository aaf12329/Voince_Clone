"""Opt-in actual GPU/GUI-controller test. Never operates the user's desktop."""
import json
import os
import sys
import time
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['PYTHONUTF8'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from desktop_app import VoiceWindow
from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import QApplication
from voice_clone.project import SETTINGS


def main():
    previous_settings = SETTINGS.read_bytes() if SETTINGS.is_file() else None
    app = QApplication.instance() or QApplication([])
    window = VoiceWindow()
    window.load_example()
    window.generate()
    started = time.monotonic()
    state = {'generated': False, 'released': False, 'cancelled': False, 'closed': False}
    errors = []

    def finish_error(message):
        errors.append(message)
        window.close()
        app.quit()

    def poll():
        try:
            if time.monotonic() - started > 180:
                return finish_error('GPU / 控制器验证超时')
            if not state['generated'] and not window.busy:
                if not window.result or not Path(window.result).is_file():
                    return finish_error(window.status.text())
                assert window.play.isEnabled() and window.save.isEnabled()
                assert window.history.currentData() == window.result
                state['generated'] = True
                state['filename'] = Path(window.result).name
                state['message'] = window.status.text()
                window.release_memory()
                state['released'] = window.worker.state() == QProcess.ProcessState.NotRunning
                assert state['released']
                window.generate()
            elif state['generated'] and not state['cancelled'] and '正在生成' in window.stage:
                window.cancel()
                assert not window.busy and window.generate_button.isEnabled()
                assert window.worker.state() == QProcess.ProcessState.NotRunning
                state['cancelled'] = True
                window.generate()
                QTimer.singleShot(1000, close_during_generation)
        except Exception as exc:
            finish_error(str(exc) or type(exc).__name__)

    def close_during_generation():
        window.close()
        state['closed'] = window.worker.state() == QProcess.ProcessState.NotRunning
        app.quit()

    timer = QTimer()
    timer.setInterval(200)
    timer.timeout.connect(poll)
    timer.start()
    try:
        app.exec()
    finally:
        window.close()
        if previous_settings is None:
            SETTINGS.unlink(missing_ok=True)
        else:
            SETTINGS.write_bytes(previous_settings)
    state['errors'] = errors
    state['verified'] = all(state[key] for key in ('generated', 'released', 'cancelled', 'closed')) and not errors
    (ROOT / 'logs/desktop-controller-validation.json').write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(state, ensure_ascii=False), flush=True)
    return 0 if state['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
