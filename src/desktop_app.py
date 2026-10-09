"""Native Chinese desktop UI for the locally installed Qwen3-TTS model."""
import codecs
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QPainter, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSlider,
    QScrollArea, QVBoxLayout, QWidget,
)
from voice_clone import __version__
from voice_clone.audio import reference_info
from voice_clone.diagnostics import export_report
from voice_clone.project import ROOT, OUTPUTS, SETTINGS, LANGUAGES, ensure_directories, read_settings, write_json

INSTANCE_NAME = 'VoiceCloneDesktop_' + hashlib.sha256(str(ROOT).casefold().encode()).hexdigest()[:16]


def app_icon():
    pix = QPixmap(128, 128)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor('#5264e8'))
    painter.drawRoundedRect(4, 4, 120, 120, 30, 30)
    painter.setBrush(QColor('white'))
    for x, h in [(28, 28), (46, 56), (64, 80), (82, 48), (100, 24)]:
        painter.drawRoundedRect(x - 5, (128 - h) // 2, 10, h, 5, 5)
    painter.end()
    return QIcon(pix)


def button(label, callback, name=''):
    widget = QPushButton(label)
    if name:
        widget.setObjectName(name)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    widget.clicked.connect(callback)
    return widget


def card(title, subtitle):
    frame = QFrame()
    frame.setObjectName('card')
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(22, 18, 22, 18)
    layout.setSpacing(10)
    heading = QLabel(title)
    heading.setObjectName('cardTitle')
    layout.addWidget(heading)
    description = QLabel(subtitle)
    description.setObjectName('muted')
    description.setWordWrap(True)
    layout.addWidget(description)
    return frame, layout


class VoiceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_directories()
        self.setWindowTitle('声音克隆 · Voice Clone')
        self.setWindowIcon(app_icon())
        self.resize(1120, 810)
        self.setMinimumSize(800, 520)
        self.reference = ''
        self.result = ''
        self.busy = False
        self.cancelled = False
        self.buffer = ''
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.stage = ''
        self.worker = QProcess(self)
        worker_env = QProcessEnvironment.systemEnvironment()
        worker_env.insert('PYTHONUTF8', '1')
        worker_env.insert('PYTHONUNBUFFERED', '1')
        self.worker.setProcessEnvironment(worker_env)
        self.worker.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.worker.readyReadStandardOutput.connect(self.read_worker)
        self.worker.finished.connect(self.worker_finished)
        self.worker.errorOccurred.connect(self.worker_error)
        self.worker.started.connect(self.send_request)
        self.clock = QTimer(self)
        self.clock.setInterval(1000)
        self.clock.timeout.connect(self.update_elapsed)
        self.idle_timer = QTimer(self)
        self.idle_timer.setSingleShot(True)
        self.idle_timer.setInterval(5 * 60 * 1000)
        self.idle_timer.timeout.connect(self.release_memory)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.8)
        self.player.setAudioOutput(self.audio)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.positionChanged.connect(self.position_changed)
        self.player.playbackStateChanged.connect(self.playback_changed)
        self.player.errorOccurred.connect(lambda *_: self.show_status('无法试听此文件，请换用 WAV、MP3 或 FLAC 格式。', True))
        self.build_ui()
        self.restore_settings()
        self.refresh_history()

    def build_ui(self):
        help_menu = self.menuBar().addMenu('帮助')
        guide = QAction('使用与部署说明', self)
        guide.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(ROOT / 'README.md'))))
        help_menu.addAction(guide)
        diagnostics = QAction('导出诊断报告…', self)
        diagnostics.triggered.connect(self.export_diagnostics)
        help_menu.addAction(diagnostics)
        about = QAction('关于声音克隆', self)
        about.triggered.connect(lambda: QMessageBox.information(self, '声音克隆',
            f'Voice Clone {__version__}\nQwen3-TTS 1.7B Base\n本机推理 · NVIDIA / CUDA\n录音、输入和生成结果保存在本机。'))
        help_menu.addAction(about)
        central = QWidget()
        central.setObjectName('page')
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(central)
        shell = QWidget()
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(scroll, 1)
        self.setCentralWidget(shell)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(26, 18, 26, 14)
        outer.setSpacing(14)
        header = QHBoxLayout()
        brand = QLabel()
        brand.setPixmap(app_icon().pixmap(46, 46))
        header.addWidget(brand)
        titles = QVBoxLayout()
        title = QLabel('声音克隆')
        title.setObjectName('title')
        titles.addWidget(title)
        subtitle = QLabel('让熟悉的声音，说出新的内容')
        subtitle.setObjectName('muted')
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch()
        badge = QLabel('●  本机运行')
        badge.setObjectName('badge')
        header.addWidget(badge)
        outer.addLayout(header)

        columns = QHBoxLayout()
        columns.setSpacing(18)
        reference_card, left = card('01  参考声音', '选择一段清晰的单人录音，建议 5–15 秒。')
        choose_row = QHBoxLayout()
        self.choose_button = button('选择录音', self.choose_reference, 'secondary')
        choose_row.addWidget(self.choose_button)
        self.ref_play = button('试听', self.preview_reference)
        self.ref_play.setEnabled(False)
        choose_row.addWidget(self.ref_play)
        left.addLayout(choose_row)
        self.ref_label = QLabel('尚未选择录音')
        self.ref_label.setObjectName('fileName')
        self.ref_label.setWordWrap(True)
        self.ref_label.setMinimumHeight(42)
        left.addWidget(self.ref_label)
        left.addWidget(QLabel('录音原文'))
        self.transcript = QPlainTextEdit()
        self.transcript.setAccessibleName('录音原文')
        self.transcript.setPlaceholderText('逐字填写录音里说出的内容…')
        self.transcript.setMinimumHeight(110)
        self.transcript.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        left.addWidget(self.transcript, 1)
        self.example_button = button('载入官方示例', self.load_example, 'textButton')
        left.addWidget(self.example_button, alignment=Qt.AlignmentFlag.AlignLeft)
        columns.addWidget(reference_card, 4)

        text_card, right = card('02  新的朗读内容', '输入希望这个声音说出的文字。长文本会自动按句子分段。')
        text_label = QHBoxLayout()
        text_label.addWidget(QLabel('朗读文字'))
        text_label.addStretch()
        self.count = QLabel('0 / 1500')
        self.count.setObjectName('muted')
        text_label.addWidget(self.count)
        right.addLayout(text_label)
        self.text = QPlainTextEdit()
        self.text.setAccessibleName('朗读文字')
        self.text.setPlaceholderText('例如：你好，很高兴和你见面。')
        self.text.setMinimumHeight(150)
        self.text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.text.textChanged.connect(self.update_count)
        right.addWidget(self.text, 1)
        language_row = QHBoxLayout()
        language_row.addWidget(QLabel('朗读语言'))
        self.language = QComboBox()
        self.language.setAccessibleName('朗读语言')
        for label, code in LANGUAGES:
            self.language.addItem(label, code)
        language_row.addWidget(self.language, 1)
        right.addLayout(language_row)
        actions = QHBoxLayout()
        self.generate_button = button('生成语音', self.generate, 'primary')
        self.generate_button.setMinimumHeight(46)
        actions.addWidget(self.generate_button, 1)
        self.cancel_button = button('取消', self.cancel)
        self.cancel_button.setEnabled(False)
        actions.addWidget(self.cancel_button)
        right.addLayout(actions)
        columns.addWidget(text_card, 6)
        outer.addLayout(columns, 1)

        result_card, result_layout = card('03  试听与保存', '生成后自动保存，可试听、另存为或打开结果文件夹。')
        result_row = QHBoxLayout()
        self.play = button('试听结果', self.preview_result, 'secondary')
        self.play.setEnabled(False)
        result_row.addWidget(self.play)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setAccessibleName('试听进度')
        self.seek.setRange(0, 0)
        self.seek.sliderMoved.connect(self.player.setPosition)
        result_row.addWidget(self.seek, 1)
        self.duration_label = QLabel('00:00 / 00:00')
        result_row.addWidget(self.duration_label)
        self.save = button('另存为…', self.save_as)
        self.save.setEnabled(False)
        result_row.addWidget(self.save)
        result_row.addWidget(button('打开文件夹', self.open_outputs))
        self.release_button = button('释放显存', self.release_memory)
        self.release_button.setEnabled(False)
        result_row.addWidget(self.release_button)
        result_layout.addLayout(result_row)
        history_row = QHBoxLayout()
        history_row.addWidget(QLabel('最近生成'))
        self.history = QComboBox()
        self.history.setAccessibleName('最近生成的语音')
        self.history.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.history.activated.connect(self.select_history)
        history_row.addWidget(self.history, 1)
        result_layout.addLayout(history_row)
        outer.addWidget(result_card)
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(30, 8, 30, 16)
        bottom_layout.setSpacing(6)
        shell_layout.addWidget(bottom)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.hide()
        bottom_layout.addWidget(self.progress)
        self.status = QLabel('准备就绪 · 首次生成时自动加载模型')
        self.status.setWordWrap(True)
        self.status.setObjectName('status')
        bottom_layout.addWidget(self.status)
        footer = QLabel(f'Voice Clone {__version__}   ·   空闲 5 分钟或关闭窗口时自动释放模型显存')
        footer.setObjectName('muted')
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_layout.addWidget(footer)

    def update_count(self):
        count = len(self.text.toPlainText().strip())
        self.count.setText(f'{count} / 1500')
        self.count.setStyleSheet('color: #cc4358;' if count > 1500 else '')

    def choose_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择参考录音', str(ROOT / 'samples'),
                                             '音频文件 (*.wav *.mp3 *.flac *.ogg);;所有文件 (*)')
        if path:
            self.set_reference(path)

    def set_reference(self, path):
        try:
            info = reference_info(path)
        except ValueError as exc:
            self.show_status(str(exc), True)
            return False
        self.reference = path
        self.ref_label.setText(f"{Path(path).name}\n{info['duration']:.1f} 秒 · {info['samplerate']} Hz · {info['channels']} 声道")
        self.ref_label.setToolTip(path)
        self.ref_play.setEnabled(True)
        return True

    def load_example(self):
        if not self.set_reference(str(ROOT / 'samples' / 'official_reference.wav')):
            return
        self.transcript.setPlainText('Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.')
        self.text.setPlainText('你好，这是本地声音克隆测试。')
        self.language.setCurrentIndex(0)
        self.show_status('官方示例已载入，可以直接点击“生成语音”。')

    def show_status(self, message, error=False):
        self.status.setText(message)
        self.status.setStyleSheet('color: #c1394e;' if error else 'color: #475569;')

    def set_busy(self, busy):
        self.busy = busy
        self.generate_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.release_button.setEnabled(not busy and self.worker.state() != QProcess.ProcessState.NotRunning)
        for widget in (self.transcript, self.text, self.language, self.choose_button, self.example_button):
            widget.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.progress.setRange(0, 0 if busy else 1)
        if busy:
            self.idle_timer.stop()
            self.clock.start()
        else:
            self.clock.stop()
            if self.worker.state() != QProcess.ProcessState.NotRunning:
                self.idle_timer.start()

    def generate(self):
        if self.busy:
            return
        transcript = self.transcript.toPlainText().strip()
        text = self.text.toPlainText().strip()
        if not self.reference or not Path(self.reference).is_file():
            return self.show_status('请先选择一份存在的参考录音。', True)
        if not transcript:
            return self.show_status('请填写参考录音中实际说出的文字。', True)
        if not text or len(text) > 1500:
            return self.show_status('请填写朗读文字，每次最多 1500 个字符。', True)
        self.persist_settings()
        self.pending = {'reference': self.reference, 'transcript': transcript, 'text': text,
                        'language': self.language.currentData()}
        self.player.stop()
        self.cancelled = False
        self.started_at = time.monotonic()
        self.stage = '正在准备生成…'
        self.set_busy(True)
        self.update_elapsed()
        if self.worker.state() == QProcess.ProcessState.NotRunning:
            self.buffer = ''
            self.decoder.reset()
            self.worker.setWorkingDirectory(str(ROOT))
            self.worker.start(str(Path(sys.executable).with_name('python.exe')), ['-X', 'utf8', '-u', str(ROOT / 'src' / 'desktop_worker.py')])
        else:
            self.send_request()

    def send_request(self):
        if self.busy:
            self.worker.write((json.dumps(self.pending, ensure_ascii=False) + '\n').encode('utf-8'))

    def read_worker(self):
        chunk = self.decoder.decode(bytes(self.worker.readAllStandardOutput()))
        with (ROOT / 'logs' / 'desktop-worker.log').open('a', encoding='utf-8') as log:
            log.write(chunk)
        self.buffer += chunk
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if not line.startswith('VOICE_EVENT '):
                continue
            try:
                data = json.loads(line[len('VOICE_EVENT '):])
            except ValueError:
                continue
            if self.cancelled:
                continue
            if not isinstance(data, dict):
                continue
            if data.get('kind') == 'stage':
                self.stage = data['message']
                self.update_elapsed()
            elif data.get('kind') == 'done':
                path = Path(data.get('path', ''))
                if not path.is_file() or not path.resolve().is_relative_to(OUTPUTS.resolve()):
                    self.set_busy(False)
                    self.show_status('后台返回的音频无效，请重新生成。', True)
                    continue
                self.set_busy(False)
                self.set_result(data['path'])
                self.refresh_history(data['path'])
                self.show_status(f"生成完成 · 用时 {data['elapsed']:.1f} 秒 · 音频 {data['duration']:.2f} 秒 · 已自动保存")
            elif data.get('kind') == 'error':
                self.set_busy(False)
                self.show_status('生成失败：' + data['message'], True)

    def update_elapsed(self):
        self.show_status(f'{self.stage}  已用时 {int(time.monotonic() - self.started_at)} 秒')

    def worker_error(self, error):
        if self.cancelled:
            return
        self.set_busy(False)
        self.show_status('后台启动或运行失败，请查看 logs\\desktop-worker.log。', True)

    def worker_finished(self, *_):
        self.release_button.setEnabled(False)
        self.idle_timer.stop()
        if self.busy and not self.cancelled:
            self.set_busy(False)
            self.show_status('生成进程意外结束，请查看 logs\\desktop-worker.log，然后重试。', True)

    def stop_worker(self):
        self.idle_timer.stop()
        if self.worker.state() != QProcess.ProcessState.NotRunning:
            pid = int(self.worker.processId())
            if pid:
                try:
                    subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                                   creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True, timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    self.worker.kill()
            self.worker.waitForFinished(3000)
        self.release_button.setEnabled(False)

    def release_memory(self):
        if self.busy:
            return
        self.stop_worker()
        self.show_status('已释放模型显存。录音和结果仍保留，下次生成会重新加载模型。')

    def cancel(self):
        self.cancelled = True
        self.stop_worker()
        self.set_busy(False)
        self.show_status('已取消生成并释放模型显存。下次生成将重新加载模型。')

    def set_result(self, path):
        self.result = path
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(path))
        self.play.setEnabled(True)
        self.save.setEnabled(True)

    def preview_reference(self):
        if self.player.source().toLocalFile() == self.reference and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.setSource(QUrl.fromLocalFile(self.reference))
            self.player.play()

    def preview_result(self):
        if not self.result:
            return
        if self.player.source().toLocalFile() != self.result:
            self.player.setSource(QUrl.fromLocalFile(self.result))
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def playback_changed(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        source = self.player.source().toLocalFile()
        self.play.setText('暂停播放' if playing and source == self.result else '试听结果')
        self.ref_play.setText('暂停' if playing and source == self.reference else '试听')

    def duration_changed(self, duration):
        self.seek.setRange(0, duration)
        self.position_changed(self.player.position())

    def position_changed(self, position):
        if not self.seek.isSliderDown():
            self.seek.setValue(position)
        def fmt(ms):
            seconds = max(ms, 0) // 1000
            return f'{seconds // 60:02d}:{seconds % 60:02d}'
        self.duration_label.setText(f'{fmt(position)} / {fmt(self.player.duration())}')

    def refresh_history(self, selected=''):
        OUTPUTS.mkdir(exist_ok=True)
        files = sorted(OUTPUTS.glob('*.wav'), key=lambda p: p.stat().st_mtime, reverse=True)[:50]
        self.history.clear()
        self.history.addItem('选择已有音频以试听…', '')
        for path in files:
            self.history.addItem(path.name, str(path))
        if selected:
            self.history.setCurrentIndex(self.history.findData(selected))

    def select_history(self, index):
        path = self.history.itemData(index)
        if path and Path(path).is_file():
            self.set_result(path)
            self.show_status('已选择：' + Path(path).name)

    def save_as(self):
        if not self.result or not Path(self.result).is_file():
            return self.show_status('生成文件不存在，请重新生成。', True)
        path, _ = QFileDialog.getSaveFileName(self, '另存为语音', self.result, 'WAV 音频 (*.wav)')
        if path:
            try:
                if Path(path).resolve() != Path(self.result).resolve():
                    shutil.copy2(self.result, path)
                self.show_status('已保存到：' + path)
            except OSError as exc:
                self.show_status('保存失败：' + str(exc), True)

    def open_outputs(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(OUTPUTS)))

    def persist_settings(self):
        try:
            write_json(SETTINGS, {'reference': self.reference, 'transcript': self.transcript.toPlainText(),
                                 'text': self.text.toPlainText(), 'language': self.language.currentData()})
        except OSError:
            pass

    def restore_settings(self):
        try:
            data = read_settings()
            if Path(data.get('reference', '')).is_file():
                self.set_reference(data['reference'])
            self.transcript.setPlainText(data.get('transcript', ''))
            self.text.setPlainText(data.get('text', ''))
            index = self.language.findData(data.get('language', 'Chinese'))
            self.language.setCurrentIndex(max(index, 0))
        except (OSError, ValueError, KeyError):
            pass

    def export_diagnostics(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存诊断报告', str(ROOT / 'logs/diagnostics.json'),
                                             'JSON 报告 (*.json)')
        if path:
            try:
                export_report(path)
                self.show_status('诊断报告已导出，不包含录音和朗读文字。')
            except OSError as exc:
                self.show_status('报告保存失败：' + str(exc), True)

    def closeEvent(self, event):
        self.persist_settings()
        self.cancelled = True
        self.player.stop()
        self.stop_worker()
        event.accept()


STYLE = '''
QWidget { font-family: "Microsoft YaHei UI"; font-size: 13px; color: #24324b; }
QMainWindow, QMainWindow > QWidget, QWidget#page { background: #f3f5fa; }
QFrame#card { background: white; border: 1px solid #e2e7f0; border-radius: 16px; }
QLabel { background: transparent; }
QLabel#title { font-size: 27px; font-weight: 700; }
QLabel#cardTitle { font-size: 18px; font-weight: 600; }
QLabel#muted { color: #8490a4; font-size: 12px; }
QLabel#badge { color: #38795e; background: #e6f3ec; padding: 8px 14px; border-radius: 15px; }
QLabel#fileName { color: #586786; background: #f5f7fb; border-radius: 8px; padding: 10px; }
QPushButton { background: #f0f3f9; border: 1px solid #e0e5ef; border-radius: 8px; padding: 9px 15px; }
QPushButton:hover { background: #e8edf7; border-color: #b8c2da; }
QPushButton:disabled { color: #aab3c4; background: #f5f6f9; border-color: #edf0f5; }
QPushButton#primary { background: #5264e8; color: white; border: none; font-size: 15px; font-weight: 600; }
QPushButton#primary:hover { background: #4253d2; }
QPushButton#primary:disabled { background: #b1b9ed; }
QPushButton#secondary { background: #edf0ff; color: #4b5dca; border-color: #dae0ff; }
QPushButton#textButton { background: transparent; border: none; color: #5264d4; padding-left: 0; }
QPlainTextEdit, QComboBox { background: #fafbfe; border: 1px solid #dfe5ef; border-radius: 8px; padding: 10px; selection-background-color: #6575e8; }
QPlainTextEdit:focus, QComboBox:focus { border: 1px solid #6878e9; background: white; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox QAbstractItemView { background: white; selection-background-color: #e8ecff; selection-color: #24324b; }
QProgressBar { border: none; background: #e2e7f6; border-radius: 2px; }
QProgressBar::chunk { background: #6575e8; border-radius: 2px; }
QSlider::groove:horizontal { height: 5px; background: #e0e6f0; border-radius: 2px; }
QSlider::handle:horizontal { background: #5b6be3; width: 13px; margin: -4px 0; border-radius: 6px; }
QSlider::sub-page:horizontal { background: #8b97ed; }
'''


def main():
    ensure_directories()
    app = QApplication(sys.argv)
    app.setApplicationName('Voice Clone')
    app.setOrganizationName('LocalVoiceClone')
    app.setFont(QFont('Microsoft YaHei UI', 10))
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    app.setWindowIcon(app_icon())
    connection = QLocalSocket()
    connection.connectToServer(INSTANCE_NAME)
    if connection.waitForConnected(500):
        connection.write(b'show')
        connection.waitForBytesWritten(500)
        return 0
    server = QLocalServer()
    QLocalServer.removeServer(INSTANCE_NAME)
    if not server.listen(INSTANCE_NAME):
        QMessageBox.critical(None, '启动失败', '无法创建窗口连接，请关闭旧窗口后重试。')
        return 1
    window = VoiceWindow()
    def raise_window():
        socket = server.nextPendingConnection()
        if socket:
            socket.close()
            socket.deleteLater()
        window.showNormal()
        window.raise_()
        window.activateWindow()
    server.newConnection.connect(raise_window)
    window.show()
    return app.exec()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        ensure_directories()
        with (ROOT / 'logs' / 'desktop-error.log').open('a', encoding='utf-8') as error_log:
            traceback.print_exc(file=error_log)
        raise
