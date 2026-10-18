"""Runtime tests using temporary paths, fake models and an offscreen Qt window.

Call tree: RuntimeTests -> shared I/O/model adapter; WorkerTests -> main -> fake
inference; DesktopTests -> VoiceWindow -> concrete idle widget state assertions.
No real recording, checkpoint or network request is used.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from common import io as project_io
from common import model, runtime


class RuntimeTests(unittest.TestCase):
    def test_model_path_comes_from_manifest_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, manifest = root / 'runtime.json', root / 'models.json'
            project_io.write_json(config, {'model_id': 'test/model'})
            with patch.multiple(runtime, ROOT=root, CONFIG=config, MODEL_MANIFEST=manifest):
                project_io.write_json(manifest, {'models': [{'id': 'test/model', 'directory': 'models/test'}]})
                self.assertEqual(runtime.get_model_path(), root / 'models/test')
                project_io.write_json(manifest, {'models': [{'id': 'test/model', 'directory': '../outside'}]})
                with self.assertRaises(ValueError):
                    runtime.get_model_path()

    def test_invalid_audio_is_never_published(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary, patch.object(project_io, 'OUTPUTS', Path(temporary)):
            with self.assertRaises(ValueError):
                project_io.save_wave(np.array([float('nan')]), 24000)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_saved_audio_is_valid_and_no_partial_file_remains(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temporary, patch.object(project_io, 'OUTPUTS', Path(temporary)):
            path = project_io.save_wave(0.1 * np.sin(np.arange(2400) * 0.1), 24000)
            info = sf.info(path)
            self.assertEqual((info.frames, info.samplerate, info.channels), (2400, 24000, 1))
            self.assertEqual(len(list(Path(temporary).iterdir())), 1)

    def test_synthesis_uses_fake_model_and_rejects_blank_text(self):
        class Fake:
            def generate_voice_clone(self, **options):
                self.options = options
                return [[0.0] * 100], 24000
        fake = Fake()
        wave, rate = model.synthesize(fake, 'synthetic.wav', 'reference text', 'hello', 'English')
        self.assertEqual((len(wave), rate), (100, 24000))
        self.assertFalse(fake.options['x_vector_only_mode'])
        with self.assertRaises(ValueError):
            model.synthesize(fake, 'synthetic.wav', 'reference text', ' ', 'English')


class WorkerTests(unittest.TestCase):
    def test_error_then_success_keeps_protocol_and_worker_alive(self):
        import desktop_worker
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / 'reference.wav'
            reference.write_bytes(b'synthetic fixture')
            output = root / 'result.wav'
            output.write_bytes(b'synthetic result')
            valid = {'reference': str(reference), 'text': 'test', 'transcript': 'reference', 'language': 'English'}
            invalid = {**valid, 'text': ''}
            stream = io.StringIO(json.dumps(invalid) + '\n' + json.dumps(valid) + '\n')
            captured = io.StringIO()
            class FakeEngine:
                def generate(self, request, emit):
                    if not request['text'].strip():
                        raise ValueError('Text is required')
                    return {'path': str(output), 'duration': 0.1}

                def describe_error(self, error):
                    return str(error)

            with patch.object(sys, 'stdin', stream), patch.object(sys, 'stderr', io.StringIO()), \
                 patch.object(desktop_worker, 'configure_runtime'), \
                 patch.object(desktop_worker, 'InferenceEngine', FakeEngine), redirect_stdout(captured):
                desktop_worker.main()
            events = [json.loads(line[len('VOICE_EVENT '):]) for line in captured.getvalue().splitlines()]
            self.assertEqual(events[0]['kind'], 'error')
            self.assertEqual(events[-1]['kind'], 'done')
            self.assertEqual(events[-1]['duration'], 0.1)


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def test_window_idle_state_and_relative_project_output_location(self):
        import desktop_app
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(desktop_app, 'SETTINGS', root / 'settings.json'), \
                 patch.object(desktop_app, 'OUTPUTS', root / 'outputs'):
                window = desktop_app.VoiceWindow()
                self.assertGreaterEqual(window.width(), 960)
                self.assertFalse(window.busy)
                self.assertFalse(window.ref_play.isEnabled())
                self.assertEqual(window.transcript.toPlainText(), '')
                self.assertEqual(window.worker.state().name, 'NotRunning')
                window.close()


if __name__ == '__main__':
    unittest.main()
