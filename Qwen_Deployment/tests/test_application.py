"""Application boundary tests. Requires the deployed environment, never allocates a model."""
import json
import math
import os
import sys
import subprocess
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication
from common import audio
from common import io as project


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_stereo_reference_is_validated_and_converted_to_mono(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'录音.wav'
            waveform=np.tile(np.sin(np.arange(16000)/30).astype('float32')[:,None],(1,2))*0.2
            sf.write(path,waveform,16000)
            (mono, rate), info=audio.load_reference(path)
            self.assertEqual(mono.shape,(16000,))
            self.assertEqual(rate,16000)
            self.assertEqual(info['channels'],2)

    def test_silence_corruption_and_excessive_duration_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.wav'
            path.write_bytes(b'not audio')
            with self.assertRaises(ValueError): audio.load_reference(path)
            sf.write(path,np.zeros(8000),16000)
            with self.assertRaises(ValueError): audio.load_reference(path)
            sf.write(path,np.ones(121*8000)*0.1,8000)
            with self.assertRaises(ValueError): audio.reference_info(path)

    def test_invalid_output_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'result.wav'
            for waveform in (np.array([]),np.array([math.nan]),np.zeros(1600)):
                with self.assertRaises(ValueError): audio.write_result(destination,waveform,16000)
                self.assertFalse(destination.exists())
            audio.write_result(destination,np.sin(np.arange(1600)/20)*0.1,16000)
            self.assertEqual(sf.info(destination).frames,1600)
            self.assertFalse(destination.with_suffix('.wav.part').exists())

    def test_segmented_generation_reuses_prompt_and_joins_audio(self):
        from common import engine as inference
        with tempfile.TemporaryDirectory() as directory:
            reference=Path(directory)/'reference.wav'
            sf.write(reference,np.sin(np.arange(8000)/30)*0.2,8000)
            calls=[]
            class Model:
                def create_voice_clone_prompt(self,**kwargs):
                    calls.append(('prompt',kwargs))
                    return 'cached prompt'
                def generate_voice_clone(self,**kwargs):
                    calls.append(('generate',kwargs))
                    return [np.ones(1600)*0.1],16000
            engine=inference.InferenceEngine()
            engine.model=Model()
            engine.torch=SimpleNamespace(inference_mode=nullcontext)
            request={'reference':str(reference),'text':'你好。'*100,'transcript':'reference','language':'Chinese'}
            with patch.object(inference,'OUTPUTS',Path(directory)):
                result=engine.generate(request,lambda *args,**kwargs:None)
            self.assertEqual(sum(kind=='prompt' for kind,_ in calls),1)
            count=result['segments']
            self.assertGreater(count,1)
            self.assertEqual(sf.info(result['path']).frames,count*1600+(count-1)*2400)
            self.assertTrue(all(args['voice_clone_prompt']=='cached prompt' for kind,args in calls if kind=='generate'))

    def test_atomic_settings_and_invalid_settings_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'preferences.json'
            project.write_json(path,{'text':'中文'})
            with patch.object(project,'SETTINGS',path):
                self.assertEqual(project.read_settings()['text'],'中文')
                path.write_text('[1,2]',encoding='utf-8')
                self.assertEqual(project.read_settings(),{})
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])

    def test_worker_protocol_preserves_utf8_with_a_legacy_console(self):
        with tempfile.TemporaryDirectory() as directory:
            reference=Path(directory)/'中文录音.wav'
            sf.write(reference,np.zeros(8000),8000)
            request={'reference':str(reference),'text':'你好','transcript':'你好','language':'Chinese'}
            env=os.environ.copy()
            env.update({'PYTHONUTF8':'0','PYTHONIOENCODING':'gbk'})
            result=subprocess.run([sys.executable,str(ROOT/'desktop_worker.py')],
                input=json.dumps(request,ensure_ascii=False)+'\n',capture_output=True,text=True,
                encoding='utf-8',env=env,timeout=30)
            self.assertEqual(result.returncode,0)
            events=[json.loads(line[len('VOICE_EVENT '):]) for line in result.stdout.splitlines() if line.startswith('VOICE_EVENT ')]
            self.assertEqual(events[-1]['kind'],'error')
            self.assertIn('静音',events[-1]['message'])
            self.assertNotIn('\ufffd',result.stdout)


    def test_window_handles_corrupt_preferences_and_missing_reference(self):
        import desktop_app
        with patch.object(desktop_app,'read_settings',return_value={}):
            window=desktop_app.VoiceWindow()
        self.assertFalse(window.set_reference(str(ROOT/'samples/nonexistent.wav')))
        self.assertIn('不存在',window.status.text())
        self.assertLessEqual(window.minimumHeight(),520)
        with patch.object(window,'persist_settings'):
            window.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
