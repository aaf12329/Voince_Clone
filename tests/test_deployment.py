import hashlib
import importlib.util
import json
import tempfile
import threading
import unittest
import sys
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / 'deploy.py'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('voice_deploy', SCRIPT)
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentTests(unittest.TestCase):
    def test_public_config_is_portable_and_locked(self):
        public = json.loads((ROOT / 'deploy_config.json').read_text(encoding='utf-8'))
        self.assertFalse(any(public['reuse'].values()))
        lock = public['dependency_lock']
        self.assertEqual(deploy.sha256(ROOT / lock['path']), lock['sha256'])
        self.assertNotIn('C:/Users/', json.dumps(public))

    def test_private_config_is_optional_and_restricted(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'deploy_config.json'
            config.write_bytes((ROOT / 'deploy_config.json').read_bytes())
            self.assertFalse(any(deploy.load_config(config)['reuse'].values()))
            local = Path(directory) / 'deploy.local.json'
            local.write_text(json.dumps({'cuda_root':str(Path(directory)/'cuda')}),encoding='utf-8')
            self.assertEqual(deploy.load_config(config)['cuda_root'],str(Path(directory)/'cuda'))
            local.write_text(json.dumps({'packages':['unapproved']}),encoding='utf-8')
            with self.assertRaises(ValueError):
                deploy.load_config(config)

    def test_repair_preserves_old_environment_and_libraries(self):
        config = json.loads((ROOT / 'deploy_config.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            config['cuda_root'] = str(Path(directory)/'cuda')
            worker = deploy.Deployer(directory, config)
            worker.venv.mkdir()
            (worker.venv/'keep.bin').write_bytes(b'old environment')
            old_libraries = worker.libraries
            old_libraries.mkdir(parents=True)
            (old_libraries/'keep.dll').write_bytes(b'old CUDA')
            with patch.object(worker,'run',return_value=''):
                worker.prepare_repair()
            backups = list(Path(directory).glob('.venv.backup-*'))
            self.assertEqual(len(backups),1)
            self.assertEqual((backups[0]/'keep.bin').read_bytes(),b'old environment')
            self.assertEqual((old_libraries/'keep.dll').read_bytes(),b'old CUDA')
            restored = deploy.Deployer(directory,config)
            self.assertEqual(restored.libraries,worker.libraries)

    def test_repair_refuses_a_running_application(self):
        config = json.loads((ROOT / 'deploy_config.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            config['cuda_root'] = str(Path(directory)/'cuda')
            worker = deploy.Deployer(directory,config)
            worker.venv.mkdir()
            with patch.object(worker,'run',return_value='1234'):
                with self.assertRaises(RuntimeError):
                    worker.prepare_repair()
            self.assertTrue(worker.venv.is_dir())

    def test_diagnostics_exclude_text_audio_and_personal_paths(self):
        from voice_clone.diagnostics import create_report
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory)/'logs'
            logs.mkdir()
            (logs/'desktop-settings.json').write_text(json.dumps({'text':'secret transcript','reference':'C:/Users/Private/secret.wav'}),encoding='utf-8')
            (logs/'deploy-state.json').write_text(json.dumps({'phase':'complete','project_root':'C:/Users/Private/project','model_signatures':{}}),encoding='utf-8')
            serialized = json.dumps(create_report(directory))
            for secret in ('secret transcript','Private','secret.wav','project_root'):
                self.assertNotIn(secret,serialized)

    def test_setup_progress_parser_rejects_invalid_events(self):
        import setup_app
        valid = '12:00 DEPLOY_EVENT '+json.dumps({'step':5,'total':8,'message':'download','phase':'model'})
        self.assertEqual(setup_app.parse_progress(valid)['step'],5)
        for invalid in ('other log','DEPLOY_EVENT []','DEPLOY_EVENT {"step":9,"total":8,"message":"x"}'):
            self.assertIsNone(setup_app.parse_progress(invalid))

    def test_long_text_segmentation_preserves_content(self):
        from voice_clone.inference import split_text
        text = ('你好，这是一个句子。' * 60) + '结束。'
        chunks = split_text(text)
        self.assertTrue(all(0 < len(chunk) <= 200 for chunk in chunks))
        self.assertEqual(''.join(chunks),text)
        self.assertEqual(split_text('short text'),['short text'])

    def test_release_is_reproducible_and_contains_only_source(self):
        import package_release
        with tempfile.TemporaryDirectory() as directory:
            first, manifest = package_release.build(destination=Path(directory)/'first.zip')
            second, _ = package_release.build(destination=Path(directory)/'second.zip')
            self.assertEqual(first.read_bytes(),second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                for member in archive.namelist():
                    self.assertFalse(set(member.split('/')) & {'models','.venv','runtime','outputs','cache','logs','samples'})
                    self.assertNotIn('deploy.local.json',member)
                for name, digest in manifest['files'].items():
                    self.assertEqual(hashlib.sha256(archive.read('Voice_Clone/'+name)).hexdigest(),digest)

    def test_gpu_selection_uses_no_fixed_hardware_threshold(self):
        valid = {'index': 0, 'name': 'test GPU', 'vram_mib': 16384, 'driver': '617.14', 'compute_capability': '12.0'}
        for device in ({**valid, 'vram_mib': 4096}, {**valid, 'driver': '528.33'},
                       {**valid, 'compute_capability': '7.5'}):
            self.assertEqual(deploy.select_gpu([device])['index'], 0)
        with self.assertRaises(RuntimeError):
            deploy.select_gpu([])
        with self.assertRaises(RuntimeError):
            deploy.select_gpu([valid], wanted=2)
        self.assertEqual(deploy.select_gpu([valid, {**valid, 'index': 1, 'vram_mib': 24576}])['index'], 1)
        self.assertEqual(deploy.select_gpu([valid], wanted=0)['index'], 0)
        self.assertEqual(deploy.memory_mib('N/A'), 0)
        self.assertEqual(deploy.memory_mib(' 16384 '), 16384)

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in ('../outside', 'models/..', 'models/../../outside'):
                for allow_link in (False, True):
                    with self.assertRaises(ValueError):
                        deploy.safe_join(Path(directory), path, allow_final_link=allow_link)

    def test_cuda_probe_falls_back_to_another_gpu(self):
        config = json.loads(SCRIPT.with_name('deploy_config.json').read_text(encoding='utf-8'))
        first = {'index':0,'name':'unsupported GPU','vram_mib':4096,'driver':'400.0','compute_capability':'6.1'}
        second = {**first,'index':1,'name':'working GPU'}
        report = {'selected_gpu':first,'candidate_gpus':[first,second]}
        with tempfile.TemporaryDirectory() as directory:
            config['cuda_root'] = str(Path(directory)/'cuda')
            worker = deploy.Deployer(Path(directory),config)
            def response(_arguments, capture=False, env=None):
                if env['CUDA_VISIBLE_DEVICES'] == '0':
                    raise RuntimeError('actual CUDA operation failed')
                return 'VOICE_VALIDATION '+json.dumps({'gpu':'working GPU','dtype':'torch.float16',
                    'torch':'2.9.1+cu128','cuda':'12.8','library_dir':str(worker.libraries)})
            with patch.object(worker,'run',side_effect=response):
                worker.validate_runtime(report,load_model=False)
            self.assertEqual(report['selected_gpu']['index'],1)
            self.assertEqual(worker.env['CUDA_VISIBLE_DEVICES'],'1')

    def download_case(self, ignore_range=False, wrong_range=False, wrong_hash=False):
        content = bytes(range(256)) * 4096
        digest = hashlib.sha256(content).hexdigest()
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                calls.append(self.headers.get('Range'))
                first = int((self.headers.get('Range') or 'bytes=0-').split('=')[1].split('-')[0])
                if ignore_range:
                    first = 0
                self.send_response(200 if ignore_range or not first else 206)
                if first:
                    start = first + 1 if wrong_range else first
                    self.send_header('Content-Range', f'bytes {start}-{len(content)-1}/{len(content)}')
                self.send_header('Content-Length', str(len(content) - first))
                self.end_headers()
                try:
                    self.wfile.write(content[first:])
                except ConnectionError:
                    pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / 'artifact.bin'
                destination.with_name('artifact.bin.part').write_bytes(content[:12345])
                if wrong_range or wrong_hash:
                    with self.assertRaises(RuntimeError):
                        deploy.download(f'http://127.0.0.1:{server.server_port}/file', destination,
                                        '0' * 64 if wrong_hash else digest, len(content), retries=1)
                    self.assertFalse(destination.exists())
                else:
                    deploy.download(f'http://127.0.0.1:{server.server_port}/file', destination, digest, len(content), retries=1)
                    self.assertEqual(destination.read_bytes(), content)
                    count = len(calls)
                    deploy.download(f'http://127.0.0.1:{server.server_port}/file', destination, digest, len(content), retries=1)
                    self.assertEqual(len(calls), count)
                self.assertEqual(calls[0], 'bytes=12345-')
        finally:
            server.shutdown()
            server.server_close()

    def test_resume_http_range(self):
        self.download_case()

    def test_server_ignores_range(self):
        self.download_case(ignore_range=True)

    def test_wrong_range_is_not_published(self):
        self.download_case(wrong_range=True)

    def test_bad_sha_is_not_published(self):
        self.download_case(wrong_hash=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
