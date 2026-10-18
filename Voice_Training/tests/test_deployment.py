"""Deployment safety tests with synthetic bytes and a fake HTTP implementation.

Structure/call tree: DownloadTests -> downloader.download/inside -> Response.
Data: temporary manifest/file bytes -> verified file or an explicit failure.
No external network, real model, or real dataset is used.
"""
import hashlib
import importlib.util
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('project_downloader', ROOT / 'deployment/download_models.py')
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}


class DownloadTests(unittest.TestCase):
    def item(self, content):
        return {'size': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
                'url': 'https://example.invalid/synthetic.bin'}

    def test_download_publishes_only_verified_bytes_and_then_reuses(self):
        content = b'synthetic model bytes'
        calls = []
        def opener(request, **kwargs):
            calls.append(request.full_url)
            return Response(content)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.bin'
            self.assertEqual(downloader.download(path, self.item(content), opener=opener), 'downloaded')
            self.assertEqual(downloader.download(path, self.item(content), opener=opener), 'existing')
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(len(calls), 1)

    def test_resume_sends_range_and_checks_response_range(self):
        content = b'0123456789'
        def opener(request, **kwargs):
            self.assertEqual(request.get_header('Range'), 'bytes=4-')
            return Response(content[4:], 206, {'Content-Range': 'bytes 4-9/10'})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.bin'
            path.with_name('model.bin.part').write_bytes(content[:4])
            downloader.download(path, self.item(content), opener=opener)
            self.assertEqual(path.read_bytes(), content)

    def test_bad_hash_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.bin'
            path.write_bytes(b'old')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                downloader.download(path, self.item(b'new'), opener=lambda *a, **k: Response(b'BAD'))
            self.assertEqual(path.read_bytes(), b'old')

    def test_bad_resume_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.bin'
            path.with_name('model.bin.part').write_bytes(b'01')
            with self.assertRaisesRegex(ValueError, 'resume range'):
                downloader.download(path, self.item(b'0123'),
                    opener=lambda *a, **k: Response(b'23', 206, {'Content-Range': 'bytes 0-1/4'}))
            self.assertFalse(path.exists())

    def test_offline_missing_file_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, 'Setup.ps1'):
                downloader.download(Path(temporary) / 'model.bin', self.item(b'X'), offline=True)

    def test_manifest_path_cannot_escape_or_use_a_windows_drive(self):
        with tempfile.TemporaryDirectory() as temporary:
            for value in ('../escape.bin', 'C:/outside.bin', '/absolute.bin'):
                with self.assertRaises(ValueError):
                    downloader.inside(Path(temporary), value)

    def test_missing_remote_file_is_not_retried(self):
        calls = []
        def opener(request, **kwargs):
            calls.append(request)
            raise urllib.error.HTTPError(request.full_url, 404, 'not found', {}, None)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(urllib.error.HTTPError):
                downloader.download(Path(temporary) / 'model.bin', self.item(b'X'), opener=opener,
                                    sleeper=lambda _: self.fail('404 must not be retried'))
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
