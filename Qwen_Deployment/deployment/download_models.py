"""Checksum-pinned model downloads; the caller supplies no machine-specific paths.

Structure: manifest/path validation -> resumable fetch -> main orchestration.
Call tree: main -> download -> valid / urllib.request.urlopen.
Data: deployment/models.json -> verified model files under the current project.
"""
import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'deployment/models.json'


def inside(root, relative):
    """Resolve a manifest path within its project; used before any file operation."""
    if PureWindowsPath(relative).is_absolute() or Path(relative).is_absolute():
        raise ValueError('Model paths must be relative.')
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError('Model path escapes project.')
    return candidate


def valid(path, item):
    """Check an existing file against the manifest's byte count and SHA256."""
    if not path.is_file() or path.stat().st_size != item['size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == item['sha256']


def download(path, item, offline=False, opener=urllib.request.urlopen, sleeper=time.sleep):
    """Fetch one verified file; retry transient failures and resume valid partial ranges."""
    if valid(path, item):
        return 'existing'
    if offline:
        raise FileNotFoundError(f'Missing/corrupt offline model file: {path.name}; run Setup.ps1 online.')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= item['size']:
            if valid(partial, item):
                partial.replace(path)
                return 'resumed'
            partial.unlink()
            offset = 0
        headers = {'User-Agent': 'Voice-Project-Deployment/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with opener(urllib.request.Request(item['url'], headers=headers), timeout=120) as response:
                resumed = response.status == 206
                if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Invalid HTTP resume range.')
                with partial.open('ab' if resumed else 'wb') as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        stream.write(block)
                        if stream.tell() > item['size']:
                            raise ValueError('Model exceeds manifest size.')
            if not valid(partial, item):
                raise ValueError('Model download failed size/SHA256 verification.')
            partial.replace(path)
            return 'downloaded'
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
        sleeper(2 ** attempt)
    raise RuntimeError('Model download did not finish.')


def main():
    """Read the repository manifest and ensure all selected base-model files exist."""
    parser = argparse.ArgumentParser(description='Download project-local, checksum-pinned base models.')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8-sig'))
    for model in manifest['models']:
        for item in model['files']:
            destination = inside(ROOT, model['directory'] + '/' + item['path'])
            result = download(destination, item, args.offline)
            print(f'{model["id"]}/{item["path"]}: {result}', flush=True)
    print('All required model files verified.')


if __name__ == '__main__':
    main()
