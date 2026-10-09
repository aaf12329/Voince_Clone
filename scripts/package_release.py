"""Build an allowlisted, reproducible source archive without any local runtime or user data."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from voice_clone import __version__

ROOT_FILES = ('.gitignore', '.gitattributes', 'README.md', 'pyproject.toml',
              'deploy.py', 'deploy_config.json', 'requirements.lock', '快速部署.bat', 'One_Click_Start.bat')


def release_files(root=ROOT):
    root = Path(root).resolve()
    files = [root / name for name in ROOT_FILES]
    for pattern in ('src/**/*.py', 'scripts/*.ps1', 'scripts/*.py', 'launchers/*.bat',
                    'assets/*.ico', 'assets/*.svg', 'assets/*.png', 'tests/*.py', '.github/workflows/*.yml'):
        files.extend(root.glob(pattern))
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(root) or '__pycache__' in path.parts:
            raise ValueError(f'Invalid source file: {path.name}')
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def build(root=ROOT, destination=None):
    root = Path(root).resolve()
    destination = Path(destination) if destination else root / 'releases' / f'Voice_Clone_{__version__}_源码.zip'
    files = release_files(root)
    config = json.loads((root / 'deploy_config.json').read_text(encoding='utf-8'))
    lock = config['dependency_lock']
    if hashlib.sha256((root / lock['path']).read_bytes()).hexdigest() != lock['sha256']:
        raise ValueError('Dependency lock does not match download config.')
    if any(config.get('reuse', {}).values()):
        raise ValueError('Public download config must not contain local reuse directories.')
    manifest = {'version': __version__, 'files': {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.zip.part')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        contents = [(path.relative_to(root).as_posix(), path.read_bytes()) for path in files]
        contents.append(('RELEASE_MANIFEST.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')))
        for name, data in contents:
            info = zipfile.ZipInfo('Voice_Clone/' + name, date_time=(2026, 10, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('Source archive verification failed.')
    temporary.replace(destination)
    return destination, manifest


if __name__ == '__main__':
    result, manifest = build()
    print(f'{result}\n{len(manifest["files"])} source files; SHA256 {hashlib.sha256(result.read_bytes()).hexdigest()}')
