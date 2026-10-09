"""Windows one-click deployment; Python stdlib only until the venv is ready."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import ctypes
import hashlib
import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
LOG = logging.getLogger('voice_deploy')


def progress(phase, message, step):
    LOG.info('DEPLOY_EVENT %s', json.dumps({'phase': phase, 'message': message, 'step': step,
                                         'total': 8}, ensure_ascii=False))


def load_config(path):
    path = Path(path)
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or config.get('schema_version') != 1:
        raise ValueError('下载配置版本不支持。')
    local_path = path.with_name('deploy.local.json')
    if local_path.is_file():
        local = json.loads(local_path.read_text(encoding='utf-8-sig'))
        if not isinstance(local, dict) or set(local) - {'reuse', 'cuda_root', 'shortcut_name'}:
            raise ValueError('deploy.local.json 仅允许设置 reuse、cuda_root 和 shortcut_name。')
        for key, value in local.items():
            config[key] = value
    for key in ('python_version', 'cuda_root', 'bootstrap', 'hardware', 'download', 'torch_wheels',
                'packages', 'package_index', 'model', 'sox', 'sample', 'shortcut_name'):
        if key not in config:
            raise ValueError(f'下载配置缺少字段：{key}')
    for item in [config['bootstrap'], config['sox'], config['sample'],
                 *config['torch_wheels'], *config['model']['files']]:
        if not re.fullmatch(r'[0-9a-f]{64}', item.get('sha256', '')) or not isinstance(item.get('size'), int) or item['size'] <= 0:
            raise ValueError('下载配置的 SHA256 或文件大小无效。')
    if not 1 <= config['download']['parallel_model_files'] <= 8:
        raise ValueError('模型并发下载数需要在 1–8 范围内。')
    return config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_join(root, relative, allow_final_link=False):
    base = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or relative.drive or '..' in relative.parts:
        raise ValueError(f'文件路径超出允许目录：{relative}')
    result = base / relative
    result = result.parent.resolve() / result.name if allow_final_link else result.resolve()
    if not result.is_relative_to(base) or result == base:
        raise ValueError(f'文件路径超出允许目录：{relative}')
    return result


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def matches(path, digest, size=None):
    path = Path(path)
    return path.is_file() and (size is None or path.stat().st_size == size) and sha256(path) == digest


def download(url, destination, digest, size=None, retries=5, timeout=120):
    """Resume .part files, validate HTTP ranges, and only publish verified bytes."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if matches(destination, digest, size):
        LOG.info('复用已校验文件：%s', destination.name)
        return destination
    partial = destination.with_name(destination.name + '.part')
    for attempt in range(1, retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if size is not None and offset >= size:
                if matches(partial, digest, size):
                    partial.replace(destination)
                    return destination
                partial.write_bytes(b'')
                offset = 0
            headers = {'User-Agent': 'VoiceClone-Deploy/1.0', 'Accept-Encoding': 'identity'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                if status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    parsed = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                    if not parsed or int(parsed[1]) != offset:
                        raise RuntimeError('服务器返回了错误的续传范围。')
                    if size is not None and int(parsed[3]) != size:
                        raise RuntimeError('服务器文件大小与配置不一致。')
                    expected_end = int(parsed[2]) + 1
                elif status == 200:
                    offset = 0  # A server may ignore Range; restart instead of appending.
                    expected_end = None
                else:
                    raise RuntimeError(f'下载服务器状态异常：{status}')
                LOG.info('下载 %s（起点 %.1f MB）', destination.name, offset / 1e6)
                written = offset
                reported = time.monotonic()
                with partial.open('ab' if offset else 'wb') as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        written += len(chunk)
                        if size is not None and written > size:
                            raise RuntimeError('下载文件超过配置大小。')
                        if time.monotonic() - reported >= 5:
                            LOG.info('%s：%.1f MB%s', destination.name, written / 1e6,
                                     f' / {size / 1e6:.1f} MB' if size else '')
                            reported = time.monotonic()
                if expected_end is not None and written != expected_end:
                    raise RuntimeError('下载响应未完整传输。')
            if size is not None and partial.stat().st_size != size:
                # Some CDNs return a bounded range. Continue from the new offset.
                raise RuntimeError('文件尚未下载完整，将继续续传。')
            if sha256(partial) != digest:
                partial.write_bytes(b'')
                raise RuntimeError('SHA256 校验失败，将重新下载。')
            partial.replace(destination)
            LOG.info('校验通过：%s', destination.name)
            return destination
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f'下载失败：{destination.name}；{exc}') from exc
            LOG.warning('下载重试 %d/%d：%s', attempt, retries, exc)
            time.sleep(min(attempt * 2, 10))
    raise RuntimeError('下载未完成。')


def version_tuple(value):
    return tuple(int(x) for x in re.findall(r'\d+', str(value)))


def select_gpu(gpus, wanted=None):
    suitable = [gpu for gpu in gpus if wanted is None or gpu['index'] == wanted]
    if not suitable:
        raise RuntimeError('没有检测到 NVIDIA 显卡，或指定的显卡编号不存在。')
    return max(suitable, key=lambda gpu: (gpu.get('free_mib', 0), gpu['vram_mib']))


def memory_mib(value):
    # Unknown telemetry must not prevent the actual CUDA/model probe.
    try:
        return int(value.strip())
    except ValueError:
        return 0


def hardware_report(config, wanted=None):
    if os.name != 'nt' or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise RuntimeError('本部署脚本支持 Windows x64。')
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ('total_phys', 'avail_phys', 'total_page',
                                                   'avail_page', 'total_virtual', 'avail_virtual', 'extended')]
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise RuntimeError('无法读取系统内存配置。')
    smi = shutil.which('nvidia-smi')
    if not smi:
        for candidate in (Path(os.environ.get('WINDIR', r'C:\Windows')) / 'System32/nvidia-smi.exe',
                          Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'NVIDIA Corporation/NVSMI/nvidia-smi.exe'):
            if candidate.is_file():
                smi = str(candidate)
                break
    if not smi:
        raise RuntimeError('未找到 nvidia-smi。需要可用的 NVIDIA 显卡及驱动；请安装驱动后重试。')
    fields = 'index,name,memory.total,memory.free,driver_version,compute_cap'
    options = {'capture_output': True, 'text': True, 'encoding': 'utf-8', 'errors': 'replace',
               'timeout': 30, 'creationflags': subprocess.CREATE_NO_WINDOW}
    probe = subprocess.run([smi, '--query-gpu=' + fields, '--format=csv,noheader,nounits'], **options)
    has_capability = probe.returncode == 0
    if not has_capability:
        probe = subprocess.run([smi, '--query-gpu=' + fields.rsplit(',', 1)[0], '--format=csv,noheader,nounits'], **options)
    if probe.returncode:
        raise RuntimeError('NVIDIA 驱动无法正常查询显卡：' + probe.stderr.strip())
    gpus = []
    for row in csv.reader(io.StringIO(probe.stdout), skipinitialspace=True):
        if len(row) < 5:
            continue
        gpus.append({'index': int(row[0]), 'name': row[1].strip(), 'vram_mib': memory_mib(row[2]),
                     'free_mib': memory_mib(row[3]), 'driver': row[4].strip(),
                     'compute_capability': row[5].strip() if has_capability and len(row) > 5 and row[5].strip() != 'N/A' else None})
    if not gpus:
        raise RuntimeError('未检测到 NVIDIA 显卡，当前 CUDA 方案无法部署。')
    chosen = select_gpu(gpus, wanted)
    return {'os': platform.platform(), 'cpu': cpu, 'logical_cores': os.cpu_count(),
            'ram_gib': round(memory.total_phys / 1024 ** 3, 1), 'gpus': gpus, 'selected_gpu': chosen,
            'candidate_gpus': [gpu for gpu in gpus if wanted is None or gpu['index'] == wanted],
            'eligible': None, 'note': '已检测到显卡；是否能运行模型由实际 CUDA 运算及短语音生成验证决定。'}


class Deployer:
    def __init__(self, root, config, verify=False):
        self.root = Path(root).resolve()
        self.config = config
        self.verify = verify
        self.venv = self.root / '.venv'
        self.python = self.venv / 'Scripts/python.exe'
        self.uv = self.root / 'tools/bootstrap/uv.exe'
        self.cuda = Path(config['cuda_root'])
        if not self.cuda.is_absolute() or self.cuda == Path(self.cuda.anchor):
            raise ValueError('cuda_root 必须为明确的绝对目录，例如 C:\\CUDA\\Voice_Clone。')
        self.cuda = self.cuda.resolve()
        self.instance = hashlib.sha256(str(self.root).casefold().encode('utf-8')).hexdigest()[:12]
        self.libraries = self.cuda / 'runtimes' / (self.instance + '-torch-2.9.1-cu128') / 'lib'
        self.state_file = self.root / 'logs/deploy-state.json'
        try:
            self.state = json.loads(self.state_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            self.state = {}
        if not isinstance(self.state, dict):
            self.state = {}
        previous_library = self.state.get('cuda_library_dir')
        if previous_library:
            previous = Path(previous_library).resolve()
            if previous.is_relative_to(self.cuda / 'runtimes') and previous.parent.name.startswith(self.instance + '-'):
                self.libraries = previous
        self.fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.env = os.environ.copy()
        for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX'):
            self.env.pop(key, None)
        values = {'UV_CACHE_DIR': self.cuda / 'cache/uv', 'UV_PYTHON_INSTALL_DIR': self.root / 'runtime',
                  'UV_PYTHON_BIN_DIR': self.root / 'tools/bin', 'HF_HOME': self.root / 'cache/huggingface',
                  'TORCH_HOME': self.root / 'cache/torch', 'XDG_CACHE_HOME': self.root / 'cache',
                  'NUMBA_CACHE_DIR': self.root / 'cache/numba', 'GRADIO_TEMP_DIR': self.root / 'temp/gradio',
                  'TEMP': self.root / 'temp', 'TMP': self.root / 'temp'}
        self.env.update({key: str(value) for key, value in values.items()})
        self.env.update({'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'HF_HUB_DISABLE_TELEMETRY': '1',
                         'GRADIO_ANALYTICS_ENABLED': 'False', 'UV_HTTP_TIMEOUT': '120', 'UV_LINK_MODE': 'copy'})
        self.env['PATH'] = str(self.root / 'tools/sox/sox-14.4.2') + ';' + self.env.get('PATH', '')

    def mark(self, phase, **extra):
        self.state.update({'phase': phase, 'project_root': str(self.root), 'cuda_library_dir': str(self.libraries), **extra})
        write_json(self.state_file, self.state)

    def run(self, arguments, capture=False, env=None):
        options = {'cwd': self.root, 'env': env or self.env, 'creationflags': subprocess.CREATE_NO_WINDOW,
                   'encoding': 'utf-8', 'errors': 'replace', 'text': True}
        if capture:
            result = subprocess.run([str(x) for x in arguments], capture_output=True, timeout=300, **options)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f'命令退出码 {result.returncode}')
            return result.stdout.strip()
        with subprocess.Popen([str(x) for x in arguments], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **options) as process:
            for line in process.stdout:
                LOG.info(line.rstrip())
            if process.wait():
                raise RuntimeError('安装命令失败，详细输出已记录在 logs/deploy.log。')

    def setup_python(self):
        LOG.info('准备项目根目录的 Python 环境：%s', self.venv)
        if not self.uv.is_file():
            bootstrap = self.config['bootstrap']
            archive = download(bootstrap['url'], self.root / 'tools/bootstrap/uv-bootstrap.zip',
                               bootstrap['sha256'], bootstrap['size'])
            with zipfile.ZipFile(archive) as source:
                names = [name for name in source.namelist() if name.endswith('/uv.exe') or name == 'uv.exe']
                if len(names) != 1:
                    raise RuntimeError('uv 安装包未包含唯一的 uv.exe。')
                self.uv.parent.mkdir(parents=True, exist_ok=True)
                self.uv.write_bytes(source.read(names[0]))
        self.run([self.uv, 'python', 'install', self.config['python_version'], '--no-bin', '--no-registry'])
        if not self.python.is_file():
            if self.venv.exists() and any(self.venv.iterdir()):
                LOG.warning('项目环境不完整，将备份旧目录后重新建立。')
                self.prepare_repair()
            self.run([self.uv, 'venv', self.venv, '--python', self.config['python_version'], '--managed-python'])
        actual = self.run([self.python, '-c', 'import platform; print(platform.python_version())'], capture=True)
        if actual != self.config['python_version']:
            raise RuntimeError(f'项目环境 Python 为 {actual}，需要 {self.config["python_version"]}。请备份原 .venv 后重试。')
        self.mark('python_ready')

    def install_packages(self):
        LOG.info('安装模型、图形界面和 CUDA 版 PyTorch 依赖。')
        wheels = []
        for item in self.config['torch_wheels']:
            target = safe_join(self.cuda / 'downloads', item['filename'])
            for candidate_dir in self.config.get('reuse', {}).get('wheel_directories', []):
                candidate = Path(candidate_dir) / item['filename']
                if matches(candidate, item['sha256'], item['size']):
                    target = candidate
                    break
            wheels.append(download(item['url'], target, item['sha256'], item['size']))
        self.run([self.uv, 'pip', 'install', '--python', self.python, '--index-url', self.config['package_index'], *wheels])
        lock = self.config.get('dependency_lock')
        if lock:
            lock_path = safe_join(self.root, lock['path'])
            if not matches(lock_path, lock['sha256']):
                raise RuntimeError('requirements.lock 校验失败，请恢复与源码版本配套的依赖清单。')
            self.run([self.uv, 'pip', 'install', '--python', self.python, '--index-url', self.config['package_index'],
                      '-r', lock_path])
        else:
            self.run([self.uv, 'pip', 'install', '--python', self.python, '--index-url', self.config['package_index'],
                      *self.config['packages']])
        self.run([self.uv, 'pip', 'check', '--python', self.python])
        self.mark('packages_ready')

    def junction(self, link, target):
        link, target = Path(link), Path(target).resolve()
        if link.exists():
            if link.resolve() == target:
                return
            raise RuntimeError(f'链接位置已被其他文件占用：{link}')
        link.parent.mkdir(parents=True, exist_ok=True)
        env = self.env.copy()
        env.update({'VOICE_DEPLOY_LINK': str(link), 'VOICE_DEPLOY_TARGET': str(target)})
        self.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                  "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path $env:VOICE_DEPLOY_LINK -Value $env:VOICE_DEPLOY_TARGET | Out-Null"], env=env)
        if link.resolve() != target:
            raise RuntimeError('目录链接创建后验证失败。')

    def relocate_cuda(self):
        LOG.info('配置位于 C 盘 CUDA 目录的实际运行库。')
        logical = self.venv / 'Lib/site-packages/torch/lib'
        if logical.exists() and logical.resolve() == self.libraries.resolve():
            return
        if not logical.exists() and self.libraries.is_dir():
            self.junction(logical, self.libraries)
            self.mark('cuda_ready')
            return
        if logical.is_symlink() or logical.is_junction():
            raise RuntimeError('torch/lib 已指向其他位置，请核对现有环境。')
        if not logical.is_dir():
            raise RuntimeError('没有找到 PyTorch 的 torch/lib 运行库。')
        source = logical.resolve()
        target = self.libraries.resolve()
        if not source.is_relative_to(self.venv.resolve()) or not target.is_relative_to(self.cuda.resolve()):
            raise RuntimeError('CUDA 运行库移动范围验证失败。')
        if target.exists():
            raise RuntimeError(f'CUDA 目标目录已有内容，请保留现有目录并在 JSON 中更换 cuda_root：{target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        try:
            self.junction(logical, target)
        except Exception:
            if not logical.exists():
                shutil.move(str(target), str(source))
            raise
        self.mark('cuda_ready')

    def model_path(self):
        return safe_join(self.root, self.config['model']['directory'], allow_final_link=True)

    def verify_model(self, directory, cached=False):
        entries = self.config['model']['files']
        signatures = {}
        saved = self.state.get('model_signatures', {}) if cached else {}
        for item in entries:
            path = safe_join(directory, item['path'])
            if not path.is_file() or path.stat().st_size != item['size']:
                return None
            signature = [path.stat().st_size, path.stat().st_mtime_ns]
            if saved.get(item['path']) != signature and sha256(path) != item['sha256']:
                return None
            signatures[item['path']] = signature
        return signatures

    def setup_model(self):
        target = self.model_path()
        LOG.info('准备模型及语音编码器：%s', target)
        signatures = self.verify_model(target, cached=not self.verify and self.state.get('config_hash') == self.fingerprint)
        if signatures is None and not target.exists():
            for candidate in self.config.get('reuse', {}).get('model_directories', []):
                if Path(candidate).resolve() == target.resolve():
                    continue
                signatures = self.verify_model(Path(candidate))
                if signatures is not None:
                    LOG.info('复用已下载并校验的本机模型：%s', candidate)
                    self.junction(target, Path(candidate))
                    break
        if signatures is None:
            if target.is_junction() or target.is_symlink():
                raise RuntimeError('已链接模型校验失败。请检查原模型；不会修改链接指向的原安装。')
            target.mkdir(parents=True, exist_ok=True)
            def fetch(item):
                url = self.config['model']['api'] + '?' + urllib.parse.urlencode(
                    {'Revision': item['revision'], 'FilePath': item['path']})
                return download(url, safe_join(target, item['path']), item['sha256'], item['size'],
                                self.config['download']['retries'], self.config['download']['timeout_seconds'])
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['download']['parallel_model_files']) as pool:
                list(pool.map(fetch, self.config['model']['files']))
            signatures = self.verify_model(target)
            if signatures is None:
                raise RuntimeError('模型完整性检查未通过。')
        self.mark('model_ready', model_signatures=signatures)

    def setup_audio(self):
        item = self.config['sox']
        executable = safe_join(self.root, item['executable'])
        tool_ready = False
        if executable.is_file() and not self.verify:
            try:
                tool_ready = '14.4.2' in self.run([executable, '--version'], capture=True)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                pass
        if not tool_ready:
            destination = self.root / 'tools/sox-14.4.2-win32.zip'
            for candidate in self.config.get('reuse', {}).get('sox_archives', []):
                if matches(candidate, item['sha256'], item['size']):
                    destination = Path(candidate)
                    break
            archive = download(item['url'], destination, item['sha256'], item['size'])
            directory = safe_join(self.root, item['directory'])
            directory.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as package:
                for entry in package.infolist():
                    safe_join(directory, entry.filename.rstrip('/'))
                package.extractall(directory)
        self.run([executable, '--version'], capture=True)
        sample = self.config['sample']
        destination = safe_join(self.root, sample['path'])
        download(sample['url'], destination, sample['sha256'], sample['size'])
        self.mark('audio_ready')

    def validate_runtime(self, report, load_model=True):
        LOG.info('实际验证 CUDA 运算%s。', '、模型加载与短语音生成' if load_model else '')
        script = r'''
import json, os, importlib.metadata
from pathlib import Path
for pin in json.loads(os.environ['VOICE_REQUIRED_PACKAGES']):
    name, required = pin.split('==', 1)
    if importlib.metadata.version(name) != required:
        raise RuntimeError(f'依赖 {name} 版本不匹配，请运行快速部署.bat --repair。')
import torch
from PySide6.QtMultimedia import QMediaPlayer
if not torch.cuda.is_available():
    raise RuntimeError('PyTorch 无法使用 CUDA；请核对 NVIDIA 驱动。')
device = torch.cuda.get_device_properties(0)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
value = torch.ones((32,32),device='cuda',dtype=dtype)
result = value @ value
torch.cuda.synchronize()
if result[0,0].item() != 32:
    raise RuntimeError('CUDA 运算校验失败。')
if os.environ['VOICE_LOAD_MODEL'] == '1':
    from qwen_tts import Qwen3TTSModel
    import soundfile as sf
    import numpy as np
    model = Qwen3TTSModel.from_pretrained(os.environ['VOICE_MODEL_DIR'],device_map='cuda:0',
             dtype=dtype,attn_implementation='sdpa',local_files_only=True)
    with torch.inference_mode():
        waves, rate = model.generate_voice_clone(text='你好，部署测试完成。', language='Chinese',
            ref_audio=os.environ['VOICE_SAMPLE_PATH'],
            ref_text='Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.',
            x_vector_only_mode=False,max_new_tokens=128)
    if not np.isfinite(waves[0]).all() or np.max(np.abs(waves[0])) < 0.001:
        raise RuntimeError('模型输出无效或静音。')
    sf.write(os.environ['VOICE_VALIDATION_WAV'],waves[0],rate,subtype='PCM_16')
print('VOICE_VALIDATION '+json.dumps({'gpu':device.name,'capability':[device.major,device.minor],
      'dtype':str(dtype),'inference_verified':os.environ['VOICE_LOAD_MODEL']=='1',
      'cuda':torch.version.cuda,'torch':torch.__version__,'library_dir':str((Path(torch.__file__).parent/'lib').resolve())}))
'''
        selected = report['selected_gpu']['index']
        candidates = sorted(report['candidate_gpus'], key=lambda gpu: gpu['index'] != selected)
        failures = []
        for gpu in candidates:
            env = self.env.copy()
            env.update({'CUDA_VISIBLE_DEVICES': str(gpu['index']), 'HF_HUB_OFFLINE': '1',
                        'TRANSFORMERS_OFFLINE': '1', 'VOICE_SAMPLE_PATH': str(safe_join(self.root, self.config['sample']['path'])),
                        'VOICE_VALIDATION_WAV': str(self.root / 'logs/deploy-validation.wav'),
                        'VOICE_MODEL_DIR': str(self.model_path()), 'VOICE_LOAD_MODEL': '1' if load_model else '0',
                        'VOICE_REQUIRED_PACKAGES': json.dumps(self.config['packages'])})
            LOG.info('尝试显卡 %s：%s（按实际能力判断，不设显存门槛）。', gpu['index'], gpu['name'])
            try:
                output = self.run([self.python, '-c', script], capture=True, env=env)
                event = next((line for line in output.splitlines() if line.startswith('VOICE_VALIDATION ')), None)
                if not event:
                    raise RuntimeError('运行环境未返回验证结果。')
                validation = json.loads(event[len('VOICE_VALIDATION '):])
                if Path(validation['library_dir']).resolve() != self.libraries.resolve():
                    raise RuntimeError('实际 CUDA 库位置与配置不一致。')
                report['selected_gpu'] = gpu
                report['eligible'] = True if load_model else None
                report['note'] = '模型已实际生成测试语音。' if load_model else 'CUDA 运算通过，模型推理待完整部署验证。'
                if load_model:
                    self.state['validated_gpu'] = {key: gpu[key] for key in ('index', 'name', 'driver')}
                self.env['CUDA_VISIBLE_DEVICES'] = str(gpu['index'])
                write_json(self.root / 'logs/hardware-report.json', report)
                write_json(self.root / 'logs/runtime-validation.json' if load_model else self.root / 'logs/cuda-probe.json', validation)
                LOG.info('验证通过：%s / %s / PyTorch %s / CUDA %s', validation['gpu'], validation['dtype'], validation['torch'], validation['cuda'])
                return
            except Exception as exc:
                failures.append(gpu['name'] + '：' + str(exc))
                LOG.warning('该显卡实测未通过：%s', exc)
        report['eligible'] = False
        report['note'] = '当前安装配置在所有候选显卡上实测失败。'
        write_json(self.root / 'logs/hardware-report.json', report)
        raise RuntimeError('当前显卡未通过实际运行验证，请查看驱动、依赖或显存不足的具体错误。\n' + '\n'.join(failures))

    def shortcut(self):
        env = self.env.copy()
        env.update({'VOICE_SHORTCUT_ROOT': str(self.root), 'VOICE_SHORTCUT_NAME': self.config['shortcut_name']})
        script = r'''
$ErrorActionPreference='Stop'
$desktop=[Environment]::GetFolderPath('Desktop')
$shell=New-Object -ComObject WScript.Shell
$link=$shell.CreateShortcut((Join-Path $desktop ($env:VOICE_SHORTCUT_NAME+'.lnk')))
$link.TargetPath=Join-Path $env:WINDIR 'System32\cmd.exe'
$launcher=Join-Path $env:VOICE_SHORTCUT_ROOT 'One_Click_Start.bat'
$link.Arguments='/c ""'+$launcher+'""'
$link.WorkingDirectory=$env:VOICE_SHORTCUT_ROOT
$link.IconLocation=(Join-Path $env:VOICE_SHORTCUT_ROOT 'assets\voice_clone.ico')+',0'
$link.Description='Voice Clone'
$link.Save()
'''
        self.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script], env=env)

    def launch(self, report):
        for directory in ('outputs', 'logs', 'temp/gradio'):
            safe_join(self.root, directory).mkdir(parents=True, exist_ok=True)
        env = self.env.copy()
        env.update({'CUDA_VISIBLE_DEVICES': str(report['selected_gpu']['index']),
                    'VOICE_CLONE_MODEL_DIR': str(self.model_path())})
        pythonw = self.venv / 'Scripts/pythonw.exe'
        launch_log = self.root / 'logs/desktop-launch.log'
        with launch_log.open('a', encoding='utf-8') as output:
            subprocess.Popen([str(pythonw), str(self.root / 'src/desktop_app.py')], cwd=self.root,
                             env=env, stdout=output, stderr=output, creationflags=subprocess.DETACHED_PROCESS)
        LOG.info('已启动独立桌面窗口。')

    def prepare_repair(self):
        """Preserve the old environment and its CUDA libraries; never recursively delete."""
        env = self.env.copy()
        env['VOICE_REPAIR_ROOT'] = str(self.root)
        probe = self.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
            "$ErrorActionPreference='Stop'; Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('python.exe','pythonw.exe') -and $_.CommandLine -and $_.CommandLine.Contains($env:VOICE_REPAIR_ROOT) -and $_.CommandLine -match '(desktop_app|desktop_worker|[\\/]app)[.]py' } | Select-Object -ExpandProperty ProcessId"], capture=True, env=env)
        if probe:
            raise RuntimeError('请先关闭本项目的桌面窗口或网页服务，再运行 --repair。旧环境将保留备份。')
        stamp = time.strftime('%Y%m%d-%H%M%S')
        if self.venv.exists():
            backup = self.root / ('.venv.backup-' + stamp)
            if self.venv.parent.resolve() != self.root or backup.exists():
                raise RuntimeError('环境备份位置检查失败。')
            self.venv.rename(backup)
            LOG.info('旧环境已保留：%s', backup.name)
        self.libraries = self.cuda / 'runtimes' / (self.instance + '-torch-2.9.1-cu128-r' + stamp) / 'lib'
        self.mark('repair_prepared', config_hash='', validated_gpu=None)

    def deploy(self, report, no_launch=False):
        for directory in ('outputs', 'logs', 'temp/gradio', 'runtime', 'tools/bootstrap', 'cache'):
            safe_join(self.root, directory).mkdir(parents=True, exist_ok=True)
        for path in ('src/desktop_app.py', 'src/desktop_worker.py', 'src/app.py',
                     'src/voice_clone/project.py', 'src/voice_clone/audio.py', 'assets/voice_clone.ico'):
            if not safe_join(self.root, path).is_file():
                raise RuntimeError(f'项目缺少文件：{path}。请保留完整源码文件夹。')
        self.env['CUDA_VISIBLE_DEVICES'] = str(report['selected_gpu']['index'])
        ready = self.state.get('phase') == 'complete' and self.state.get('config_hash') == self.fingerprint
        if not ready or self.verify or not self.python.is_file():
            progress('python', '准备项目内的 Python 环境', 2)
            self.setup_python()
            progress('packages', '下载并安装固定版本依赖', 3)
            self.install_packages()
            progress('cuda', '配置 CUDA 运行库并验证显卡', 4)
            self.relocate_cuda()
            self.validate_runtime(report, load_model=False)
            progress('model', '下载或校验模型与语音编码器', 5)
            self.setup_model()
            progress('audio', '准备音频工具和官方示例', 6)
            self.setup_audio()
            progress('inference', '实际加载模型并生成测试语音', 7)
            self.validate_runtime(report)
            self.mark('complete', config_hash=self.fingerprint)
        else:
            progress('health', '检查已有环境、模型与显卡', 4)
            self.relocate_cuda()
            self.setup_audio()
            signatures = self.verify_model(self.model_path(), cached=True)
            if signatures is None:
                self.setup_model()
                self.validate_runtime(report)
            else:
                self.validate_runtime(report, load_model=False)
                identity = {key: report['selected_gpu'][key] for key in ('index', 'name', 'driver')}
                if self.state.get('validated_gpu') != identity:
                    self.validate_runtime(report)
                else:
                    report['eligible'] = True
                    report['note'] = 'CUDA 运算通过，当前显卡及配置已有短语音生成验证记录。'
                    write_json(self.root / 'logs/hardware-report.json', report)
            self.mark('complete', config_hash=self.fingerprint)
        self.shortcut()
        LOG.info('部署完成。环境：%s；CUDA 库：%s', self.venv, self.libraries)
        if not no_launch and self.config.get('launch_after_setup', True):
            self.launch(report)
        progress('complete', '准备完成，可以开始使用声音克隆', 8)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='Voice Clone 一键环境、模型及 CUDA 部署')
    parser.add_argument('--config', type=Path, default=ROOT / 'deploy_config.json')
    parser.add_argument('--check-only', action='store_true', help='仅检测电脑，不安装和下载模型')
    parser.add_argument('--no-launch', action='store_true', help='部署完成后不打开窗口')
    parser.add_argument('--verify', action='store_true', help='重新校验依赖、模型 SHA256 与加载')
    parser.add_argument('--gpu', type=int, help='选择 nvidia-smi 显示的显卡编号')
    parser.add_argument('--repair', action='store_true', help='备份旧环境后重建，保留模型和音频')
    parser.add_argument('--diagnostics', action='store_true', help='导出不含录音与个人路径的诊断报告')
    args = parser.parse_args(argv)
    (ROOT / 'logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(ROOT / 'logs/deploy.log', encoding='utf-8')])
    try:
        if args.diagnostics:
            from voice_clone.diagnostics import export_report
            LOG.info('诊断报告：%s', export_report())
            return 0
        progress('hardware', '读取电脑配置，按实际能力验证显卡', 1)
        config = load_config(args.config)
        report = hardware_report(config, args.gpu)
        write_json(ROOT / 'logs/hardware-report.json', report)
        gpu = report['selected_gpu']
        LOG.info('CPU：%s；内存：%.1fGB；逻辑核心：%s', report['cpu'], report['ram_gib'], report['logical_cores'])
        LOG.info('GPU：%s；显存：%.1fGB；驱动：%s；计算能力：%s', gpu['name'], gpu['vram_mib'] / 1024,
                 gpu['driver'], gpu['compute_capability'] or '待实际验证')
        if report['ram_gib'] < config['hardware']['ram_warning_gib'] - 0.5:
            LOG.warning('系统内存低于建议值 %sGB，首次加载可能较慢。', config['hardware']['ram_warning_gib'])
        if args.check_only:
            LOG.info('已读取电脑配置，未安装依赖或下载模型。是否能运行会在完整部署时实测。')
            return 0
        with (ROOT / 'logs/deploy.lock').open('a+b') as lock:
            import msvcrt
            lock.seek(0)
            lock.write(b'0')
            lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise RuntimeError('另一个部署进程正在运行，请等待它完成。')
            try:
                installer = Deployer(ROOT, config, args.verify)
                if args.repair:
                    installer.prepare_repair()
                installer.deploy(report, args.no_launch)
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        return 0
    except KeyboardInterrupt:
        LOG.warning('已暂停。重新运行会复用已完成的文件，并从 .part 文件续传。')
        return 130
    except Exception as exc:
        LOG.exception('部署停止：%s', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
