"""Training entry orchestration; this layer coordinates independent feature modules.

Call tree: main -> parse_args -> execute -> import_character OR stage workflow
(prepare -> teacher cache -> train -> export), or exported-role preview.
Data: relative JSONL/config -> role-specific artifacts, checkpoints and model directory.
"""
import argparse
import json
import sys
import traceback

from .common.io import append_text, file_hash, inside_project, read_json
from .common.lease import InferenceLease
from .common.paths import ERROR_LOG, TRAINING_LOCK
from .common.state import load_training_config, manifest, source_identity
from .features.datasets import import_character
from .features.exporting import export
from .features.optimization import train
from .features.preparation import prepare
from .features.preview import generate
from .features.teacher import teacher_cache


def parse_args():
    """Parse explicit stage and role options for the PowerShell/Python entries."""
    parser = argparse.ArgumentParser(description='独立角色声音训练：导入录音、编码、老师缓存、LoRA 蒸馏、导出和试听。')
    parser.add_argument('command', choices=('import','validate','prepare','teacher','train','export','all','speak'))
    parser.add_argument('--config', default='configs/training.example.json')
    parser.add_argument('--name')
    parser.add_argument('--manifest')
    parser.add_argument('--language', default='Chinese')
    parser.add_argument('--reference-id')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--text')
    parser.add_argument('--output', default='outputs/preview.wav')
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps 必须大于 0')
    if args.command == 'import' and (not args.name or not args.manifest):
        parser.error('import 需要 --name 和项目相对路径 --manifest')
    if args.command == 'speak' and not args.text:
        parser.error('speak 需要 --text')
    return args


def check_teacher_cache(config, data):
    """Reject mismatched teacher artifacts before optimizer startup."""
    directory = inside_project(config['teacher_cache'])
    cache = read_json(directory / 'manifest.json')
    if cache['signature'] != data['signature'] or any(row['id'] not in cache['files'] or
            file_hash(directory / (row['id'] + '.pt')) != cache['files'][row['id']] for row in data['samples']):
        raise ValueError('老师缓存缺失或已改变，请先运行 teacher 阶段。')


def execute(args):
    """Coordinate the requested stages without importing one feature from another."""
    if args.command == 'import':
        return import_character(args.name, args.manifest, args.language, args.reference_id)
    config = load_training_config(args.config)
    if args.command == 'speak':
        return generate(config, args.text, args.output)
    if not inside_project(config['profiles']).is_file():
        raise FileNotFoundError('缺少角色录音清单。先将录音放到 data/raw，再运行 New-Character.ps1。')
    if args.command == 'validate':
        selected, excluded, _ = source_identity(config)
        return {'valid':True, 'speaker':config['speaker_name'], 'samples':len(selected), 'excluded':excluded}
    data = None
    if args.command in ('prepare','all'):
        data = prepare(config)
    elif args.command != 'export':
        data = manifest(config)
    if args.command in ('teacher','all'):
        teacher_cache(config, data)
    if args.command in ('train','all'):
        check_teacher_cache(config, data)
        train(config, data, args.resume, args.max_steps)
    if args.command in ('export','all'):
        destination = export(config)
        return {'export': str(destination.relative_to(inside_project('.'))) if destination else None}
    return {'stage':args.command, 'speaker':config['speaker_name'], 'completed':True}


def main():
    """Own the project lock and report an actionable error at the command boundary."""
    args = parse_args()
    lease = InferenceLease(TRAINING_LOCK)
    try:
        lease.acquire()
        result = execute(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print('已停止；训练中已完成的优化步会保存，可用 -Resume 继续。', file=sys.stderr)
        return 130
    except Exception as exc:
        append_text(ERROR_LOG, traceback.format_exc() + '\n')
        print('未完成：' + str(exc) + '\n详情见 logs/training-error.log。', file=sys.stderr)
        return 1
    finally:
        lease.close()
