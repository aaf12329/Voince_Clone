"""Character isolation/export checks on synthetic audio and tiny temporary files.

Call tree: CharacterTests -> import_character -> row/path/hash validation;
ExportTests -> validate_export -> metadata/hash checks before any GPU loading.
"""
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from voice_training.common import io
from voice_training.features import datasets, preview


class CharacterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        template = self.root / 'template.json'
        original = Path(__file__).resolve().parents[1] / 'configs/training.example.json'
        template.write_bytes(original.read_bytes())
        self.stack.enter_context(patch.object(io, 'ROOT', self.root))
        self.stack.enter_context(patch.multiple(datasets, ROOT=self.root, TEMPLATE_CONFIG=template,
            CHARACTER_CONFIGS=self.root / 'configs/characters', CHARACTER_DATA=self.root / 'data/characters'))
        self.raw = self.root / 'data/raw/test'
        self.raw.mkdir(parents=True)
        self.rows = []
        for index, identifier in enumerate(('train_001','train_002','validation_001')):
            sf.write(self.raw / (identifier + '.wav'), np.sin(np.arange(8000) / (20 + index)) * 0.1, 16000)
            self.rows.append({'id':identifier,'audio':identifier + '.wav','text':'准确台词 ' + str(index),
                              'split':'validation' if index == 2 else 'train'})
        self.manifest = self.raw / 'manifest.jsonl'
        self.save_manifest()

    def save_manifest(self):
        self.manifest.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in self.rows), encoding='utf-8')

    def test_imported_roles_have_independent_relative_output_paths(self):
        first = datasets.import_character('hero_one', 'data/raw/test/manifest.jsonl')
        second = datasets.import_character('hero_two', 'data/raw/test/manifest.jsonl', language='English')
        a, b = io.read_json(self.root / first['config']), io.read_json(self.root / second['config'])
        self.assertEqual((first['train_samples'],first['validation_samples']), (2,1))
        self.assertNotEqual(a['checkpoint_directory'],b['checkpoint_directory'])
        self.assertNotEqual(a['export_directory'],b['export_directory'])
        self.assertEqual(b['language'],'English')
        self.assertTrue(all(not Path(a[key]).is_absolute() for key in ('profiles','teacher_cache','checkpoint_directory')))

    def test_validation_reference_and_existing_role_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'训练集'):
            datasets.import_character('hero', 'data/raw/test/manifest.jsonl', reference_id='validation_001')
        datasets.import_character('hero', 'data/raw/test/manifest.jsonl')
        with self.assertRaisesRegex(ValueError,'已存在'):
            datasets.import_character('hero', 'data/raw/test/manifest.jsonl')

    def test_duplicate_recording_cannot_leak_into_validation(self):
        self.rows[-1]['audio'] = 'train_001.wav'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError,'泄漏'):
            datasets.import_character('hero', 'data/raw/test/manifest.jsonl')
        self.assertFalse((self.root / 'configs/characters/hero.json').exists())

    def test_recording_path_and_role_name_cannot_escape_project(self):
        self.rows[0]['audio'] = '../../../../outside.wav'
        self.save_manifest()
        with self.assertRaises(ValueError):
            datasets.import_character('hero', 'data/raw/test/manifest.jsonl')
        with self.assertRaises(ValueError):
            datasets.import_character('../escape', 'data/raw/test/manifest.jsonl')

    def test_changed_recording_does_not_reuse_old_profile_hash(self):
        result = datasets.import_character('hero', 'data/raw/test/manifest.jsonl')
        config = io.read_json(self.root / result['config'])
        (self.raw / 'train_001.wav').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'已改变'):
            io.load_profile(config, 'train_001')


class ExportTests(unittest.TestCase):
    def validate(self, metadata, config, weight=b'weights'):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            io.write_json(root / 'distillation.json', metadata)
            io.write_json(root / 'config.json', config)
            (root / 'model.safetensors').write_bytes(weight)
            return preview.validate_export(root, 'hero')

    def test_untrained_export_is_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError,'训练记录'):
            self.validate({'checkpoint_step':0,'speaker':'hero'},{'tts_model_type':'custom_voice'})

    def test_base_model_cannot_be_presented_as_a_character_export(self):
        with self.assertRaisesRegex(ValueError,'角色模型'):
            self.validate({'checkpoint_step':1,'speaker':'hero'},{'tts_model_type':'base'})

    def test_altered_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'校验失败'):
            self.validate({'checkpoint_step':1,'speaker':'hero','model_sha256':'bad'},
                          {'tts_model_type':'custom_voice'})


if __name__ == '__main__':
    unittest.main()
