"""Answer isolation and the frozen Source prompt/truncation contract."""
import hashlib
import ctypes
import errno
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(Path(__file__).resolve().parent))


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(value) for value in ids)


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode('utf-8'))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode('utf-8', errors='replace')


class FakeTemplate:
    def encode_oneturn(self, tokenizer, messages, system):
        assert messages[0]['role'] == 'user'
        assert messages[1] == {'role': 'assistant', 'content': ''}
        return ([101, 102] + tokenizer.encode(system) + [103]
                + tokenizer.encode(messages[0]['content']) + [104, 105], [])


def frozen_encode_prompt():
    """Load the self-contained test fixture, without importing torch/PEFT."""
    return importlib.import_module('frozen_prompt_reference').encode_prompt


class InputTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((BUNDLE / 'input_tokens.py').is_file(),
                        'answer-independent input implementation is missing')
        self.module = importlib.import_module('input_tokens')

    def test_none_system_preserves_the_models_default_system(self):
        class DefaultTemplate(FakeTemplate):
            def encode_oneturn(self, tokenizer, messages, system):
                return super().encode_oneturn(tokenizer,messages,'default' if system is None else system)
        record={'source_index':0,'instruction':'question','input':''}
        ids,_=self.module.encode_input(record,CharacterTokenizer(),DefaultTemplate(),None)
        expected=frozen_encode_prompt()(record,CharacterTokenizer(),DefaultTemplate(),None,4096)[1]
        self.assertEqual(ids,expected)

    def test_frozen_fixture_is_bundled_for_standalone_upload(self):
        self.assertTrue((BUNDLE / 'tests' / 'frozen_prompt_reference.py').is_file(),
                        'Portable frozen prompt fixture is missing')

    def test_answer_replacement_cannot_change_tokens_or_identity(self):
        first = {'instruction': '解释岩石', 'input': '🙂 sediment', 'output': 'short'}
        second = dict(first, output='different' * 10000, answer=['untrusted'])
        left = self.module.sanitize_record(first, 7)
        right = self.module.sanitize_record(second, 7)
        self.assertEqual(left, {'source_index': 7, 'instruction': '解释岩石',
                                'input': '🙂 sediment'})
        self.assertEqual(left, right)
        self.assertEqual(self.module.encode_input(left, CharacterTokenizer(), FakeTemplate(), '', 16),
                         self.module.encode_input(right, CharacterTokenizer(), FakeTemplate(), '', 16))

    def test_strict_schema_rejects_answers_bad_indexes_and_empty_text(self):
        good = {'source_index': 0, 'instruction': 'question', 'input': ''}
        self.module.require_input_record(good)
        for bad in [dict(good, output='secret'), dict(good, source_index=True),
                    dict(good, source_index=-1), dict(good, source_index=1.0),
                    dict(good, instruction=None), dict(good, input=[]),
                    dict(good, instruction='  ', input='\t'),
                    {'instruction': 'q', 'input': ''}]:
            with self.subTest(record=bad), self.assertRaises(ValueError):
                self.module.require_input_record(bad)
        for bad in [{}, {'instruction': 1, 'input': ''},
                    {'instruction': 'q'}, {'instruction': '', 'input': ''}]:
            with self.subTest(raw=bad), self.assertRaises(ValueError):
                self.module.sanitize_record(bad, 0)

    def test_prompt_preserves_text_and_only_adds_newline_for_context(self):
        for instruction, context, expected in [(' q ', '', ' q '),
                                                ('q', ' x ', 'q\n x '),
                                                ('', 'x', '\nx')]:
            record = self.module.sanitize_record({'instruction': instruction, 'input': context}, 0)
            self.assertEqual(self.module.prompt_text(record), expected)

    def test_tokens_and_truncation_match_frozen_source_with_unicode(self):
        tokenizer, template = CharacterTokenizer(), FakeTemplate()
        old_encode = frozen_encode_prompt()
        for text in ['a', '地层🙂 α\n' * 15]:
            record = self.module.sanitize_record({'instruction': text, 'input': 'context'}, 2)
            for budget in [12, 25, 4096]:
                with self.subTest(text=text[:8], budget=budget):
                    _, old_ids, old_info = old_encode(record, tokenizer, template, 'sys', budget)
                    ids, info = self.module.encode_input(record, tokenizer, template, 'sys', budget)
                    self.assertEqual(ids, old_ids)
                    self.assertEqual({key: info[key] for key in old_info}, old_info)
                    self.assertEqual(info['source_index'], 2)
                    expected_hash = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
                    self.assertEqual(info['token_sha256'], expected_hash)

    def test_impossible_template_budget_and_invalid_budget_fail(self):
        record = self.module.sanitize_record({'instruction': 'long input', 'input': ''}, 0)
        for budget in [1, 5, 0, -1, True, 1.2]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.module.encode_input(record, CharacterTokenizer(), FakeTemplate(), '', budget)

    def test_unicode_token_fragment_decode_matches_source(self):
        record = self.module.sanitize_record({'instruction': 'a🙂地层' * 8, 'input': ''}, 0)
        tokenizer, template = ByteTokenizer(), FakeTemplate()
        for budget in range(9, 20):
            _, expected, _ = frozen_encode_prompt()(record, tokenizer, template, '', budget)
            actual, _ = self.module.encode_input(record, tokenizer, template, '', budget)
            self.assertEqual(actual, expected)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((BUNDLE / 'prepare_inputs.py').is_file(),
                        'separate input/reference preparation is missing')
        self.module = importlib.import_module('prepare_inputs')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data.json'
        self.rows = [{'instruction': '问题一', 'input': '', 'output': '答案🙂'},
                     {'instruction': 'q2', 'input': 'context', 'output': 'answer2'}]
        self.data.write_text(json.dumps(self.rows, ensure_ascii=False), encoding='utf-8')

    def test_prepare_separates_references_records_hashes_and_refuses_overwrite(self):
        output = self.root / 'prepared'
        digest = hashlib.sha256(self.data.read_bytes()).hexdigest()
        manifest = self.module.prepare(self.data, output, expected_sha256=digest, expected_records=2)
        inputs = [json.loads(line) for line in (output / 'inputs.jsonl').read_text(encoding='utf-8').splitlines()]
        refs = [json.loads(line) for line in (output / 'references.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(inputs, [{'source_index': index, 'instruction': row['instruction'], 'input': row['input']}
                                  for index, row in enumerate(self.rows)])
        self.assertEqual(refs, [{'source_index': index, 'output': row['output']}
                                for index, row in enumerate(self.rows)])
        self.assertEqual(manifest['source_sha256'], digest)
        self.assertEqual(manifest['record_count'], 2)
        for name in ['inputs.jsonl', 'references.jsonl']:
            self.assertEqual(manifest['files'][name], hashlib.sha256((output / name).read_bytes()).hexdigest())
        self.assertEqual(json.loads((output / 'manifest.json').read_text()), manifest)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.module.prepare(self.data, output)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_validates_whole_file_hash_and_count_before_publishing(self):
        for options in [{'expected_sha256': '0' * 64}, {'expected_records': 5000}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.module.prepare(self.data, self.root / 'out', **options)
            self.assertFalse((self.root / 'out').exists())
        for invalid in [{}, [], self.rows + [{'instruction': 'bad', 'input': '', 'output': 1}],
                        self.rows + [{'instruction': 'bad', 'output': 'answer'}]]:
            self.data.write_text(json.dumps(invalid), encoding='utf-8')
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.module.prepare(self.data, self.root / 'out')
            self.assertEqual(sorted(path.name for path in self.root.iterdir()), ['data.json'])

    def test_write_failure_does_not_publish_partial_directory(self):
        with mock.patch.object(self.module, '_write_jsonl', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.module.prepare(self.data, self.root / 'out')
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ['data.json'])

    def test_atomic_publication_refuses_even_empty_existing_directory(self):
        self.assertTrue(hasattr(self.module, '_publish_new_directory'),
                        'atomic no-replace directory publication is missing')
        stage, existing = self.root / 'stage', self.root / 'existing'
        stage.mkdir()
        (stage / 'complete').write_text('ready')
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            self.module._publish_new_directory(stage, existing)
        self.assertTrue((stage / 'complete').exists())
        self.assertEqual(list(existing.iterdir()), [])

    def test_cli_runs_without_model_packages(self):
        result = subprocess.run([sys.executable, str(BUNDLE / 'prepare_inputs.py'),
                                 '--data', str(self.data), '--output', str(self.root / 'cli'),
                                 '--expected-records', '2'], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['record_count'], 2)

    def test_linux_publication_supports_libc_without_renameat2(self):
        syscall = mock.Mock(return_value=0)
        with mock.patch('ctypes.CDLL', return_value=SimpleNamespace(syscall=syscall)):
            with mock.patch('platform.machine', return_value='x86_64'):
                self.module._publish_linux_directory('/tmp/stage', '/tmp/output')
        self.assertEqual(syscall.call_args.args, (316, -100, b'/tmp/stage', -100, b'/tmp/output', 1))

    def test_linux_syscall_fallback_preserves_no_replace_error(self):
        def existing(*args):
            ctypes.set_errno(errno.EEXIST)
            return -1
        with mock.patch('ctypes.CDLL', return_value=SimpleNamespace(syscall=mock.Mock(side_effect=existing))):
            with mock.patch('platform.machine', return_value='x86_64'):
                with self.assertRaises(FileExistsError):
                    self.module._publish_linux_directory('/tmp/stage', '/tmp/output')


if __name__ == '__main__':
    unittest.main()
