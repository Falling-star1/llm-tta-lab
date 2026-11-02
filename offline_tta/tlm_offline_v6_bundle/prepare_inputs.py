"""Split AdaptEval JSON into answer-free inputs and scoring-only references.

This standard-library-only preparation command does not load a model. Training
should receive inputs.jsonl only; references.jsonl is reserved for evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from input_tokens import sanitize_record


def _write_jsonl(path, rows):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _publish_linux_directory(staging, output):
    """Use the kernel interface when older libc does not export renameat2."""
    import ctypes
    import platform
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    args = (-100, os.fsencode(staging), -100, os.fsencode(output), 1)
    types = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if rename is not None:
        rename.argtypes = types
        rename.restype = ctypes.c_int
        result = rename(*args)
    else:
        numbers = {'x86_64': 316, 'amd64': 316, 'aarch64': 276, 'arm64': 276}
        number = numbers.get(platform.machine().lower())
        if number is None:
            raise OSError('No renameat2 syscall mapping for this Linux architecture')
        syscall = libc.syscall
        syscall.argtypes = [ctypes.c_long] + types
        syscall.restype = ctypes.c_long
        result = syscall(number, *args)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def _publish_new_directory(staging, output):
    """Atomic publication with no overwrite, including a concurrent empty dir.

    Windows rename refuses an existing destination. Linux requires the explicit
    RENAME_NOREPLACE flag: ordinary POSIX rename could replace an empty directory.
    Refuse unsupported systems rather than weaken this guarantee.
    """
    if os.name == 'nt':
        os.rename(staging, output)
    elif sys.platform.startswith('linux'):
        _publish_linux_directory(staging, output)
    else:
        raise OSError('Atomic no-replace publication is supported on Windows and Linux only')


def prepare(data, output, expected_sha256=None, expected_records=None):
    """Validate all rows, stage complete files, then publish a new directory.

    A sibling exclusive lock serializes this command's writers. Existing output
    paths (including files, empty directories and symlinks) are always rejected.
    expected_records=5000 is required by the formal experiment protocol; smaller
    counts can be explicitly supplied for engineering fixtures and pilots.
    """
    data, output = Path(data), Path(output).absolute()
    if os.path.lexists(output):
        raise FileExistsError('Output already exists: ' + str(output))
    payload = data.read_bytes()
    source_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(char not in '0123456789abcdefABCDEF' for char in expected_sha256)):
            raise ValueError('expected_sha256 must contain 64 hexadecimal characters')
        if source_sha256 != expected_sha256.lower():
            raise ValueError('Source SHA256 does not match')
    if expected_records is not None and (type(expected_records) is not int or expected_records < 1):
        raise ValueError('expected_records must be a positive integer')
    try:
        rows = json.loads(payload.decode('utf-8-sig'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('Source must be UTF-8 JSON') from error
    if not isinstance(rows, list) or not rows:
        raise ValueError('Source must be a nonempty JSON list')
    if expected_records is not None and len(rows) != expected_records:
        raise ValueError('Expected %d records, found %d' % (expected_records, len(rows)))
    inputs, references = [], []
    for index, raw in enumerate(rows):
        try:
            inputs.append(sanitize_record(raw, index))
            if 'output' not in raw or not isinstance(raw['output'], str):
                raise ValueError('Reference output must be a string')
            references.append({'source_index': index, 'output': raw['output']})
        except ValueError as error:
            raise ValueError('Invalid source row %d: %s' % (index, error)) from error

    lock = output.with_name('.' + output.name + '.prepare.lock')
    staging = None
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if os.path.lexists(output):
            raise FileExistsError('Output already exists: ' + str(output))
        staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.staging-', dir=output.parent))
        _write_jsonl(staging / 'inputs.jsonl', inputs)
        _write_jsonl(staging / 'references.jsonl', references)
        manifest = {'schema_version': 1, 'record_count': len(inputs),
                    'source_sha256': source_sha256,
                    'files': {name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                              for name in ['inputs.jsonl', 'references.jsonl']}}
        with (staging / 'manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(output):
            raise FileExistsError('Output already exists: ' + str(output))
        _publish_new_directory(staging, output)
        staging = None
        return manifest
    finally:
        try:
            if staging is not None:
                shutil.rmtree(staging)
        finally:
            os.close(lock_fd)
            lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='new directory under an existing parent')
    parser.add_argument('--expected-sha256')
    parser.add_argument('--expected-records', type=int,
                        help='use 5000 for formal experiment inputs')
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.data, args.output, args.expected_sha256, args.expected_records)
    except (OSError, ValueError) as error:
        parser.exit(2, 'Preparation failed: %s\n' % error)
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))


if __name__ == '__main__':
    main()
