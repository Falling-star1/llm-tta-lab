"""Check frozen snapshots, delivery hashes, Python syntax and current doc links."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = json.loads((ROOT / 'archive/manifest.json').read_text(encoding='utf-8'))
    archived = ROOT / 'archive/github-main-20261018'
    for name, expected in manifest['main_snapshot_sha256'].items():
        assert sha(archived / name) == expected, f'Archive changed: {name}'
    old_branch = ROOT / 'archive/fix-label-isolation-ec8dae8.zip'
    assert sha(old_branch) == manifest['feature_zip_sha256'], 'Feature archive changed'
    with zipfile.ZipFile(old_branch) as snapshot:
        assert snapshot.testzip() is None, 'Invalid feature ZIP'
    bundles = sorted((ROOT / 'offline_tta').glob('*/SHA256SUMS'))
    required = {'tlm_stage1_bundle', 'tlm_source_smoke_bundle',
                'tlm_tent_preflight_bundle', 'tlm_tent_offline_smoke_bundle',
                'tlm_tent_full_3gpu_det_bundle', 'tlm_offline_v1_bundle',
                'tlm_offline_v2_bundle', 'tlm_offline_v3_bundle'}
    assert required <= {p.parent.name for p in bundles}, 'Missing required frozen delivery'
    hashed_files = 0
    for checksum in bundles:
        for row in checksum.read_text(encoding='utf-8').splitlines():
            expected, name = row.split(maxsplit=1)
            path = checksum.parent / name.lstrip('*')
            assert path.resolve().is_relative_to(checksum.parent.resolve()), 'Unsafe checksum path'
            assert sha(path) == expected, f'Bundle hash mismatch: {path.relative_to(ROOT)}'
            hashed_files += 1
    syntax_files = (list((ROOT / 'offline_tta').rglob('*.py'))
                    + list((ROOT / 'scripts').glob('*.py'))
                    + list((ROOT / 'local_repro').rglob('*.py')))
    for path in syntax_files:
        compile(path.read_bytes(), str(path), 'exec')
    documents = list(ROOT.glob('*.md')) + list((ROOT / 'docs').rglob('*.md')) + [ROOT / 'offline_tta/README.md']
    link_count = 0
    for document in documents:
        if 'records' in document.relative_to(ROOT).parts:
            continue  # Historical evidence retains its original prose and paths.
        for target in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            target = target.strip('<>').split('#')[0]
            if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                continue
            path = document.parent / unquote(target)
            assert path.exists(), f'Broken link in {document.relative_to(ROOT)}: {target}'
            link_count += 1
    forbidden = {'.pt', '.pth', '.ckpt', '.bin', '.safetensors', '.whl'}
    files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts]
    for path in files:
        assert path.suffix not in forbidden, f'Unexpected binary: {path}'
        assert path.stat().st_size < 2 * 1024 * 1024, f'Unexpected large file: {path}'
    print(json.dumps({'status': 'PASS', 'archived_main_files': len(manifest['main_snapshot_sha256']),
        'frozen_bundles': len(bundles), 'bundle_hashed_files': hashed_files,
        'python_syntax_files': len(syntax_files), 'local_doc_links': link_count,
        'files_checked': len(files)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
