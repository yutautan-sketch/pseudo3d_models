#!/usr/bin/env python3
"""Replay patches without a Git repository, index, commits or network."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
manifest = json.loads((OUT / 'manifest.json').read_text())
def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
def descriptor(p):
    if not p.exists():
        return None
    return {'mode': '100755' if p.stat().st_mode & 0o111 else '100644',
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

with tempfile.TemporaryDirectory(prefix='replay-', dir=OUT) as tmp:
    tree = Path(tmp)
    raw = run(['git', '-C', str(ROOT), 'ls-tree', '-rz', manifest['base']]).stdout
    for record in raw.split(b'\0'):
        if not record:
            continue
        meta, name = record.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        if kind != 'blob':
            continue
        p = tree / name.decode()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(run(['git', '-C', str(ROOT), 'cat-file', 'blob', oid]).stdout)
        p.chmod(int(mode, 8) & 0o777)
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(OUT))
    report = []
    introduced = set()
    for step in manifest['steps']:
        patch = OUT / step['patch']
        assert hashlib.sha256(patch.read_bytes()).hexdigest() == step['sha256']
        for name, expected in step['before'].items():
            assert descriptor(tree / name) == expected, ('before', step['number'], name)
        run(['git', 'apply', '--check', '--whitespace=nowarn', str(patch)], cwd=tree, env=env)
        run(['git', 'apply', '--whitespace=nowarn', str(patch)], cwd=tree, env=env)
        for name, expected in step['after'].items():
            assert descriptor(tree / name) == expected, ('after', step['number'], name)
            if expected is not None:
                introduced.add(name)
            else:
                introduced.discard(name)
        py = sh = embeds = 0
        for name in sorted(introduced):
            p = tree / name
            if p.suffix == '.py':
                syntax = ast.parse(p.read_text(), filename=name)
                py += 1
                root = tree / p.relative_to(tree).parts[0]
                for node in ast.walk(syntax):
                    if not isinstance(node, ast.ImportFrom) or not node.module:
                        continue
                    if node.level:
                        base = p.parent
                        for _ in range(node.level - 1):
                            base = base.parent
                        base /= Path(*node.module.split('.'))
                    else:
                        if not node.module.startswith(('pseudo3d', 'stage5', 'checks',
                            'evaluate_stage5', 'infer_stage5', 'train_stage5', 'export_anonymized_stage5_metrics')):
                            continue
                        base = root / Path(*node.module.split('.'))
                    assert base.is_dir() or base.with_suffix('.py').is_file(), (
                        'missing local import', step['number'], name, ast.unparse(node))
            elif p.suffix == '.sh':
                run(['bash', '-n', str(p)])
                sh += 1
                for payload in re.findall(r"<<[\-]?['\"]?PY['\"]?[^\n]*\n(.*?)\nPY(?:\n|$)", p.read_text(), re.S):
                    ast.parse(payload, filename=name + ':heredoc')
                    embeds += 1
        report.append({'step': step['number'], 'python_ast': py, 'bash_n': sh,
                       'python_heredocs': embeds, 'patch_and_hashes': 'pass', 'local_import_paths': 'pass'})
        print(f"{step['number']:03d}: replay, hashes, {py} Python, {sh} shell, {embeds} heredocs PASS", flush=True)
    for name, expected in manifest['expected_final'].items():
        assert descriptor(tree / name) == expected, ('final', name)
    optional = json.loads((OUT / 'optional-environment.json').read_text())
    patch = OUT / 'optional-environment.patch'
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == optional['sha256']
    for name, expected in optional['before'].items():
        assert descriptor(tree / name) == expected
    run(['git', 'apply', '--check', '--whitespace=nowarn', str(patch)], cwd=tree, env=env)
    run(['git', 'apply', '--whitespace=nowarn', str(patch)], cwd=tree, env=env)
    for name, expected in optional['after'].items():
        assert descriptor(tree / name) == expected
    (OUT / 'verification.json').write_text(json.dumps({'steps': report,
        'final_source_match': True, 'runtime_tests': 'not run: missing dependencies/data/GPU',
        'optional_environment_patch': 'apply and hashes pass; installation not run',
        'git_writes': False}, indent=2) + '\n')
print('All intermediate trees passed static checks; final file bytes/modes verified.')
