#!/usr/bin/env python3
"""User-run helper. Preflight is read-only; prepare/step/finalize write Git.

The assistant must not run the mutation subcommands in the container.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BUNDLE = Path(__file__).resolve().parent
M = json.loads((BUNDLE / 'manifest.json').read_text())
BRANCH = 'split/uncommitted-20260912'
def fail(message):
    raise SystemExit(message)
def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.STDOUT)
def execute(repo, *args):
    subprocess.run(['git', '-C', str(repo), *args], check=True)
def digest(data):
    return hashlib.sha256(data).hexdigest()
def check_bundle():
    for step in M['steps']:
        if digest((BUNDLE / step['patch']).read_bytes()) != step['sha256']:
            fail('Patch checksum mismatch: ' + step['patch'])
def check_files(repo, expected):
    for name, info in expected.items():
        path = repo / name
        if info is None:
            if path.exists() or path.is_symlink():
                fail('Expected absent path: ' + name)
        else:
            if path.is_symlink() or not path.is_file() or digest(path.read_bytes()) != info['sha256']:
                fail('File content mismatch: ' + name)
            if os.name != 'nt':
                mode = '100755' if path.stat().st_mode & 0o111 else '100644'
                if mode != info['mode']:
                    fail('File mode mismatch: ' + name)
def branch(repo):
    return git(repo, 'symbolic-ref', '--quiet', '--short', 'HEAD').decode().strip()
def head(repo):
    return git(repo, 'rev-parse', 'HEAD').decode().strip()
def check_no_operation(repo):
    for name in ['MERGE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD', 'rebase-merge', 'rebase-apply']:
        path = Path(git(repo, 'rev-parse', '--git-path', name).decode().strip())
        if not path.is_absolute():
            path = repo / path
        if path.exists():
            fail('Finish the existing Git operation first: ' + name)
def source_preflight(repo):
    check_bundle()
    if head(repo) != M['base']:
        fail('Source HEAD does not match the recorded base. Do not reset it; regenerate the bundle.')
    branch(repo)  # Require attached branch for later finalization.
    check_no_operation(repo)
    check_files(repo, M['expected_source'])
    known = set(M['expected_source'])
    others = git(repo, 'ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
    extra = [p for p in others if p and p not in known and not p.startswith(('.venvs/', '.tmp/'))]
    if extra:
        fail('Unaccounted untracked files: ' + repr(extra[:20]))
    index_paths = set(git(repo, 'ls-files', '-z').decode().split('\0')) - {''}
    if index_paths - known - set(M['gitlinks']):
        fail('Unexpected staged paths: ' + repr(sorted(index_paths - known - set(M['gitlinks']))))
    # Deliberately different staged content would otherwise disappear at finalization.
    for entry in git(repo, 'ls-files', '--stage', '-z').split(b'\0'):
        if not entry:
            continue
        meta, raw_name = entry.split(b'\t', 1)
        mode, oid, stage = meta.decode().split()
        name = raw_name.decode()
        if stage != '0':
            fail('Unmerged index path: ' + name)
        if mode == '160000':
            if M['gitlinks'].get(name) != oid:
                fail('Changed submodule index: ' + name)
            continue
        observed = digest(git(repo, 'cat-file', 'blob', oid))
        allowed = M['allowed_index'].get(name, [])
        if {'mode': mode, 'sha256': observed} not in allowed:
            fail('Index contains uncaptured content: ' + name)
    for name, oid in M['gitlinks'].items():
        path = repo / name
        if (path / '.git').exists():
            if head(path) != oid or git(path, 'status', '--porcelain').strip():
                fail('Submodule changed: ' + name)
    print('Source HEAD, all file contents/modes, index contents and bundle checksums match.')
def history(work):
    if branch(work) != BRANCH:
        fail('Not on the dedicated split branch')
    execute(work, 'merge-base', '--is-ancestor', M['base'], 'HEAD')
    commits = git(work, 'rev-list', '--reverse', M['base'] + '..HEAD').decode().splitlines()
    if len(commits) > len(M['steps']):
        fail('Unexpected extra commits on split branch')
    for commit, step in zip(commits, M['steps']):
        parents = git(work, 'rev-list', '--parents', '-n', '1', commit).decode().split()
        if len(parents) != 2:
            fail('Unexpected merge commit')
        title = git(work, 'show', '-s', '--format=%s', commit).decode().strip()
        if title != step['title']:
            fail('Unexpected commit title at ' + commit)
        changed = set(git(work, 'diff-tree', '--no-commit-id', '--name-only', '-r', '--no-renames', commit).decode().splitlines())
        if changed != set(step['after']):
            fail('Unexpected committed paths at ' + commit)
        for name, info in step['after'].items():
            if info is None:
                continue
            if digest(git(work, 'show', commit + ':' + name)) != info['sha256']:
                fail('Unexpected committed bytes at ' + commit + ':' + name)
    return commits
def verify_step_sources(work, step):
    for name, info in step['after'].items():
        if info is None:
            continue
        p = work / name
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=name)
        if p.suffix == '.sh':
            subprocess.run(['bash', '-n', str(p)], check=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['preflight', 'prepare', 'step', 'finalize'])
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--worktree', type=Path)
    p.add_argument('--number', type=int)
    a = p.parse_args()
    repo = a.repo.resolve()
    if Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip()).resolve() != repo:
        fail('--repo must name the repository root')
    check_bundle()
    if a.action == 'preflight':
        source_preflight(repo)
        return
    if a.worktree is None:
        fail('--worktree is required')
    work = a.worktree.resolve()
    if work == repo or repo in work.parents or work in repo.parents:
        fail('Use a separate sibling worktree, outside the source repository')
    if a.action == 'prepare':
        source_preflight(repo)
        if work.exists():
            fail('Worktree destination already exists; use step to resume an existing preparation')
        # Branch creation also refuses an existing branch; nothing is force-replaced.
        backup = BUNDLE / 'user-backup'
        backup.mkdir(exist_ok=True)
        backups = {
            'source.json': (json.dumps({'repo': str(repo), 'branch': branch(repo),
                'head': head(repo), 'worktree': str(work)}, indent=2) + '\n').encode(),
            'staged.patch': git(repo, 'diff', '--cached', '--binary'),
            'unstaged.patch': git(repo, 'diff', '--binary'),
        }
        for name, payload in backups.items():
            path = backup / name
            if path.exists() and path.read_bytes() != payload:
                fail('Existing backup differs; preserve it and investigate: ' + str(path))
            path.write_bytes(payload)
        execute(repo, 'worktree', 'add', '-b', BRANCH, str(work), M['base'])
        print('Prepared separate worktree. Source worktree and index are unchanged.')
        return
    check_no_operation(work)
    commits = history(work)
    if a.action == 'step':
        if a.number is None or not 1 <= a.number <= len(M['steps']):
            fail('--number must be within the series')
        step = M['steps'][a.number - 1]
        if a.number <= len(commits):
            print('Already committed and verified:', a.number, commits[a.number - 1])
            return
        if a.number != len(commits) + 1:
            fail('Apply steps in order; next is ' + str(len(commits) + 1))
        status = git(work, 'status', '--porcelain', '--untracked-files=all')
        if status.strip():
            # Resume a failed/interrupted commit only if its entire staged tree is exact.
            check_files(work, step['after'])
            if git(work, 'diff', '--name-only').strip():
                fail('Unstaged changes exist in the replay worktree')
            staged = set(git(work, 'diff', '--cached', '--name-only', '--no-renames').decode().splitlines())
            if staged != set(step['after']) or git(work, 'ls-files', '--others', '--exclude-standard').strip():
                fail('Unexpected staged or untracked changes; inspect worktree manually')
            for name, info in step['after'].items():
                if info and digest(git(work, 'show', ':' + name)) != info['sha256']:
                    fail('Staged bytes differ: ' + name)
        else:
            check_files(work, step['before'])
            execute(work, 'apply', '--check', '--index', '--whitespace=nowarn', str(BUNDLE / step['patch']))
            execute(work, 'apply', '--index', '--whitespace=nowarn', str(BUNDLE / step['patch']))
        check_files(work, step['after'])
        verify_step_sources(work, step)
        execute(work, 'commit', '-m', step['title'])
        history(work)  # Detect hook-induced committed changes immediately.
        print('Committed step', a.number, head(work))
        return
    if len(commits) != len(M['steps']):
        fail('Finish all steps before finalization')
    if git(work, 'status', '--porcelain').strip():
        fail('Split worktree must be clean')
    check_files(work, M['expected_final'])
    source_preflight(repo)
    prepared = json.loads((BUNDLE / 'user-backup/source.json').read_text())
    if prepared != {'repo': str(repo), 'branch': branch(repo), 'head': head(repo), 'worktree': str(work)}:
        fail('Original repository/branch differs from prepare record')
    # This advances only the original branch and index. --mixed never checks out files.
    execute(repo, 'reset', '--mixed', head(work))
    check_files(repo, M['expected_source'])
    print('Original branch/index now match the split history; working files were preserved.')
    print('The local .venvs directory remains untracked until optional environment cleanup.')

if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as e:
        if e.output:
            print(e.output.decode(errors='replace'), file=sys.stderr)
        raise SystemExit(e.returncode)
