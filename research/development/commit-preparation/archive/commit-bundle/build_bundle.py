#!/usr/bin/env python3
"""Read source Git; create patches and a non-Git replay tree only under .tmp."""
import ast
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = '5e72cc93d679acae2cb16b582f27bfe44886d2d0'
rows = json.loads((ROOT / '.tmp/commit_inventory.json').read_text())
def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])
assert git('rev-parse', 'HEAD').decode().strip() == BASE
baseline = {}
gitlinks = {}
for entry in git('ls-tree', '-rz', BASE).split(b'\0'):
    if not entry:
        continue
    meta, path = entry.split(b'\t', 1)
    mode, kind, oid = meta.decode().split()
    path = path.decode()
    if kind == 'blob':
        baseline[path] = (mode, git('cat-file', 'blob', oid))
    else:
        gitlinks[path] = oid
current = {}
for row in rows:
    path = ROOT / row['path']
    if path.is_file():
        mode = '100755' if path.stat().st_mode & 0o111 else '100644'
        current[row['path']] = (mode, path.read_bytes())
    else:
        current[row['path']] = None

by_name = {}
for path in current:
    by_name.setdefault(Path(path).name, []).append(path)
def resolve(name):
    if name in current:
        return name
    matches = by_name.get(name, [])
    assert len(matches) == 1, (name, matches)
    return matches[0]

steps = []
def add(title, names=(), overrides=None, note=''):
    changes = {resolve(n): current[resolve(n)] for n in names}
    changes.update(overrides or {})
    steps.append({'title': title, 'changes': changes, 'note': note})

add('chore: record local workspace ignore rules', ['.gitignore'])
add('build: move the container Dockerfile to the repository root', overrides={
    'Stage2to4/Dockerfile.codex': None,
    'Dockerfile.codex': baseline['Stage2to4/Dockerfile.codex'],
})
add('build: install Claude Code in the development container', ['Dockerfile.codex'])
moves = {}
for row in rows:
    if row['old_path']:
        moves[row['old_path']] = None
        moves[row['path']] = baseline[row['old_path']]
add('docs: consolidate existing documents under docs', overrides=moves,
    note='Content-preserving relocation; historical relative paths are retained until the documentation update.')
add('feat(stage4): convert and validate CVAT segmentation mask archives', [
    'convert_masks_to_cvat_segmentation_mask_1_1.py', 'check_stage4_cvat_segmentation_mask_export.py'])
add('feat(stage4): define contour refinement candidates and frozen configurations', [
    'contour_teacher_refinement.py', 'stage4_contour_auto_refine_phase3.yaml',
    'stage4_contour_auto_refine_phase3_production_v1.yaml'])
add('feat(stage4): add read-only contour refinement screening', [
    'prototype_stage4_contour_auto_refine.py', 'check_stage4_contour_auto_refine.py'])
add('feat(stage4): define manual review mask and label contracts', [
    'stage4_manual_review.py', 'stage4_manual_review_cvat_v1.yaml'])
add('feat(stage4): export review packages and import manual corrections', [
    'batch_export_stage4_manual_review_cvat.py', 'import_cvat_segmentation_mask_corrections.py',
    'batch_import_cvat_segmentation_mask_corrections.py'],
    note='The shared exporter already supports selected and full-video layouts; no historical implementation is fabricated.')
add('feat(stage4): validate full-video review inputs and transfer packages', [
    'preflight_stage4_phase5_fullvideo_cvat_review.py', 'validate_stage4_phase5_fullvideo_cvat_package.py',
    'preflight_stage4_phase5_fullvideo_cvat_review.sh', 'export_stage4_phase5_fullvideo_cvat_review.sh'])
add('feat(stage4): rebuild text-free review packages and verify round trips', [
    'rebuild_stage4_phase5_textfree_review_package.py', 'rebuild_stage4_phase5_fullvideo_textfree_review.sh',
    'check_stage4_cvat_manual_roundtrip.py'])
add('feat(stage4): apply accepted automatic refinements to teacher v3', [
    'batch_apply_stage4_contour_refinement.py', 'check_stage4_contour_teacher_phase5_apply.py',
    'build_stage4_bbox_ranked_v3_refined_auto.sh'])
add('feat(stage4): export all Phase 3 decisions for manual review', ['export_stage4_phase5_cvat_review_cases.sh'])
add('feat(stage4): track reversible video exclusions', [
    'build_stage4_exclusion_manifest.py', 'stage4_video_exclusions_v1.csv', 'check_stage4_sampling_sweep_manifest.py'])
add('feat(stage4): create and resume CVAT review tasks', [
    'batch_create_stage4_phase5_cvat_tasks.py', 'check_stage4_phase5_fullvideo_cvat_tasks.py',
    'create_stage4_phase5_cvat_tasks.sh', 'create_stage4_phase5_fullvideo_cvat_tasks.sh',
    'create_stage4_phase5_fullvideo_textfree_cvat_tasks.sh'])
add('feat(stage4): export verified CVAT task snapshots', [
    'batch_export_stage4_phase5_cvat_task_snapshots.py', 'check_stage4_phase5_cvat_task_snapshots.py',
    'export_stage4_phase5_fullvideo_cvat_task_snapshots.sh'])
add('feat(stage4): import full-video snapshots with versioned label authority', [
    'batch_import_stage4_phase5_fullvideo_cvat.py', 'check_stage4_phase5_fullvideo_final_import.py',
    'check_stage4_bbox_ranked_label_policy.py'],
    note='Both v4 target-only and v5 authoritative modes are introduced together with the shared reader and contracts.')
add('feat(stage4): add the teacher v4 build entry point', ['build_stage4_bbox_ranked_v4_manual_fullvideo.sh'])
add('feat(stage4): visualize saved point labels and frame provenance', [
    'export_stage4_point_label_visualization.py', 'batch_export_stage4_point_label_visualization.py',
    'check_stage4_point_label_visualization.py', 'export_stage4_v4_point_label_visualizations.sh'],
    note='The renderer includes v5/v6 schema readers; later commits add their pipelines and invalidation integration checks.')
add('feat(stage4): preflight authoritative CVAT labels without rewriting inputs', [
    'preflight_stage4_cvat_authoritative_labels.py', 'check_stage4_cvat_authoritative_preflight.py',
    'preflight_stage4_v5_cvat_authoritative_labels.sh'])
add('feat(stage4): build and accept teacher v5 authoritative labels', [
    'check_stage4_v5_cvat_authoritative_acceptance.py', 'build_stage4_bbox_ranked_v5_cvat_authoritative.sh',
    'export_stage4_v5_cvat_authoritative_point_label_visualizations.sh'])
add('feat(stage4): audit deleted XML annotations against saved teacher data', [
    'audit_stage4_deleted_xml_annotations.py', 'check_stage4_deleted_xml_annotation_audit.py',
    'audit_stage4_v5_deleted_xml_annotations.sh'])
add('feat(stage4): validate explicit frame invalidation manifests', [
    'build_stage4_deleted_xml_invalidation_manifest.py', 'check_stage4_deleted_xml_invalidation_manifest.py',
    'stage4_deleted_xml_invalidations_v1.csv', 'build_stage4_deleted_xml_invalidation_manifest.sh'])
add('feat(stage4): apply XML invalidations with auditable H5 provenance', [
    'apply_deleted_xml_invalidations.py', 'check_stage4_deleted_xml_invalidation_apply.py'])
add('feat(stage4): batch build and collect teacher v6', [
    'batch_apply_stage4_deleted_xml_invalidations.py', 'check_stage4_deleted_xml_invalidation_batch.py',
    'build_stage4_bbox_ranked_v6_xml_invalidation.sh'])
add('feat(stage4): verify invalidation overlays and export final teacher point clouds', [
    'check_stage4_deleted_xml_invalidation_visualization.py', 'export_stage4_v6_xml_invalidation_point_label_visualizations.sh',
    'export_stage4_bbox_ranked_pointcloud_visualizations.sh'])
add('docs(stage4): record teacher evolution and deferred contour revisions', [
    r['path'] for r in rows if r['path'].startswith('docs/stage2to4/') and
    current[r['path']] != moves.get(r['path'], baseline.get(r['path']))],
    note='Retrospective reports describe previously executed experiments, not tests rerun on each reconstructed commit.')
add('test(stage5): audit H5 dataset and batch integrity', [
    'check_stage5_batch_integrity.py', 'check_stage5_batch_integrity.sh'])
add('test(stage5): diagnose padding effects on predictions and gradients', [
    'check_stage5_padding_parity.py', 'check_stage5_padding_parity.sh'])
add('feat(stage5): expose weighted loss sums and normalization factors', ['losses.py'])

# Remove only normalization-option AST statements from the current training code.
# Gradient accumulation remains unchanged; the final normalization commit restores exact source bytes.
def without_norm_python(path):
    text = current[path][1].decode()
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    cuts = set()
    for node in ast.walk(tree):
        src = ast.get_source_segment(text, node) or ''
        if isinstance(node, ast.ImportFrom) and node.module == 'stage5.models.norm_layers':
            cuts.update(range(node.lineno - 1, node.end_lineno))
        if isinstance(node, ast.If) and 'args.pointnext_norm_groups' in ast.unparse(node.test):
            cuts.update(range(node.lineno - 1, node.end_lineno))
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and 'parser.add_argument' in src and 'pointnext_norm' in src:
            cuts.update(range(node.lineno - 1, node.end_lineno))
    for i, line in enumerate(lines):
        if '"pointnext_norm": args.' in line or '"pointnext_norm_groups": args.' in line:
            cuts.add(i)
    result = ''.join(line for i, line in enumerate(lines) if i not in cuts)
    assert 'pointnext_norm' not in result
    ast.parse(result)
    return (current[path][0], result.encode())

train = 'Stage5/train_stage5.py'
add('feat(stage5): accumulate point-weighted microbatch gradients', ['check_dummy_training.py'],
    overrides={train: without_norm_python(train)},
    note='Only GroupNorm-specific imports, model kwargs, validation and CLI statements are withheld.')

shell = current['Stage5/train_stage5.sh'][1].decode()
shell = re.sub(r'# Normalization used throughout.*?POINTNEXT_NORM_GROUPS=.*?\n', '', shell, flags=re.S)
shell = shell.replace(', norm=${POINTNEXT_NORM}, norm_groups=${POINTNEXT_NORM_GROUPS}', '')
shell = ''.join(l for l in shell.splitlines(keepends=True) if '--pointnext_norm' not in l)
assert 'POINTNEXT_NORM' not in shell
add('feat(stage5): use padding-free training and teacher v6 run paths', [
    'infer_stage5.sh', 'evaluate_stage5.sh', 'export_anonymized_stage5_metrics.sh'],
    overrides={'Stage5/train_stage5.sh': (current['Stage5/train_stage5.sh'][0], shell.encode())})
add('test(stage5): compare overlap probability aggregation methods', [
    'check_stage5_overlap_aggregation.py', 'check_stage5_overlap_aggregation.sh',
    'stage5_overlap_aggregation_implementation_policy.md'])
add('test(stage5): diagnose BatchNorm train and eval parity', [
    'check_stage5_batchnorm_mode_parity.py', 'check_stage5_batchnorm_mode_parity.sh'])
add('test(stage5): diagnose training-only BatchNorm recalibration', [
    'check_stage5_batchnorm_recalibration.py', 'check_stage5_batchnorm_recalibration.sh'])
add('feat(stage5): support GroupNorm throughout model and checkpoint workflows', [
    'norm_layers.py', 'pointnext_decoder_patch.py', 'pointnext_s_segmentor.py',
    'train_stage5.py', 'train_stage5.sh', 'infer_stage5.py', 'evaluate_stage5.py',
    'check_dummy_pointnext_s_training.py', 'check_dummy_pointnext_s_training.sh',
    'check_dummy_pointnext_s_groupnorm.py', 'check_dummy_pointnext_s_groupnorm.sh',
    'check_stage5_batchnorm_to_groupnorm_transfer.py', 'check_stage5_batchnorm_to_groupnorm_transfer.sh'],
    note='Decoder normalization, CLI/checkpoint propagation and tests stay together; the known incomplete decoder conversion is never reintroduced.')
add('docs(stage5): record diagnostics and refresh project navigation', [
    'docs/README.md', 'docs/stage5/FILES.md', 'stage5_overlap_aggregation_handoff_prompt.md',
    'stage5_pointnext_s_training_evaluation_report.md', 'stage5_revision_management_record.md'])

def oid(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
def file_patch(path, old, new):
    if old == new:
        return ''
    om, ob = old or ('000000', b'')
    nm, nb = new or ('000000', b'')
    out = [f'diff --git a/{path} b/{path}\n']
    if old is None:
        out.append(f'new file mode {nm}\n')
    elif new is None:
        out.append(f'deleted file mode {om}\n')
    elif om != nm:
        out.extend([f'old mode {om}\n', f'new mode {nm}\n'])
    out.append(f'index {oid(ob) if old else "0"*40}..{oid(nb) if new else "0"*40}\n')
    diff = difflib.unified_diff(ob.decode().splitlines(keepends=True), nb.decode().splitlines(keepends=True),
                               fromfile=f'a/{path}' if old else '/dev/null',
                               tofile=f'b/{path}' if new else '/dev/null')
    for line in diff:
        out.append(line)
        if not line.endswith('\n'):
            out.append('\n\\ No newline at end of file\n')
    return ''.join(out)

def descriptor(value):
    return None if value is None else {'mode': value[0], 'sha256': hashlib.sha256(value[1]).hexdigest()}

state = dict(baseline)
manifest = {'base': BASE, 'gitlinks': gitlinks, 'steps': [], 'expected_source': {}, 'expected_final': {}}
patch_dir = OUT / 'patches'
patch_dir.mkdir(exist_ok=True)
for num, step in enumerate(steps, 1):
    changes = {p: v for p, v in step['changes'].items() if state.get(p) != v}
    assert changes, step['title']
    patch = ''.join(file_patch(p, state.get(p), v) for p, v in sorted(changes.items()))
    path = patch_dir / f'{num:03d}.patch'
    path.write_text(patch)
    manifest['steps'].append({'number': num, 'title': step['title'], 'note': step['note'],
        'patch': f'patches/{path.name}', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'before': {p: descriptor(state.get(p)) for p in changes},
        'after': {p: descriptor(v) for p, v in changes.items()}})
    for p, v in changes.items():
        if v is None:
            state.pop(p, None)
        else:
            state[p] = v

expected = dict(baseline)
for row in rows:
    if row['old_path']:
        expected.pop(row['old_path'], None)
    v = current[row['path']]
    if v is None:
        expected.pop(row['path'], None)
    else:
        expected[row['path']] = v
assert state == expected, sorted(p for p in state.keys() | expected.keys() if state.get(p) != expected.get(p))
for path in baseline.keys() | expected.keys():
    manifest['expected_source'][path] = descriptor(expected.get(path))
    manifest['expected_final'][path] = descriptor(expected.get(path))
manifest['allowed_index'] = {}
for path in baseline.keys() | expected.keys():
    manifest['allowed_index'][path] = [descriptor(v) for v in (baseline.get(path), expected.get(path)) if v]
for row in rows:
    if row['old_path']:
        manifest['allowed_index'][row['path']].append(descriptor(baseline[row['old_path']]))
    if row['status'] == 'AM':
        data = git('show', ':' + row['path'])
        manifest['allowed_index'][row['path']].append(descriptor((current[row['path']][0], data)))
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
(OUT / 'series.txt').write_text(''.join(f"{s['number']:03d}\t{s['title']}\n" for s in manifest['steps']))
print(f'Built {len(steps)} patches; final bytes and modes match all source changes exactly.')

# Optional, explicitly additional cleanup after the exact 37-step reconstruction.
from email.parser import Parser
packages = []
site = ROOT / '.venvs/cvat273/lib/python3.11/site-packages'
for metadata in sorted(site.glob('*.dist-info/METADATA')):
    parsed = Parser().parsestr(metadata.read_text())
    packages.append((parsed['Name'], parsed['Version']))
assert any(n.lower().replace('_', '-') == 'cvat-sdk' and v == '2.73.0' for n, v in packages)
requirements = '# Observed local CVAT review environment; installation has not been replayed.\n'
requirements += '# Python 3.11.15; package snapshot, not a hash-locked cross-platform lockfile.\n'
requirements += ''.join(f'{name}=={version}\n' for name, version in sorted(packages, key=lambda v: v[0].lower()))
envdoc = '''# CVAT review environment snapshot

This snapshot records the local `.venvs/cvat273` environment used during review.
Python: 3.11.15. CVAT SDK: 2.73.0. The installed package versions are recorded in
`requirements-cvat-review.txt` at the repository root, including packaging tools.
This is an observed package snapshot, not a hash-locked or tested portable environment.
It does not provide the Stage 4/Stage 5 PyTorch, OpenCV, H5 or CUDA environment.

From the repository root, using an available Python 3.11 interpreter:

```bash
python3.11 -m venv .venvs/cvat273
.venvs/cvat273/bin/python -m pip install -r requirements-cvat-review.txt
```

These commands are for recreating a missing environment. Do not recreate an existing
environment as part of committing the repository. Installation and CVAT connectivity
must be verified on the target system; no installation was run during patch preparation.
The `.venvs/` directory remains local and is excluded from version control.
'''
ignore = current['.gitignore'][1]
optional = {
    '.gitignore': (current['.gitignore'][0], ignore + (b'' if ignore.endswith(b'\n') else b'\n') + b'.venvs/\n'),
    'requirements-cvat-review.txt': ('100644', requirements.encode()),
    'docs/development/cvat_environment.md': ('100644', envdoc.encode()),
}
optional_patch = ''.join(file_patch(p, state.get(p), v) for p, v in sorted(optional.items()))
(OUT / 'optional-environment.patch').write_text(optional_patch)
(OUT / 'optional-environment.json').write_text(json.dumps({
    'title': 'chore: exclude local virtual environments and record CVAT dependencies',
    'sha256': hashlib.sha256(optional_patch.encode()).hexdigest(),
    'before': {p: descriptor(state.get(p)) for p in optional},
    'after': {p: descriptor(v) for p, v in optional.items()},
}, indent=2) + '\n')
print(f'Optional cleanup patch adds .venvs ignore and an observed {len(packages)}-package snapshot.')
