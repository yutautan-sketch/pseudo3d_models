import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('user_commit', Path(__file__).with_name('user_commit.py'))
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)

class GuardTests(unittest.TestCase):
    def test_content_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'file').write_bytes(b'new user edit')
            with self.assertRaisesRegex(SystemExit, 'content mismatch'):
                u.check_files(root, {'file': {'sha256': u.digest(b'old'), 'mode': '100644'}})

    def test_expected_deletion_does_not_remove_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'file').write_bytes(b'keep')
            with self.assertRaisesRegex(SystemExit, 'Expected absent'):
                u.check_files(root, {'file': None})
            self.assertEqual((root / 'file').read_bytes(), b'keep')

    def test_patch_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '001.patch').write_bytes(b'tampered')
            with patch.object(u, 'BUNDLE', root), patch.object(u, 'M', {'steps': [
                {'patch': '001.patch', 'sha256': u.digest(b'original')}]}):
                with self.assertRaisesRegex(SystemExit, 'checksum mismatch'):
                    u.check_bundle()

    def test_wrong_head_never_mutates_git(self):
        with patch.object(u, 'check_bundle'), patch.object(u, 'head', return_value='other'), patch.object(u, 'execute') as mutate:
            with self.assertRaisesRegex(SystemExit, 'Source HEAD'):
                u.source_preflight(Path('/dummy'))
            mutate.assert_not_called()

    def test_uncaptured_index_content_rejected(self):
        def read(repo, *args):
            if args == ('ls-files', '--others', '--exclude-standard', '-z'):
                return b''
            if args == ('ls-files', '-z'):
                return b'.gitignore\0'
            if args == ('ls-files', '--stage', '-z'):
                return b'100644 1111111111111111111111111111111111111111 0\t.gitignore\0'
            if args[:2] == ('cat-file', 'blob'):
                return b'unrecorded staged changes'
            self.fail(repr(args))
        with patch.object(u, 'check_bundle'), patch.object(u, 'head', return_value=u.M['base']), \
             patch.object(u, 'branch', return_value='main'), patch.object(u, 'check_no_operation'), \
             patch.object(u, 'check_files'), patch.object(u, 'git', side_effect=read), patch.object(u, 'execute') as mutate:
            with self.assertRaisesRegex(SystemExit, 'uncaptured content'):
                u.source_preflight(Path('/dummy'))
            mutate.assert_not_called()

    def test_history_rejects_different_branch(self):
        with patch.object(u, 'branch', return_value='main'), patch.object(u, 'execute') as mutate:
            with self.assertRaisesRegex(SystemExit, 'dedicated split branch'):
                u.history(Path('/dummy'))
            mutate.assert_not_called()

if __name__ == '__main__':
    unittest.main()
