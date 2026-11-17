"""Tests for scripts/fetch_e768.sh against a local fake release (file:// URLs; the real weights and the checkout are never touched)."""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/fetch_e768.sh'
ONNX, MEMBERS = b'fake onnx graph', {'s2026': b'fake member A', 's17': b'fake member B'}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@unittest.skipUnless(shutil.which('bash') and shutil.which('curl') and shutil.which('sha256sum'), 'needs bash, curl and sha256sum')
class FetchE768Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.repo, self.release = base / 'repo', base / 'release'
        (self.repo / 'scripts').mkdir(parents=True)
        shutil.copy(SCRIPT, self.repo / 'scripts/fetch_e768.sh')
        self.variant = self.repo / 'matlab/variant_R3_E_768'
        self.variant.mkdir(parents=True)
        (self.variant / 'netrai_bundle.json').write_text(json.dumps({'artifacts': [{'role': 'model', 'sha256': sha(ONNX)}]}))
        (self.variant / 'torch_members.json').write_text(json.dumps({'members': {
            k: {'file': f'member_{k}.pt', 'sha256': sha(v)} for k, v in MEMBERS.items()}}))
        self.release.mkdir()
        (self.release / 'netrai_E_768.onnx').write_bytes(ONNX)
        for k, v in MEMBERS.items():
            (self.release / f'member_{k}.pt').write_bytes(v)

    def run_script(self, *args, onnx_url=None):
        env = dict(os.environ, E768_URL=onnx_url or (self.release / 'netrai_E_768.onnx').as_uri(), E768_MEMBER_URL_BASE=self.release.as_uri())
        return subprocess.run(['bash', str(self.repo / 'scripts/fetch_e768.sh'), *args], env=env, capture_output=True, text=True, timeout=120)

    def members_installed(self):
        return all((self.variant / 'torch' / f'member_{k}.pt').is_file() for k in MEMBERS)

    def test_fresh_install_with_gradcam(self):
        result = self.run_script('--with-gradcam')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.variant / 'netrai_fold0.onnx').read_bytes(), ONNX)
        self.assertTrue(self.members_installed())

    def test_installed_onnx_still_gets_the_members(self):
        self.assertEqual(self.run_script().returncode, 0)                         # an earlier install without --with-gradcam
        self.assertFalse(self.members_installed())
        result = self.run_script('--with-gradcam', onnx_url=(self.release / 'absent.onnx').as_uri())   # the ONNX is not downloaded again
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('already present and verified', result.stdout)
        self.assertTrue(self.members_installed())
        again = self.run_script('--with-gradcam', onnx_url=(self.release / 'absent.onnx').as_uri())
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(again.stdout.count('already verified: member_'), 2)

    def test_tampered_files_are_not_installed(self):
        (self.release / 'member_s17.pt').write_bytes(b'tampered')
        result = self.run_script('--with-gradcam')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA-256 mismatch', result.stderr)
        self.assertFalse((self.variant / 'torch/member_s17.pt').exists())
        self.assertEqual(list((self.variant / 'torch').glob('*.part')), [])
        (self.variant / 'netrai_fold0.onnx').unlink()
        (self.release / 'netrai_E_768.onnx').write_bytes(b'tampered graph')
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertFalse((self.variant / 'netrai_fold0.onnx').exists())


if __name__ == '__main__':
    unittest.main()
