"""Offline checks for the direct-source S5000 delivery; no GPU or data download."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]

class SourceDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((HERE / "SOURCE_MANIFEST.json").read_text())

    def test_pinned_source_hashes(self):
        expected = self.manifest["expected_source_sha256"]
        self.assertEqual(len(expected), 136)
        for name, digest in expected.items():
            with self.subTest(path=name):
                path = ROOT / name
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_retired_sources_absent(self):
        for name in self.manifest["expected_absent_source_paths"]:
            with self.subTest(path=name):
                self.assertFalse((ROOT / name).exists())

    def test_one_source_commit_and_report_per_optimization(self):
        items = self.manifest["optimizations"]
        self.assertEqual([x["id"] for x in items], list(range(1, 14)))
        self.assertEqual(len({x["commit"] for x in items}), 13)
        for item in items:
            self.assertTrue((ROOT / item["report"]).is_file())
            for name in item["changed_source_paths"]:
                self.assertTrue((ROOT / name).is_file(), name)

    def test_python_and_shell_syntax(self):
        paths = [ROOT / p for p in self.manifest["expected_source_sha256"]]
        paths.extend(p for p in HERE.rglob("*") if p.is_file())
        for path in paths:
            with self.subTest(path=str(path)):
                if path.suffix == ".py":
                    ast.parse(path.read_text(), filename=str(path))
                elif path.suffix == ".sh":
                    subprocess.run(["bash", "-n", str(path)], check=True)

    def test_reproduction_assets_unchanged(self):
        for name, digest in self.manifest["reproduction_asset_sha256"].items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_direct_source_delivery(self):
        self.assertEqual(self.manifest["delivery_format"], "direct_source")
        self.assertEqual(self.manifest["optimization_count"], 13)
        self.assertFalse((HERE / "patches").exists())
        self.assertFalse((HERE / "apply_patches.py").exists())
        self.assertFalse((HERE / "adapters").exists())
        self.assertFalse(list(HERE.rglob("*.patch")))

    def test_native_dependency_scope(self):
        self.assertEqual(len(self.manifest["required_native_dependencies"]), 2)
        self.assertEqual(len(list(HERE.glob("reports/[0-9][0-9].md"))), 14)


if __name__ == "__main__":
    unittest.main()
