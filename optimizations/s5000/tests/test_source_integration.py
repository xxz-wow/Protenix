"""Guard the direct model call graph against accidental wrapper reintroduction."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]


class DirectSourceTest(unittest.TestCase):
    def test_model_and_entry_do_not_rebind_functions(self):
        paths = list((ROOT / 'protenix/model').rglob('*.py'))
        paths += [ROOT / 'optimizations/s5000/entry.py']
        for path in paths:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Attribute):
                            self.assertNotIn(target.attr, {'forward', '_tri_attention', '_attention', '_local_attention'}, str(path))
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [node.module or ''] if isinstance(node, ast.ImportFrom) else [n.name for n in node.names]
                    for name in names:
                        self.assertFalse(name.startswith('optimizations.'), (path, name))

    def test_cache_is_an_explicit_argument(self):
        path = ROOT / 'protenix/model/modules/transformer.py'
        tree = ast.parse(path.read_text())
        expected = {'AttentionPairBias': {'forward', 'standard_multihead_attention'},
                    'DiffusionTransformer': {'forward', '_prep_blocks'},
                    'DiffusionTransformerBlock': {'forward'}}
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef) and cls.name in expected:
                for fn in cls.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name in expected[cls.name]:
                        self.assertIn('pair_bias_cache', [a.arg for a in fn.args.args])

    def test_launcher_needs_no_adapter_search_path(self):
        source = (ROOT / 'optimizations/s5000/launch.py').read_text()
        self.assertNotIn('"adapters"', source)
        self.assertNotIn('"compat"', source)
        self.assertIn('str(REPO / "runner/inference.py")', source)
        for path in ['adapters', 'compat', 'quick_controls.py', 'original_primitives.py']:
            self.assertFalse((ROOT / 'optimizations/s5000' / path).exists())


if __name__ == '__main__':
    unittest.main()
