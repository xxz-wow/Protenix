"""Check mask cache invalidation without importing the full model."""
import importlib.util
from pathlib import Path
import unittest
import torch

PATH = Path(__file__).resolve().parents[3] / "protenix/model/musa_kernels/runtime.py"
spec = importlib.util.spec_from_file_location("mask_runtime_checks", PATH)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)

class MaskCacheTest(unittest.TestCase):
    def setUp(self):
        runtime.STATS.clear()
        runtime._MASK_RESULTS.clear()

    def test_none_and_mask_with_a_zero(self):
        self.assertTrue(runtime.mask_all_ones(None))
        self.assertTrue(runtime.mask_all_ones(torch.ones(2, 3)))
        self.assertFalse(runtime.mask_all_ones(torch.tensor([1., 0.])))

    def test_inplace_edit_invalidates_cached_result(self):
        mask = torch.ones(2, 3)
        self.assertTrue(runtime.mask_all_ones(mask))
        self.assertTrue(runtime.mask_all_ones(mask))
        self.assertEqual(runtime.STATS["mask_cache_hits"], 1)
        mask[:, 0].zero_()
        self.assertFalse(runtime.mask_all_ones(mask))
        self.assertEqual(runtime.STATS["mask_cache_misses"], 2)

    def test_bias_edit_and_different_tensor_do_not_reuse_true(self):
        bias = torch.zeros(2, 3)
        self.assertTrue(runtime.bias_all_zeros(bias))
        bias[0, 0] = -10000
        self.assertFalse(runtime.bias_all_zeros(bias))
        self.assertFalse(runtime.bias_all_zeros(torch.ones(2, 3)))

    def test_inference_tensor_without_version_is_not_cached(self):
        with torch.inference_mode():
            mask = torch.ones(2)
            self.assertTrue(runtime.mask_all_ones(mask))
            mask[0] = 0
            self.assertFalse(runtime.mask_all_ones(mask))
        self.assertEqual(runtime.STATS["mask_cache_uncacheable"], 2)

if __name__ == "__main__":
    unittest.main()
