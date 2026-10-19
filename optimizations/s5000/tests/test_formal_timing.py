import importlib.util
from pathlib import Path
import unittest
PATH = Path(__file__).resolve().parents[1] / "summarize.py"
spec = importlib.util.spec_from_file_location("formal_flow_checks", PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class FormalFlowTest(unittest.TestCase):
    def test_warm_and_post_success_cleanup_are_excluded(self):
        log = "\n".join([
            "2026-10-19 10:00:00,000 INFO Featurizing a_warm...",
            "2026-10-19 10:00:10,000 INFO [Rank 0] a_warm [seed:101] succeeded.",
            "2026-10-19 10:00:12,000 INFO Featurizing a...",
            "2026-10-19 10:00:15,250 INFO [Rank 0] a [seed:101] succeeded.",
            "2026-10-19 10:01:00,000 INFO Job completed in 60.00s.",
        ])
        self.assertEqual(mod.formal_flow_durations(log, ["a"]), {"a": 3.25})

    def test_missing_start_is_rejected(self):
        with self.assertRaises(ValueError):
            mod.formal_flow_durations("2026-10-19 10:00:15,250 INFO [Rank 0] a [seed:101] succeeded.", ["a"])

    def test_missing_success_is_rejected(self):
        with self.assertRaises(ValueError):
            mod.formal_flow_durations("2026-10-19 10:00:12,000 INFO Featurizing a...", ["a"])

if __name__ == "__main__":
    unittest.main()
