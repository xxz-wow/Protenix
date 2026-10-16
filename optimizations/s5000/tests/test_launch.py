import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "launch.py"
spec = importlib.util.spec_from_file_location("s5000_launch", path)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class WarmProtocolTest(unittest.TestCase):
    def setUp(self):
        self.inputs = [
            {"name": "case_a", "sequences": [{"proteinChain": {"sequence": "ACD", "count": 1}}]},
            {"name": "case_b", "sequences": [{"proteinChain": {"sequence": "EFG", "count": 2}}]},
        ]

    def test_formal_inputs_expand_without_mutation(self):
        original = copy.deepcopy(self.inputs)
        result = launch.pair_inputs(self.inputs)
        self.assertEqual([x["name"] for x in result],
                         ["case_a_warm", "case_a", "case_b_warm", "case_b"])
        result[0]["sequences"][0]["proteinChain"]["sequence"] = "changed"
        self.assertEqual(self.inputs, original)
        self.assertEqual(result[1]["sequences"], original[0]["sequences"])

    def test_paired_inputs_do_not_gain_extra_warm_round(self):
        paired = launch.pair_inputs(self.inputs)
        self.assertEqual(launch.pair_inputs(paired), paired)

    def test_mismatched_warm_biology_is_rejected(self):
        paired = launch.pair_inputs(self.inputs)
        paired[0]["sequences"][0]["proteinChain"]["sequence"] = "wrong"
        with self.assertRaises(ValueError):
            launch.pair_inputs(paired)

    def test_duplicate_names_and_invalid_order_are_rejected(self):
        with self.assertRaises(ValueError):
            launch.pair_inputs([self.inputs[0], self.inputs[0]])
        paired = launch.pair_inputs(self.inputs)
        with self.assertRaises(ValueError):
            launch.pair_inputs([paired[1], paired[0]])

    def test_unsafe_names_are_rejected(self):
        for name in ["../outside", "/outside", ".", "..", r"..\outside"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                launch.pair_inputs([{"name": name, "sequences": []}])


if __name__ == "__main__":
    unittest.main()
