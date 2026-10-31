#!/usr/bin/env python3
"""Stdlib self-test for compare_step_metrics.py (no training logs required).

Run: python .claude/skills/align-automodel-xtuner-text/scripts/test_compare_step_metrics.py
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("compare_step_metrics.py")
spec = importlib.util.spec_from_file_location("compare_step_metrics", MODULE_PATH)
csm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = csm
spec.loader.exec_module(csm)

AM_REGEX = r"rank=(?P<rank>\d+) step=(?P<step>\d+) loss=(?P<loss>[\d.eE+-]+) grad_norm=(?P<grad_norm>[\d.eE+-]+)"
XT_REGEX = r"Step (?P<step>\d+): loss=(?P<loss>[\d.eE+-]+) grad_norm=(?P<grad_norm>[\d.eE+-]+)"


class CompareStepMetricsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text)
        return str(path)

    def base_args(self, a_log, b_log, metrics="loss,grad_norm"):
        return [
            "--a-log", a_log, "--a-regex", AM_REGEX, "--a-rank", "0", "--a-offset", "1",
            "--b-log", b_log, "--b-regex", XT_REGEX,
            "--metrics", metrics, "--out-prefix", str(self.dir / "pair"),
        ]

    def test_carriage_returns_offsets_and_diffs(self):
        a = self.write("am.log", "rank=0 step=0 loss=10.0 grad_norm=25.0\rrank=0 step=1 loss=9.5 grad_norm=24.0\n"
                                 "rank=1 step=0 loss=99.0 grad_norm=99.0\n")
        b = self.write("xt.log", "Step 1: loss=10.1 grad_norm=25.5\nStep 2: loss=9.4 grad_norm=24.2\n")
        self.assertEqual(csm.main(self.base_args(a, b)), 0)
        summary = json.loads((self.dir / "pair.json").read_text())
        self.assertEqual(summary["paired_updates"], 2)
        self.assertAlmostEqual(summary["metrics"]["loss"]["mean_diff"], 0.0)  # -0.1 and +0.1 cancel
        self.assertAlmostEqual(summary["metrics"]["loss"]["mean_abs_diff"], 0.1)
        self.assertEqual(summary["metrics"]["grad_norm"]["max_abs_diff_update"], 1)
        self.assertAlmostEqual(summary["metrics"]["grad_norm"]["max_abs_diff"], 0.5)
        rows = list(csm.csv.DictReader((self.dir / "pair.csv").open()))
        self.assertEqual([r["update"] for r in rows], ["1", "2"])
        self.assertAlmostEqual(float(rows[0]["b_loss"]), 10.1)  # rank=1 row must not leak in

    def test_threshold_failure_reports_first_update(self):
        a = self.write("am.log", "rank=0 step=0 loss=10.0 grad_norm=25.0\nrank=0 step=1 loss=9.0 grad_norm=24.0\n")
        b = self.write("xt.log", "Step 1: loss=10.0 grad_norm=25.5\nStep 2: loss=9.0 grad_norm=24.2\n")
        args = self.base_args(a, b, metrics="loss") + ["--atol", "0.05"]
        self.assertEqual(csm.main(args), 0)
        args = self.base_args(a, b) + ["--atol", "0.1", "--rtol", "0.0"]
        self.assertEqual(csm.main(args), 1)
        summary = json.loads((self.dir / "pair.json").read_text())
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["metrics"]["grad_norm"]["criterion"]["first_failure"]["update"], 1)

    def test_duplicate_steps_last_wins_and_error_mode(self):
        a = self.write("am.log", "rank=0 step=0 loss=10.0 grad_norm=25.0\nrank=0 step=0 loss=11.0 grad_norm=25.0\n")
        b = self.write("xt.log", "Step 1: loss=11.0 grad_norm=25.0\n")
        self.assertEqual(csm.main(self.base_args(a, b, metrics="loss")), 0)
        summary = json.loads((self.dir / "pair.json").read_text())
        self.assertEqual(summary["a"]["duplicates"], 1)
        self.assertAlmostEqual(summary["metrics"]["loss"]["max_abs_diff"], 0.0)
        with self.assertRaises(ValueError):
            csm.main(self.base_args(a, b, metrics="loss") + ["--on-duplicate", "error"])

    def test_rank_group_requires_explicit_selection(self):
        a = self.write("am.log", "rank=0 step=0 loss=10.0 grad_norm=25.0\n")
        b = self.write("xt.log", "Step 1: loss=10.0 grad_norm=25.0\n")
        args = [
            "--a-log", a, "--a-regex", AM_REGEX, "--a-offset", "1",
            "--b-log", b, "--b-regex", XT_REGEX,
            "--metrics", "loss", "--out-prefix", str(self.dir / "pair"),
        ]
        with self.assertRaises(ValueError):
            csm.main(args)

    def test_jsonl_marker_and_dotted_keys(self):
        rows = [
            {"step": 0, "eval": {"acc": 0.4}},
            {"step": 1, "train": {"loss": 10.0, "grad_norm": 25.0}},
            {"step": 2, "train": {"loss": 9.0, "grad_norm": 24.0}},
        ]
        a = self.write("a.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
        b = self.write("xt.log", "Step 1: loss=10.2 grad_norm=25.0\nStep 2: loss=9.1 grad_norm=24.0\n")
        args = [
            "--a-log", a, "--a-format", "jsonl", "--a-marker", "train.loss",
            "--a-map", "loss=train.loss", "--a-map", "grad_norm=train.grad_norm",
            "--b-log", b, "--b-regex", XT_REGEX,
            "--metrics", "loss,grad_norm", "--out-prefix", str(self.dir / "jsonl_pair"),
        ]
        self.assertEqual(csm.main(args), 0)
        summary = json.loads((self.dir / "jsonl_pair.json").read_text())
        self.assertEqual(summary["paired_updates"], 2)  # eval-only row excluded
        self.assertAlmostEqual(summary["metrics"]["loss"]["max_abs_diff"], 0.2)

    def test_no_shared_updates_is_an_error(self):
        a = self.write("am.log", "rank=0 step=5 loss=10.0 grad_norm=25.0\n")
        b = self.write("xt.log", "Step 1: loss=10.0 grad_norm=25.0\n")
        with self.assertRaises(ValueError):
            csm.main(self.base_args(a, b, metrics="loss"))

    def test_unmatched_updates_are_reported(self):
        a = self.write("am.log", "rank=0 step=0 loss=10.0 grad_norm=25.0\nrank=0 step=1 loss=9.0 grad_norm=24.0\n")
        b = self.write("xt.log", "Step 1: loss=10.0 grad_norm=25.0\n")
        self.assertEqual(csm.main(self.base_args(a, b, metrics="loss")), 0)
        summary = json.loads((self.dir / "pair.json").read_text())
        self.assertEqual(summary["a_only_updates"], [2])
        self.assertEqual(summary["b_only_updates"], [])


if __name__ == "__main__":
    unittest.main()
