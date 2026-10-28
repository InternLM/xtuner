#!/usr/bin/env python3
"""Stdlib self-test for compare_step_metrics.py (no training logs required).

Run: python .claude/skills/align-automodel-xtuner-text/scripts/test_compare_step_metrics.py
"""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
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

    def run_main(self, args: list[str]) -> int:
        with redirect_stdout(io.StringIO()):
            return csm.main(args)

    def json_args(self, a_rows: list[dict], b_rows: list[dict], metrics: str = "loss") -> list[str]:
        args = ["--metrics", metrics, "--out-prefix", str(self.dir / "pair")]
        for side, rows in (("a", a_rows), ("b", b_rows)):
            path = self.write(f"{side}.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
            args += [
                f"--{side}-log", path, f"--{side}-format", "jsonl",
                f"--{side}-map", "loss=loss", f"--{side}-map", "grad_norm=grad_norm",
            ]
        return args

    def run_cli(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(MODULE_PATH), *args], capture_output=True, text=True, timeout=15,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def assert_cli_error(self, args: list[str], message: str) -> None:
        for suffix in (".json", ".csv"):
            (self.dir / f"pair{suffix}").unlink(missing_ok=True)
        result = self.run_cli(args)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(message, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn('"passed": true', result.stdout)
        self.assertFalse((self.dir / "pair.json").exists())
        self.assertFalse((self.dir / "pair.csv").exists())

    def test_carriage_returns_offsets_and_diffs(self):
        a = self.write("am.log", "rank=0 step=0 loss=10.0 grad_norm=25.0\rrank=0 step=1 loss=9.5 grad_norm=24.0\n"
                                 "rank=1 step=0 loss=99.0 grad_norm=99.0\n")
        b = self.write("xt.log", "Step 1: loss=10.1 grad_norm=25.5\nStep 2: loss=9.4 grad_norm=24.2\n")
        self.assertEqual(self.run_main(self.base_args(a, b)), 0)
        summary = json.loads((self.dir / "pair.json").read_text())
        self.assertEqual(summary["paired_updates"], 2)
        self.assertAlmostEqual(summary["metrics"]["loss"]["mean_diff"], 0.0)  # -0.1 and +0.1 cancel
        self.assertAlmostEqual(summary["metrics"]["loss"]["mean_abs_diff"], 0.1)
        self.assertEqual(summary["metrics"]["grad_norm"]["max_abs_diff_update"], 1)
        self.assertAlmostEqual(summary["metrics"]["grad_norm"]["max_abs_diff"], 0.5)
        self.assertNotIn("passed", summary)
        with (self.dir / "pair.csv").open(newline="") as handle:
            rows = list(csm.csv.DictReader(handle))
        self.assertEqual([r["update"] for r in rows], ["1", "2"])
        self.assertAlmostEqual(float(rows[0]["b_loss"]), 10.1)  # rank=1 row must not leak in

    def test_threshold_failure_reports_first_update(self):
        a = self.write("am.log", "rank=0 step=0 loss=10.0 grad_norm=25.0\nrank=0 step=1 loss=9.0 grad_norm=24.0\n")
        b = self.write("xt.log", "Step 1: loss=10.0 grad_norm=25.5\nStep 2: loss=9.0 grad_norm=24.2\n")
        args = self.base_args(a, b, metrics="loss") + ["--expected-updates", "1:2", "--atol", "0.05"]
        self.assertEqual(self.run_main(args), 0)
        args = self.base_args(a, b) + ["--expected-updates", "1:2", "--atol", "0.1", "--rtol", "0.0"]
        self.assertEqual(self.run_main(args), 1)
        summary = json.loads((self.dir / "pair.json").read_text())
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["metrics"]["grad_norm"]["criterion"]["first_failure"]["update"], 1)

    def test_duplicate_strategies_text_and_jsonl(self):
        a = self.write("am.log", "rank=0 step=0 loss=10.0 grad_norm=25.0\nrank=0 step=0 loss=11.0 grad_norm=25.0\n")
        b = self.write("xt.log", "Step 1: loss=11.0 grad_norm=25.0\n")
        for args in (
            self.base_args(a, b, metrics="loss"),
            self.json_args([{"step": 1, "loss": 10}, {"step": 1, "loss": 11}], [{"step": 1, "loss": 11}]),
        ):
            self.assert_cli_error(args, "duplicate logical update 1")
            self.assert_cli_error(args + ["--on-duplicate", "error"], "duplicate logical update 1")
            for strategy, diff, status in (("first", 1.0, 1), ("last", 0.0, 0)):
                with self.subTest(strategy=strategy, args=args):
                    self.assertEqual(self.run_main(args + [
                        "--on-duplicate", strategy, "--expected-updates", "1:1", "--atol", "0",
                    ]), status)
                    summary = json.loads((self.dir / "pair.json").read_text())
                    self.assertEqual(summary["a"]["duplicates"], 1)
                    self.assertEqual(summary["a"]["matched_lines"], 2)
                    self.assertEqual(summary["on_duplicate"], strategy)
                    self.assertAlmostEqual(summary["metrics"]["loss"]["max_abs_diff"], diff)

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
        self.assertEqual(self.run_main(args), 0)
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
        self.assertEqual(self.run_main(self.base_args(a, b, metrics="loss")), 0)
        summary = json.loads((self.dir / "pair.json").read_text())
        self.assertEqual(summary["a_only_updates"], [2])
        self.assertEqual(summary["b_only_updates"], [])

    def test_cli_exit_codes_and_scalar_scope(self):
        args = self.json_args([{"step": 0, "loss": 11}], [{"step": 0, "loss": 10}])
        for flags, status, passed in (
            ([], 0, None),
            (["--expected-updates", "0:0"], 0, None),
            (["--expected-updates", "0:0", "--rtol", "0.1"], 0, True),
            (["--expected-updates", "0:0", "--rtol", "0"], 1, False),
        ):
            with self.subTest(flags=flags):
                result = self.run_cli(args + flags)
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertEqual(result.stderr, "")
                self.assertIn("scalar metric criteria do not establish overall training alignment", result.stdout)
                summary = json.loads((self.dir / "pair.json").read_text())
                if passed is None:
                    self.assertNotIn("passed", summary)
                    self.assertIn("no pass/fail verdict", result.stdout)
                else:
                    self.assertIs(summary["passed"], passed)
                    self.assertEqual(summary["metrics"]["loss"]["criterion"]["atol"], 0.0)
                self.assertTrue((self.dir / "pair.csv").exists())
        equal = self.json_args([{"step": 0, "loss": 10}], [{"step": 0, "loss": 10}])
        self.assertEqual(self.run_cli(equal + ["--expected-updates", "0:0", "--rtol", "0"]).returncode, 0)

    def test_thresholds_require_declared_window_even_when_zero(self):
        args = self.json_args([{"step": 0, "loss": 1}], [{"step": 0, "loss": 1}])
        for flag in ("--atol", "--rtol"):
            for value in ("0", "0.1"):
                with self.subTest(flag=flag, value=value):
                    self.assert_cli_error(args + [flag, value], "--expected-updates is required")

    def test_invalid_windows_thresholds_and_metrics(self):
        args = self.json_args([{"step": 0, "loss": 1}], [{"step": 0, "loss": 1}])
        for window in ("", "2:1", "0", ":", "0:", ":1", "0:1:2", "0.5:1", "zero:1"):
            with self.subTest(window=window):
                self.assert_cli_error(args + ["--expected-updates", window], "--expected-updates")
        for flag in ("--atol", "--rtol"):
            for value in ("-1", "nan", "inf", "-inf"):
                with self.subTest(flag=flag, value=value):
                    self.assert_cli_error(
                        args + ["--expected-updates", "0:0", f"{flag}={value}"], "finite and non-negative"
                    )
        for metrics in ("", " ", ",", "loss,", "loss,,grad_norm", "loss, loss"):
            with self.subTest(metrics=metrics):
                self.assert_cli_error(args + ["--metrics", metrics], "--metrics")

    def test_window_coverage_catches_single_and_shared_gaps(self):
        full = [{"step": step, "loss": 1} for step in (0, 1, 2)]
        gaps = [full[0], full[2]]
        for a, b, missing_sides in ((gaps, full, ("A",)), (full, gaps, ("B",)), (gaps, gaps, ("A", "B"))):
            for flags in ([], ["--atol", "0"]):
                with self.subTest(missing_sides=missing_sides, flags=flags):
                    args = self.json_args(a, b) + ["--expected-updates", "0:2"] + flags
                    self.assert_cli_error(args, "first: 1")
                    result = self.run_cli(args)
                    for side in missing_sides:
                        self.assertIn(f"{side}: missing 1 expected updates", result.stderr)
        for a, b in (([], []), ([], full), (full, [])):
            self.assert_cli_error(self.json_args(a, b) + ["--expected-updates", "0:2"], "no records parsed")
        self.assert_cli_error(self.json_args(full, full) + ["--expected-updates", "3:4"], "missing 2")
        self.assert_cli_error(self.json_args(full, full) + ["--expected-updates", "0:3"], "first: 3")

    def test_window_uses_offsets_and_excludes_outside_metrics(self):
        args = self.json_args(
            [{"step": -1, "loss": float("nan")}, {"step": 0, "loss": 2}, {"step": 1}],
            [{"step": 9}, {"step": 10, "loss": 2}, {"step": 11, "loss": float("inf")}],
        ) + ["--a-offset", "10", "--expected-updates", "10:10", "--atol", "0"]
        self.assertEqual(self.run_main(args), 0)
        summary = json.loads((self.dir / "pair.json").read_text())
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["expected_updates"], {"start": 10, "end": 10})
        self.assertEqual(summary["paired_updates"], 1)
        for side in ("a", "b"):
            self.assertEqual(summary[side]["records"], 3)
            self.assertEqual(summary[side]["compared_records"], 1)
            self.assertEqual(summary[side]["outside_window_records"], 2)
        # Offsets are not changed to manufacture coverage.
        self.assert_cli_error(args + ["--a-offset", "0"], "A: missing 1 expected updates")

    def test_nonfinite_metrics_rejected_on_either_or_both_sides(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for av, bv in ((value, 1.0), (1.0, value), (value, value)):
                for fmt in ("jsonl", "text"):
                    if fmt == "jsonl":
                        args = self.json_args([{"step": 0, "loss": av}], [{"step": 0, "loss": bv}])
                    else:
                        args = ["--metrics", "loss", "--out-prefix", str(self.dir / "pair")]
                        for side, metric_value in (("a", av), ("b", bv)):
                            path = self.write(f"{side}.log", f"step=0 loss={metric_value}\n")
                            args += [f"--{side}-log", path, f"--{side}-regex", r"step=(?P<step>\S+) loss=(?P<loss>\S+)"]
                    for flags in ([], ["--expected-updates", "0:0"], ["--expected-updates", "0:0", "--atol", "0"]):
                        with self.subTest(value=value, av=av, bv=bv, fmt=fmt, flags=flags):
                            self.assert_cli_error(args + flags, "non-finite")
        args = self.json_args([{"step": 0, "loss": 1}, {"step": 1, "loss": float("inf")}], [{"step": 0, "loss": 1}])
        self.assert_cli_error(args, "update 1, metric loss")

    def test_missing_metrics_descriptive_only(self):
        for a, b in (({}, {}), ({}, {"loss": 1}), ({"loss": 1}, {}), ({"loss": "bad"}, {"loss": 1})):
            args = self.json_args([{"step": 0, **a}], [{"step": 0, **b}])
            for flags in ([], ["--expected-updates", "0:0"]):
                self.assertEqual(self.run_main(args + flags), 0)
                summary = json.loads((self.dir / "pair.json").read_text())
                self.assertNotIn("passed", summary)
                self.assertEqual(summary["metrics"]["loss"]["paired"], 0)
                self.assertIsNone(summary["metrics"]["loss"]["mean_diff"])
                self.assertNotIn("NaN", (self.dir / "pair.json").read_text())
                self.assertNotIn("Infinity", (self.dir / "pair.json").read_text())
            self.assert_cli_error(args + ["--expected-updates", "0:0", "--atol", "0"], "missing")

    def test_sparse_descriptive_csv_has_requested_schema(self):
        args = self.json_args(
            [{"step": 0, "loss": 1}, {"step": 1, "loss": 2, "grad_norm": 3}],
            [{"step": 0}, {"step": 1, "loss": 2, "grad_norm": 4}], metrics="grad_norm,loss",
        )
        self.assertEqual(self.run_main(args), 0)
        fields = ["update", "a_grad_norm", "b_grad_norm", "diff_grad_norm", "a_loss", "b_loss", "diff_loss"]
        with (self.dir / "pair.csv").open(newline="") as handle:
            reader = csm.csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, fields)
            rows = list(reader)
        self.assertEqual(rows[0]["a_loss"], "1.0")
        self.assertEqual(rows[0]["diff_loss"], "")
        self.assertEqual(rows[0]["a_grad_norm"], "")
        self.assertEqual(rows[1]["diff_grad_norm"], "-1.0")
        summary = json.loads((self.dir / "pair.json").read_text())
        self.assertEqual(summary["metrics"]["grad_norm"]["a_missing_updates"], [0])
        self.assertEqual(summary["metrics"]["loss"]["b_missing_updates"], [0])
        self.assertEqual(summary["metrics"]["grad_norm"]["paired"], 1)
        self.assert_cli_error(args + ["--expected-updates", "0:1", "--rtol", "0"], "missing")
        # Also exercise the writer directly with absent keys and no rows.
        for sparse in ([{"update": 0}, {"update": 1, "diff_grad_norm": -1.0}], []):
            csm.write_outputs(sparse, summary, self.dir / "pair")
            with (self.dir / "pair.csv").open(newline="") as handle:
                self.assertEqual(next(csm.csv.reader(handle)), fields)

    def test_noninteger_steps_are_not_truncated(self):
        for raw in (0.5, -0.5, True, "0.5", "oops", float("nan"), float("inf"), [], {}):
            with self.subTest(raw=raw):
                self.assert_cli_error(
                    self.json_args([{"step": raw, "loss": 1}], [{"step": 0, "loss": 1}]), "expected an integer"
                )
        for raw in (0, 0.0, "0"):
            self.assertEqual(self.run_main(self.json_args([{"step": raw, "loss": 1}], [{"step": 0, "loss": 1}])), 0)
        a = self.write("a.log", "step=0.5 loss=1\n")
        args = [
            "--a-log", a, "--b-log", a, "--a-regex", r"step=(?P<step>\S+) loss=(?P<loss>\S+)",
            "--b-regex", r"step=(?P<step>\S+) loss=(?P<loss>\S+)",
            "--metrics", "loss", "--out-prefix", str(self.dir / "pair"),
        ]
        self.assert_cli_error(args, "expected an integer")

    def test_explicit_rank_requires_rank_group(self):
        a = self.write("a.log", "Step 1: loss=1 grad_norm=2\n")
        args = [
            "--a-log", a, "--a-regex", XT_REGEX, "--a-rank", "0",
            "--b-log", a, "--b-regex", XT_REGEX, "--metrics", "loss", "--out-prefix", str(self.dir / "pair"),
        ]
        self.assert_cli_error(args, "requires a named 'rank' group")
        args = self.json_args([{"step": 0, "loss": 1}], [{"step": 0, "loss": 1}])
        self.assert_cli_error(args + ["--a-rank", "0"], "only supported for text logs")

    def test_parse_errors_have_clear_cli_diagnostics(self):
        args = self.json_args([{"step": 0, "loss": 1}], [{"step": 0, "loss": 1}])
        self.assert_cli_error(args + ["--a-map", "bad"], "METRIC=DOTTED.KEY")
        self.assert_cli_error(args + ["--a-log", str(self.dir / "absent")], "No such file")
        self.assert_cli_error(args + ["--a-format", "text", "--a-regex", "["], "error:")
        self.assert_cli_error(args + ["--a-format", "text", "--a-regex", "no-group"], "named group 'step'")
        self.assert_cli_error(args + ["--a-log", self.write("bad.jsonl", "{invalid\n")], "invalid JSON")

    def test_finite_arithmetic_overflow_is_an_input_error(self):
        cases = (
            ([1e308], [-1e308], [], "difference"),
            ([1e308, 1e308], [0, 0], [], "mean difference"),
            ([1e308, -1e308], [0, 0], [], "mean absolute difference"),
            ([1e308], [1e308], ["--rtol", "2", "--expected-updates", "0:0"], "tolerance bound"),
            ([1e308], [1e308], ["--rtol", "1", "--atol", "1e308", "--expected-updates", "0:0"], "tolerance bound"),
        )
        for av, bv, flags, message in cases:
            with self.subTest(message=message, flags=flags):
                args = self.json_args(
                    [{"step": i, "loss": v} for i, v in enumerate(av)],
                    [{"step": i, "loss": v} for i, v in enumerate(bv)],
                )
                self.assert_cli_error(args + flags, message)

    def test_apply_criteria_rejects_incomplete_or_nonfinite_rows(self):
        a, b = csm.Side("A", {0: {"loss": 1}}), csm.Side("B", {0: {"loss": 1}})
        rows, summary = csm.compare(a, b, ["loss"])
        with self.assertRaisesRegex(ValueError, "--expected-updates"):
            csm.apply_criteria(rows, summary, ["loss"], 0, 0)
        rows, summary = csm.compare(a, b, ["loss"], expected_updates=(0, 0))
        for invalid in ([], rows * 2, [{**rows[0], "update": 1}]):
            with self.assertRaisesRegex(ValueError, "every expected update"):
                csm.apply_criteria(invalid, summary, ["loss"], 0, 0)
        for key in ("a_loss", "b_loss", "diff_loss"):
            for value in (None, float("nan"), float("inf"), -float("inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    csm.apply_criteria([{**rows[0], key: value}], summary, ["loss"], 0, 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            csm.apply_criteria(rows, summary, ["loss"], -1, 0)

    def test_writer_refuses_nonstandard_json_before_creating_outputs(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                csm.write_outputs([], {"metrics_requested": ["loss"], "bad": value}, self.dir / "pair")
            self.assertFalse((self.dir / "pair.json").exists())
            self.assertFalse((self.dir / "pair.csv").exists())


if __name__ == "__main__":
    unittest.main()
