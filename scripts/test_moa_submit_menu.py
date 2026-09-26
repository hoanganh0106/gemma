"""Offline menu tests. The submission runner is mocked in every menu test."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import moa_submit_menu as menu


class MenuTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "src/device").mkdir(parents=True)
        (self.root / "Cargo.toml").write_text("[package]\n")
        (self.root / "src/ops.rs").write_text("kernel")

    def invoke(self, answers):
        with patch("builtins.input", side_effect=answers), patch.object(menu.runner, "main", return_value=0) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(menu.main(self.root), 0)
        return run

    def test_exit_never_calls_runner(self):
        self.invoke(["0"]).assert_not_called()
        self.invoke(["1", "0"]).assert_not_called()

    def test_dry_run_does_not_submit_or_save_settings(self):
        run = self.invoke(["1", "1", "8,2", "10", "2"])
        argv = run.call_args.args[0]
        self.assertIn("--dry-run", argv)
        self.assertEqual(argv[argv.index("--target-score") + 1], "8.2")
        self.assertFalse((self.root / "submission_staging").exists())

    def test_start_saves_settings_and_passes_user_choices(self):
        run = self.invoke(["1", "1", "7.9", "20", "1"])
        argv = run.call_args.args[0]
        self.assertNotIn("--dry-run", argv)
        saved = json.loads((self.root / "submission_staging/moa-menu.json").read_text())
        self.assertEqual(saved["source"], str(self.root))
        self.assertEqual(saved["target_score"], 7.9)
        self.assertEqual(saved["max_submissions"], 20)

    def test_resume_passes_exact_state_and_remembers_target(self):
        staging = self.root / "submission_staging"
        directory = staging / "old-run"
        directory.mkdir(parents=True)
        (directory / "state.json").write_text(json.dumps({"source": str(self.root), "hashes": {},
                                                        "phase": "waiting", "active_id": "1234abcd"}))
        (staging / "moa-menu.json").write_text(json.dumps({"source": str(self.root), "target_score": 8.3,
                                                         "max_submissions": 12}))
        run = self.invoke(["1", "", "", "1"])
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("--state-dir") + 1], str(directory))
        self.assertEqual(argv[argv.index("--target-score") + 1], "8.3")
        self.assertEqual(argv[argv.index("--max-submissions") + 1], "12")

    def test_uncertain_submit_allows_exit_without_resubmit(self):
        directory = self.root / "submission_staging/old-run"
        directory.mkdir(parents=True)
        (directory / "state.json").write_text(json.dumps({"source": str(self.root), "hashes": {}, "phase": "submitting"}))
        self.invoke(["1", ""]).assert_not_called()

    def test_remembers_version_even_if_already_in_menu(self):
        baseline = self.root / "research/furiosa-score-20260923/baseline-source"
        baseline.mkdir(parents=True)
        (baseline / "Cargo.toml").write_text("[package]")
        with patch("builtins.input", return_value=""), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(menu.select_source(self.root, str(self.root)), self.root)

    def test_windows_path_passed_as_argument_without_shell(self):
        with patch.object(menu.subprocess, "check_output", return_value="/mnt/d/a b\n") as convert:
            result = menu.version_path('"D:\\a b"', self.root)
        convert.assert_called_once_with(["wslpath", "-u", "D:\\a b"], text=True)
        self.assertEqual(result, Path("/mnt/d/a b").resolve())

    def test_reprompt_invalid_numbers(self):
        with patch("builtins.input", side_effect=["nan", "-1", "8,5"]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(menu.number("score", 8), 8.5)


if __name__ == "__main__":
    unittest.main()
