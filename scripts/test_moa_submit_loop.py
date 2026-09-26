"""Offline state-machine tests; never calls the real MOA service."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import moa_submit_loop as loop


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source = root / "version"
        (source / "src/device").mkdir(parents=True)
        (source / "Cargo.toml").write_text('[dependencies]\nfuriosa-opt-std = "=0.8.1"\n')
        (source / "src/ops.rs").write_text("weighted_input\n")
        (source / "src/device/mod.rs").write_text("kernel\n")
        self.state_dir = root / "run"
        self.state_dir.mkdir()
        self.args = argparse.Namespace(source=source, recover_id=None, target_score=8.0,
                                       poll_seconds=300, cooldown_seconds=0, max_submissions=0,
                                       continue_on_failure=False, max_consecutive_failures=3)

    def state(self):
        return json.loads((self.state_dir / "state.json").read_text())

    def invoke(self, responses):
        calls = []

        def fake(_cli, args, _timeout):
            calls.append(args)
            expected, response = responses.pop(0)
            self.assertEqual(args, expected)
            if isinstance(response, BaseException):
                raise response
            return response

        with patch.object(loop, "run_cli", side_effect=fake), patch.object(loop.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()):
            result = loop.loop(self.args, "fake", self.state_dir)
        self.assertEqual(responses, [])
        return result, calls

    def submit(self, ident="1234abcd"):
        return (["submit", "--source", str(self.state_dir / "source")],
                (0, f"Submission: {ident}\n"))

    def status(self, ident, status, score=""):
        text = f"Submission: {ident}\nStatus: {status}\n"
        if score:
            text += f"Score: {score}\n"
        return (["status", ident], (0, text))

    def log(self, ident="1234abcd"):
        return (["log", ident], (0, "log content"))

    def test_only_resubmit_after_terminal_and_stop_at_target(self):
        result, calls = self.invoke([
            (["status", "--all"], (0, "No submissions")), self.submit(),
            self.status("1234abcd", "building"), self.log(),
            self.status("1234abcd", "completed", "7.2"), self.log(),
            (["status", "--all"], (0, "1234abcd Team completed 7.2 2026-09-23 12:00")),
            self.submit("abcd1234"), self.status("abcd1234", "completed", "8.0"), self.log("abcd1234"),
        ])
        self.assertEqual(result, 0)
        self.assertEqual(self.state()["best_score"], 8)
        self.assertEqual(sum(call[0] == "submit" for call in calls), 2)
        # Resume an already successful run causes no network calls.
        self.invoke([])

    def test_wait_for_other_submission_and_retry_read_error(self):
        self.invoke([
            (["status", "--all"], (0, "aabbccdd My Team evaluating - 2026-09-23 12:00")),
            (["status", "--all"], (1, "network unavailable")),
            (["status", "--all"], (0, "aabbccdd My Team completed 9.0 2026-09-23 12:00")),
            self.submit(), self.status("1234abcd", "completed", "8.1"), self.log(),
        ])

    def test_uncertain_submit_never_retries_and_can_recover_id(self):
        submit_args, _ = self.submit()
        with self.assertRaisesRegex(ValueError, "uncertain"):
            self.invoke([(["status", "--all"], (0, "No submissions")),
                         (submit_args, subprocess.TimeoutExpired("fake", 180))])
        self.assertEqual(self.state()["phase"], "submitting")
        with self.assertRaisesRegex(ValueError, "uncertain"):
            self.invoke([])
        self.args.recover_id = "1234abcd"
        self.invoke([
            (["status", "--all"], (0, "1234abcd Team building - 2026-09-23 12:00")),
            self.status("1234abcd", "completed", "9"), self.log(),
        ])
        self.assertEqual(self.state()["attempts"], 1)

    def test_resume_active_uses_frozen_source_despite_original_edits(self):
        state = loop.load_or_create(self.args, self.state_dir)
        state.update(active_id="1234abcd", phase="waiting", attempts=1)
        loop.save(self.state_dir / "state.json", state)
        (self.args.source / "src/ops.rs").write_text("changed original")
        self.invoke([self.status("1234abcd", "completed", "8"), self.log()])
        self.assertEqual((self.state_dir / "source/src/ops.rs").read_text(), "weighted_input\n")

    def test_frozen_source_tampering_stops(self):
        loop.load_or_create(self.args, self.state_dir)
        (self.state_dir / "source/src/ops.rs").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "modified"):
            self.invoke([])

    def test_failure_stops_and_missing_score_preserves_active_id(self):
        result, _ = self.invoke([(["status", "--all"], (0, "No submissions")), self.submit(),
                                 self.status("1234abcd", "failed"), self.log()])
        self.assertEqual(result, 2)
        self.assertIsNone(self.state()["best_score"])
        with self.assertRaisesRegex(ValueError, "no numeric score"):
            loop.detail("Submission: 1234abcd\nStatus: completed\n", "1234abcd")

    def test_limit_waits_for_last_submission(self):
        self.args.max_submissions = 1
        self.invoke([(["status", "--all"], (0, "No submissions")), self.submit(),
                     self.status("1234abcd", "evaluating"), self.log(),
                     self.status("1234abcd", "completed", "7"), self.log()])
        self.assertIsNone(self.state()["active_id"])

    def test_opt_in_failure_retry_has_a_limit(self):
        self.args.continue_on_failure = True
        self.args.max_consecutive_failures = 2
        result, _ = self.invoke([
            (["status", "--all"], (0, "No submissions")), self.submit(),
            self.status("1234abcd", "failed"), self.log(),
            (["status", "--all"], (0, "1234abcd Team failed - 2026-09-23 12:00")),
            self.submit("abcd1234"), self.status("abcd1234", "rejected"), self.log("abcd1234"),
        ])
        self.assertEqual(result, 2)
        self.assertEqual(self.state()["attempts"], 2)

    @unittest.skipIf(__import__("sys").platform == "win32", "Linux/WSL flock")
    def test_lock_excludes_second_runner_and_releases_after_exit(self):
        with patch.object(loop.Path, "home", return_value=Path(self.temp.name)):
            with loop.account_lock():
                with self.assertRaisesRegex(ValueError, "Another MOA loop"):
                    with loop.account_lock():
                        self.fail("second loop acquired lock")
            with loop.account_lock():
                pass

    def test_parse_errors_and_wrong_identity_stop(self):
        for text in ("unknown format", "1234abcd Team newstate - 2026-09-23", "1234abcd bad row"):
            with self.assertRaises(ValueError):
                loop.listing(text)
        with self.assertRaises(ValueError):
            loop.detail("Submission: aabbccdd\nStatus: completed\nScore: 9", "1234abcd")
        with self.assertRaises(ValueError):
            loop.detail("Submission: 1234abcd\nStatus: completed\nScore: nan", "1234abcd")


if __name__ == "__main__":
    unittest.main()
