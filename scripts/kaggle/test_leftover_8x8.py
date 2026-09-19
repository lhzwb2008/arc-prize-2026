#!/usr/bin/env python3
"""CPU tests for the leftover-B 8×8 Kaggle kernel (no Kaggle submit)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "notebooks/nvarc_2026/nvarc_qwen3_4b_ttt_2026.ipynb"
INSTALL = ROOT / "scripts/kaggle/install_leftover_8x8_cron.sh"
SUBMIT = ROOT / "scripts/kaggle/submit_leftover_8x8.py"
WATCH = ROOT / "scripts/kaggle/watch_submission.py"
PUSH = ROOT / "scripts/kaggle/push_leftover_8x8.py"
REFRESH = ROOT / "scripts/kaggle/refresh_kaggle_oauth.py"


def notebook_src() -> str:
    nb = json.loads(NB.read_text())
    return "\n".join("".join(c.get("source") or []) for c in nb["cells"])


class NotebookLeftover8x8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = notebook_src()

    def test_schedule_leftover(self):
        self.assertIn('SCHEDULE = "leftover"', self.src)
        self.assertNotIn('SCHEDULE = "single"', self.src)
        self.assertNotIn('SCHEDULE = "half2"', self.src)

    def test_recipe_8x8(self):
        self.assertIn('COMMON["ARC_N_TRAIN_AUG"] = "8"', self.src)
        self.assertIn('COMMON["ARC_N_EVAL_GEOS"] = "8"', self.src)
        self.assertNotIn('COMMON["ARC_N_EVAL_GEOS"] = "6"', self.src)

    def test_leftover_passes_include_b_file_order(self):
        idx = self.src.find('SCHEDULE = "leftover"')
        self.assertGreaterEqual(idx, 0)
        self.assertIn('name="B"', self.src)
        self.assertIn('order="file"', self.src)
        self.assertIn("b_priority_queue.py", self.src)

    def test_save_version_skips_b(self):
        self.assertIn("Save Version smoke: skip leftover-B", self.src)
        self.assertIn("gate_hidden()", self.src)

    def test_keep_primary_and_hard_merge_only(self):
        self.assertIn('NVARC_POOL_MODE": "keep-primary"', self.src)
        self.assertIn("_hard_merge_watchdog", self.src)
        self.assertIn("hard-Tminus5", self.src)
        self.assertNotIn('live_checkpoint("tick"', self.src)
        self.assertNotIn('"NVARC_CHECKPOINT_MIN_GAP": "300"', self.src)
        self.assertNotIn('"NVARC_CHECKPOINT_EVERY": "90"', self.src)

    def test_wall_hours_printed(self):
        self.assertIn("wall_h=", self.src)
        self.assertIn("schedule=leftover 8x8 A+B-unc", self.src)

    def test_workers_stop_at_2min_not_20min(self):
        self.assertIn("global_end_time = T0 + 12 * 3600 - 2 * 60", self.src)
        self.assertNotIn("global_end_time = T0 + 12 * 3600 - 20 * 60", self.src)
        self.assertIn("hard_merge_time = T0 + 12 * 3600 - 5 * 60", self.src)


class LockSplitTests(unittest.TestCase):
    def test_cron_flock_not_fcntl_lock(self):
        sh = INSTALL.read_text()
        self.assertIn("leftover_8x8_submit.cron.lock", sh)
        self.assertIn("leftover_8x8_watch.cron.lock", sh)
        submit = SUBMIT.read_text()
        watch = WATCH.read_text()
        self.assertIn("/tmp/nvarc_leftover_8x8_submit.lock", submit)
        self.assertIn("/tmp/nvarc_leftover_8x8_watch.lock", sh)
        self.assertNotIn("leftover_8x8_submit.cron.lock", submit)
        self.assertNotIn("leftover_8x8_watch.cron.lock", watch)
        self.assertIn("*/10 * * * *", sh)
        self.assertIn("*/30 * * * *", sh)
        self.assertIn("2026-09-20T00:00:00", submit)
        self.assertIn("2026-09-20T00:00:00Z", sh)
        self.assertIn("leftover_unc auto-submit", sh)
        self.assertIn("RUN_ID = \"leftover_8x8\"", submit)
        self.assertIn("refresh_kaggle_oauth.py", sh)
        self.assertTrue(REFRESH.is_file())
        self.assertIn("Does not print tokens", REFRESH.read_text())


class DryRunScriptTests(unittest.TestCase):
    def test_push_dry_run_accepts_notebook(self):
        p = subprocess.run(
            [sys.executable, str(PUSH), "--dry-run"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("SCHEDULE=leftover 8×8", p.stdout)

    def test_submit_too_early(self):
        with tempfile.TemporaryDirectory() as td:
            env = os.environ.copy()
            env["NVARC_LEFTOVER_NOT_BEFORE"] = "2099-01-01T00:00:00+00:00"
            env["NVARC_LEFTOVER_STATE"] = str(Path(td) / "submit.json")
            env["NVARC_LEFTOVER_LOCK"] = str(Path(td) / "submit.lock")
            p = subprocess.run(
                [sys.executable, str(SUBMIT), "--once", "--dry-run"],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("too early", p.stdout)

    def test_submit_already_done(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "submit.json"
            state.write_text(json.dumps({
                "submitted_utc": "2026-09-20T00:05:00+00:00",
                "ref": "999",
                "runs": {"leftover_8x8": {"submitted_utc": "2026-09-20T00:05:00+00:00", "ref": "999"}},
            }))
            env = os.environ.copy()
            env["NVARC_LEFTOVER_NOT_BEFORE"] = "2020-01-01T00:00:00+00:00"
            env["NVARC_LEFTOVER_STATE"] = str(state)
            env["NVARC_LEFTOVER_LOCK"] = str(Path(td) / "submit.lock")
            p = subprocess.run(
                [sys.executable, str(SUBMIT), "--once", "--dry-run"],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("already done", p.stdout)

    def test_leftover_unc_state_does_not_block(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "submit.json"
            state.write_text(json.dumps({
                "submitted_utc": "2026-09-19T00:00:06+00:00",
                "ref": "56343080",
                "runs": {"leftover_unc": {"submitted_utc": "2026-09-19T00:00:06+00:00", "ref": "56343080"}},
            }))
            env = os.environ.copy()
            env["NVARC_LEFTOVER_NOT_BEFORE"] = "2099-01-01T00:00:00+00:00"
            env["NVARC_LEFTOVER_STATE"] = str(state)
            env["NVARC_LEFTOVER_LOCK"] = str(Path(td) / "submit.lock")
            p = subprocess.run(
                [sys.executable, str(SUBMIT), "--once", "--dry-run"],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("too early", p.stdout)
            self.assertNotIn("already done", p.stdout)

    def test_watch_waits_for_ref(self):
        with tempfile.TemporaryDirectory() as td:
            env = os.environ.copy()
            env["NVARC_WATCH_STATE"] = str(Path(td) / "watch.json")
            env["NVARC_SINGLE_STATE"] = str(Path(td) / "missing.json")
            env["NVARC_WATCH_LOCK"] = str(Path(td) / "watch.lock")
            env["NVARC_WATCH_LABEL"] = "leftover_8x8"
            p = subprocess.run(
                [sys.executable, str(WATCH), "--once"],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("waiting for leftover_8x8 submit ref", p.stdout)

    def test_extract_version_camel_case(self):
        sys.path.insert(0, str(ROOT / "scripts/kaggle"))
        import push_leftover_8x8
        self.assertEqual(push_leftover_8x8.extract_version({"versionNumber": 18}), 18)
        self.assertEqual(push_leftover_8x8.extract_version({"version_number": 19}), 19)


if __name__ == "__main__":
    unittest.main()
