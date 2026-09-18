#!/usr/bin/env python3
"""CPU tests for the single-pass 8×6 Kaggle probe (no Kaggle submit)."""
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
INSTALL = ROOT / "scripts/kaggle/install_single8x6_cron.sh"
SUBMIT = ROOT / "scripts/kaggle/submit_single8x6.py"
WATCH = ROOT / "scripts/kaggle/watch_submission.py"
PUSH = ROOT / "scripts/kaggle/push_single8x6.py"


def notebook_src() -> str:
    nb = json.loads(NB.read_text())
    return "\n".join("".join(c.get("source") or []) for c in nb["cells"])


class NotebookSinglePassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = notebook_src()
        if 'SCHEDULE = "leftover"' in cls.src:
            raise unittest.SkipTest("notebook is leftover-B v17; single-pass probe tests retired")

    def test_schedule_single(self):
        self.assertIn('SCHEDULE = "single"', self.src)
        self.assertNotIn('SCHEDULE = "half2"', self.src)

    def test_recipe_8x6(self):
        self.assertIn('COMMON["ARC_N_TRAIN_AUG"] = "8"', self.src)
        self.assertIn('COMMON["ARC_N_EVAL_GEOS"] = "6"', self.src)

    def test_single_passes_a_only(self):
        idx = self.src.find('SCHEDULE == "single"')
        self.assertGreaterEqual(idx, 0)
        chunk = self.src[idx: idx + 500].split("RESERVE_PER_LATER")[0]
        self.assertIn('name="A"', chunk)
        self.assertNotIn('name="B"', chunk)

    def test_wall_hours_printed(self):
        self.assertIn("wall_h=", self.src)
        self.assertIn("WALL hours=", self.src)


class LockSplitTests(unittest.TestCase):
    def test_cron_flock_not_fcntl_lock(self):
        sh = INSTALL.read_text()
        self.assertIn("single8x6_submit.cron.lock", sh)
        self.assertIn("single8x6_watch.cron.lock", sh)
        submit = SUBMIT.read_text()
        watch = WATCH.read_text()
        self.assertIn("/tmp/nvarc_single8x6_submit.lock", submit)
        self.assertIn("/tmp/nvarc_single8x6_watch.lock", watch)
        self.assertNotIn("single8x6_submit.cron.lock", submit)
        self.assertNotIn("single8x6_watch.cron.lock", watch)
        self.assertIn("*/10 * * * *", sh)
        self.assertIn("*/30 * * * *", sh)
        self.assertIn("2026-09-18T00:00:00", SUBMIT.read_text())


class DryRunScriptTests(unittest.TestCase):
    def test_push_dry_run_accepts_notebook(self):
        src = notebook_src()
        if 'SCHEDULE = "leftover"' in src:
            self.skipTest("notebook is leftover-B v17")
        p = subprocess.run(
            [sys.executable, str(PUSH), "--dry-run"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("SCHEDULE=single", p.stdout)

    def test_submit_too_early(self):
        with tempfile.TemporaryDirectory() as td:
            env = os.environ.copy()
            env["NVARC_SINGLE_NOT_BEFORE"] = "2099-01-01T00:00:00+00:00"
            env["NVARC_SINGLE_STATE"] = str(Path(td) / "submit.json")
            env["NVARC_SINGLE_LOCK"] = str(Path(td) / "submit.lock")
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
                "submitted_utc": "2026-09-18T00:05:00+00:00",
                "ref": "999",
                "runs": {"single8x6": {"submitted_utc": "2026-09-18T00:05:00+00:00", "ref": "999"}},
            }))
            env = os.environ.copy()
            env["NVARC_SINGLE_NOT_BEFORE"] = "2020-01-01T00:00:00+00:00"
            env["NVARC_SINGLE_STATE"] = str(Path(td) / "submit.json")
            env["NVARC_SINGLE_LOCK"] = str(Path(td) / "submit.lock")
            p = subprocess.run(
                [sys.executable, str(SUBMIT), "--once", "--dry-run"],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("already done", p.stdout)

    def test_watch_waits_for_ref(self):
        with tempfile.TemporaryDirectory() as td:
            env = os.environ.copy()
            env["NVARC_WATCH_STATE"] = str(Path(td) / "watch.json")
            env["NVARC_SINGLE_STATE"] = str(Path(td) / "missing.json")
            env["NVARC_WATCH_LOCK"] = str(Path(td) / "watch.lock")
            p = subprocess.run(
                [sys.executable, str(WATCH), "--once"],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("waiting for single8x6 submit ref", p.stdout)

    def test_watch_terminal_noop(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "watch.json"
            state.write_text(json.dumps({
                "terminal": True,
                "ref": "123",
                "status": "COMPLETE",
                "elapsed_h": 8.5,
                "publicScore": "0.28750",
            }))
            env = os.environ.copy()
            env["NVARC_WATCH_STATE"] = str(state)
            env["NVARC_WATCH_LOCK"] = str(Path(td) / "watch.lock")
            p = subprocess.run(
                [sys.executable, str(WATCH), "--once"],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("already terminal", p.stdout)
            self.assertIn("elapsed_h=8.5", p.stdout)

    def test_extract_version_camel_case(self):
        sys.path.insert(0, str(ROOT / "scripts/kaggle"))
        import push_single8x6
        self.assertEqual(push_single8x6.extract_version({"versionNumber": 15}), 15)
        self.assertEqual(push_single8x6.extract_version({"version_number": 16}), 16)


if __name__ == "__main__":
    unittest.main()
