#!/usr/bin/env python3
"""Run the baseline on bundled example tasks and write an HTML report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc_solver.baseline import predict_task, score_task  # noqa: E402
from arc_solver.visualize import grid_html, pair_html, write_report  # noqa: E402

EXAMPLES = ROOT / "data" / "examples"
OUT_DIR = ROOT / "demo" / "output"


def load_tasks() -> dict[str, dict]:
    tasks = {}
    for path in sorted(EXAMPLES.glob("*.json")):
        tasks[path.stem] = json.loads(path.read_text())
    return tasks


def main() -> int:
    tasks = load_tasks()
    solved = 0
    test_hits = 0
    test_total = 0
    blocks = []
    submission = {}

    for task_id, task in tasks.items():
        scored = score_task(task)
        preds = predict_task(task)
        submission[task_id] = preds
        if scored["solved"]:
            solved += 1
        names = scored["per_test"][0]["names"] if scored["per_test"] else []
        status = "ok" if scored["solved"] else "bad"
        status_text = "解出" if scored["solved"] else "未解出"
        parts = [
            f'<div class="task"><h2>{task_id} <span class="{status}">{status_text}</span></h2>',
            f"<p>拟合到的变换：<code>{', '.join(names) or 'identity_fallback'}</code></p>",
            "<h3>示范 train</h3>",
        ]
        for i, pair in enumerate(task["train"], start=1):
            parts.append(pair_html(pair["input"], pair["output"], f"train {i}"))
        parts.append("<h3>测试 test（左：输入，中：预测 attempt_1，右：标准答案）</h3>")
        for item in scored["per_test"]:
            test_total += 1
            test_hits += int(item["hit"])
            mark = "命中" if item["hit"] else "未命中"
            parts.append(f"<p>{mark}</p>")
            parts.append(
                '<div class="io">'
                + grid_html(item["input"])
                + '<div class="arrow">预测→</div>'
                + grid_html(item["attempt_1"])
                + '<div class="arrow">答案→</div>'
                + (grid_html(item["truth"]) if item["truth"] is not None else "")
                + "</div>"
            )
        parts.append("</div>")
        blocks.append("".join(parts))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = OUT_DIR / "demo_submission.json"
    sub_path.write_text(json.dumps(submission, separators=(",", ":")), encoding="utf-8")

    summary = (
        f"本地示例 {len(tasks)} 题，整题解出 {solved}/{len(tasks)}；"
        f"测试格子命中 {test_hits}/{test_total}。"
        f"提交样例写在 <code>{sub_path.relative_to(ROOT)}</code>。"
        "这些题来自公开训练集，只能证明管道能跑通，不能代表 Kaggle 隐藏题分数。"
    )
    report_path = OUT_DIR / "report.html"
    write_report(report_path, blocks, "ARC-AGI-2 首次 baseline demo", summary)
    print(summary.replace("<code>", "").replace("</code>", ""))
    print(f"HTML: {report_path}")
    print(f"JSON: {sub_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
