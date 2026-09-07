"""Render ARC grids as a standalone HTML report. Stdlib only."""

from __future__ import annotations

from html import escape
from pathlib import Path

from .baseline import Grid, shape

ARC_COLORS = {
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}


def grid_html(grid: Grid, cell: int = 16) -> str:
    h, w = shape(grid)
    cells = []
    for row in grid:
        for val in row:
            color = ARC_COLORS.get(val, "#ffffff")
            cells.append(
                f'<div class="cell" style="background:{color}" title="{val}"></div>'
            )
    return (
        f'<div class="arc-grid" style="grid-template-columns:repeat({w},{cell}px);'
        f'grid-auto-rows:{cell}px;width:{w * cell}px">'
        + "".join(cells)
        + "</div>"
        + f'<div class="dim">{h}x{w}</div>'
    )


def pair_html(inp: Grid, out: Grid | None, label: str) -> str:
    right = grid_html(out) if out is not None else "<div class='missing'>?</div>"
    return (
        f'<div class="pair"><div class="label">{escape(label)}</div>'
        f'<div class="io">{grid_html(inp)}<div class="arrow">→</div>{right}</div></div>'
    )


def render_report(task_blocks: list[str], title: str, summary: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; background:#111; color:#eee; margin:24px; }}
    a {{ color:#7FDBFF; }}
    .summary {{ background:#1b1b1b; padding:16px; border-radius:8px; margin-bottom:24px; }}
    .task {{ border:1px solid #333; border-radius:8px; padding:16px; margin-bottom:24px; }}
    .ok {{ color:#2ECC40; }}
    .bad {{ color:#FF4136; }}
    .pair {{ margin:12px 0; }}
    .io {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
    .arrow {{ font-size:24px; color:#888; }}
    .arc-grid {{ display:grid; gap:1px; background:#222; padding:4px; }}
    .cell {{ width:100%; height:100%; box-sizing:border-box; }}
    .dim {{ color:#888; font-size:12px; margin-top:4px; }}
    .label {{ color:#aaa; font-size:13px; margin-bottom:6px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <div class="summary">{summary}</div>
  {''.join(task_blocks)}
</body>
</html>
"""


def write_report(path: Path, task_blocks: list[str], title: str, summary: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(task_blocks, title, summary), encoding="utf-8")
