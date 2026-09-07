"""First-pass ARC-AGI-2 baseline: fit a small set of grid transforms on train pairs.

No third-party dependencies. Safe to paste into a Kaggle notebook (no internet).
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

Grid = list[list[int]]
Transform = Callable[[Grid], Grid]


def shape(grid: Grid) -> tuple[int, int]:
    if not grid:
        return 0, 0
    return len(grid), len(grid[0])


def copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def equal(a: Grid, b: Grid) -> bool:
    return a == b


def rot90(grid: Grid) -> Grid:
    h, w = shape(grid)
    return [[grid[h - 1 - j][i] for j in range(h)] for i in range(w)]


def rot180(grid: Grid) -> Grid:
    return rot90(rot90(grid))


def rot270(grid: Grid) -> Grid:
    return rot90(rot180(grid))


def flip_h(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def flip_v(grid: Grid) -> Grid:
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    h, w = shape(grid)
    return [[grid[i][j] for i in range(h)] for j in range(w)]


def shift_down(grid: Grid) -> Grid:
    h, w = shape(grid)
    out = [[0] * w for _ in range(h)]
    for i in range(h - 1):
        out[i + 1] = grid[i][:]
    return out


def gravity_down(grid: Grid) -> Grid:
    h, w = shape(grid)
    out = [[0] * w for _ in range(h)]
    for j in range(w):
        vals = [grid[i][j] for i in range(h) if grid[i][j] != 0]
        for k, val in enumerate(vals):
            out[h - len(vals) + k][j] = val
    return out


def crop_nonzero(grid: Grid) -> Grid:
    coords = [(i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val]
    if not coords:
        return [[0]]
    rows = [i for i, _ in coords]
    cols = [j for _, j in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return [row[c0 : c1 + 1] for row in grid[r0 : r1 + 1]]


def pattern_kronecker(grid: Grid) -> Grid:
    n, m = shape(grid)
    out = [[0] * (m * m) for _ in range(n * n)]
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 0:
                continue
            for r in range(n):
                for c in range(m):
                    out[i * n + r][j * m + c] = grid[r][c]
    return out


def fill_unique_nonzero(grid: Grid) -> Grid:
    colors = {val for row in grid for val in row if val}
    if len(colors) != 1:
        return copy_grid(grid)
    color = next(iter(colors))
    h, w = shape(grid)
    return [[color] * w for _ in range(h)]


def majority_fill(grid: Grid) -> Grid:
    h, w = shape(grid)
    if h * w == 0:
        return copy_grid(grid)
    color = Counter(val for row in grid for val in row).most_common(1)[0][0]
    return [[color] * w for _ in range(h)]


def uniform_rows_to_five(grid: Grid) -> Grid:
    out = []
    for row in grid:
        color = 5 if row and all(v == row[0] for v in row) else 0
        out.append([color] * len(row))
    return out


def largest_color_block(grid: Grid) -> Grid:
    counts = Counter(val for row in grid for val in row if val)
    if not counts:
        return [[0, 0], [0, 0]]
    color = counts.most_common(1)[0][0]
    return [[color, color], [color, color]]


def connected_components(grid: Grid) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = shape(grid)
    seen = [[False] * w for _ in range(h)]
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 or seen[i][j]:
                continue
            color = grid[i][j]
            stack = [(i, j)]
            seen[i][j] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            comps.append((color, cells))
    return comps


def recolor_objects_by_size(grid: Grid) -> Grid:
    comps = connected_components(grid)
    if not comps:
        return copy_grid(grid)
    comps.sort(key=lambda item: len(item[1]), reverse=True)
    h, w = shape(grid)
    out = [[0] * w for _ in range(h)]
    for idx, (_, cells) in enumerate(comps, start=1):
        color = min(idx, 9)
        for r, c in cells:
            out[r][c] = color
    return out


def fill_enclosed_zeros(grid: Grid, fill: int = 4) -> Grid:
    h, w = shape(grid)
    out = copy_grid(grid)
    seen = [[False] * w for _ in range(h)]
    stack = []
    for i in range(h):
        for j in (0, w - 1) if w else ():
            if grid[i][j] == 0 and not seen[i][j]:
                seen[i][j] = True
                stack.append((i, j))
    for j in range(w):
        for i in (0, h - 1) if h else ():
            if grid[i][j] == 0 and not seen[i][j]:
                seen[i][j] = True
                stack.append((i, j))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == 0:
                seen[nr][nc] = True
                stack.append((nr, nc))
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 and not seen[i][j]:
                out[i][j] = fill
    return out


def split_and_and(grid: Grid) -> Grid:
    """Left/right halves AND into a smaller grid. Used by some ARC-1 tasks."""
    h, w = shape(grid)
    if w < 3 or w % 2 == 0:
        mid = w // 2
    else:
        mid = w // 2
    left_w = mid
    right = [row[mid + (1 if w % 2 else 0) :] for row in grid]
    left = [row[:left_w] for row in grid]
    if shape(left) != shape(right) or not left:
        return copy_grid(grid)
    rh, rw = shape(right)
    out = [[0] * rw for _ in range(rh)]
    for i in range(rh):
        for j in range(rw):
            out[i][j] = left[i][j] if left[i][j] and right[i][j] else 0
    return out


def integer_scale(grid: Grid, factor: int) -> Grid:
    out = []
    for row in grid:
        scaled = []
        for val in row:
            scaled.extend([val] * factor)
        for _ in range(factor):
            out.append(scaled[:])
    return out


def recolor(grid: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(val, val) for val in row] for row in grid]


def infer_color_map(pairs: list[tuple[Grid, Grid]]) -> Transform | None:
    mapping: dict[int, int] = {}
    for inp, out in pairs:
        if shape(inp) != shape(out):
            return None
        for i, row in enumerate(inp):
            for j, val in enumerate(row):
                target = out[i][j]
                if val in mapping and mapping[val] != target:
                    return None
                mapping[val] = target
    if not mapping or all(k == v for k, v in mapping.items()):
        return None

    def apply(grid: Grid) -> Grid:
        return recolor(grid, mapping)

    return apply


def infer_scale_factor(pairs: list[tuple[Grid, Grid]]) -> Transform | None:
    factor = None
    for inp, out in pairs:
        ih, iw = shape(inp)
        oh, ow = shape(out)
        if ih == 0 or iw == 0 or oh % ih or ow % iw:
            return None
        fh, fw = oh // ih, ow // iw
        if fh != fw or fh < 2:
            return None
        if factor is None:
            factor = fh
        elif factor != fh:
            return None
        if integer_scale(inp, factor) != out:
            return None
    if factor is None:
        return None

    def apply(grid: Grid) -> Grid:
        return integer_scale(grid, factor)

    return apply


def infer_enclosed_fill(pairs: list[tuple[Grid, Grid]]) -> Transform | None:
    fill_colors = set()
    for inp, out in pairs:
        if shape(inp) != shape(out):
            return None
        extra = {out[i][j] for i, row in enumerate(out) for j, val in enumerate(row) if val != inp[i][j]}
        extra.discard(0)
        if len(extra) != 1:
            return None
        fill_colors.add(next(iter(extra)))
    if len(fill_colors) != 1:
        return None
    fill = next(iter(fill_colors))

    def apply(grid: Grid) -> Grid:
        return fill_enclosed_zeros(grid, fill)

    return apply


NAMED_TRANSFORMS: list[tuple[str, Transform]] = [
    ("identity", copy_grid),
    ("rot90", rot90),
    ("rot180", rot180),
    ("rot270", rot270),
    ("flip_h", flip_h),
    ("flip_v", flip_v),
    ("transpose", transpose),
    ("shift_down", shift_down),
    ("gravity_down", gravity_down),
    ("crop_nonzero", crop_nonzero),
    ("pattern_kronecker", pattern_kronecker),
    ("fill_unique_nonzero", fill_unique_nonzero),
    ("majority_fill", majority_fill),
    ("uniform_rows_to_five", uniform_rows_to_five),
    ("largest_color_block", largest_color_block),
    ("recolor_objects_by_size", recolor_objects_by_size),
    ("split_and_and", split_and_and),
]


def fits_all(transform: Transform, pairs: list[tuple[Grid, Grid]]) -> bool:
    try:
        return all(equal(transform(inp), out) for inp, out in pairs)
    except Exception:
        return False


def fit_transforms(train_pairs: list[tuple[Grid, Grid]]) -> list[tuple[str, Transform]]:
    fitted: list[tuple[str, Transform]] = []
    seen_ids: set[int] = set()

    def add(name: str, transform: Transform) -> None:
        if id(transform) in seen_ids:
            return
        if fits_all(transform, train_pairs):
            fitted.append((name, transform))
            seen_ids.add(id(transform))

    for name, transform in NAMED_TRANSFORMS:
        add(name, transform)

    color_map = infer_color_map(train_pairs)
    if color_map is not None:
        add("color_map", color_map)

    scale = infer_scale_factor(train_pairs)
    if scale is not None:
        add("integer_scale", scale)

    enclosed = infer_enclosed_fill(train_pairs)
    if enclosed is not None:
        add("enclosed_fill", enclosed)

    if not fitted:
        fitted.append(("identity_fallback", copy_grid))
    return fitted


def predict_grid(train_pairs: list[tuple[Grid, Grid]], test_input: Grid) -> tuple[Grid, Grid, list[str]]:
    fitted = fit_transforms(train_pairs)
    names = [name for name, _ in fitted]
    attempt_1 = fitted[0][1](test_input)
    attempt_2 = fitted[1][1](test_input) if len(fitted) > 1 else copy_grid(attempt_1)
    if equal(attempt_1, attempt_2) and len(fitted) == 1:
        attempt_2 = flip_h(attempt_1) if shape(attempt_1)[1] > 1 else copy_grid(attempt_1)
    return attempt_1, attempt_2, names


def train_pairs_of(task: dict) -> list[tuple[Grid, Grid]]:
    return [(pair["input"], pair["output"]) for pair in task["train"]]


def predict_task(task: dict) -> list[dict]:
    pairs = train_pairs_of(task)
    predictions = []
    for test_pair in task["test"]:
        attempt_1, attempt_2, _ = predict_grid(pairs, test_pair["input"])
        predictions.append({"attempt_1": attempt_1, "attempt_2": attempt_2})
    return predictions


def score_task(task: dict) -> dict:
    pairs = train_pairs_of(task)
    results = []
    solved = True
    for test_pair in task["test"]:
        truth = test_pair.get("output")
        attempt_1, attempt_2, names = predict_grid(pairs, test_pair["input"])
        hit = truth is not None and (equal(attempt_1, truth) or equal(attempt_2, truth))
        solved = solved and hit
        results.append(
            {
                "hit": hit,
                "names": names,
                "attempt_1": attempt_1,
                "attempt_2": attempt_2,
                "truth": truth,
                "input": test_pair["input"],
            }
        )
    return {"solved": solved, "per_test": results}


def build_submission(challenges: dict[str, dict]) -> dict[str, list[dict]]:
    return {task_id: predict_task(task) for task_id, task in challenges.items()}


def flatten_challenges(payload: dict | list, stem: str | None = None) -> dict[str, dict]:
    """Normalize Kaggle combined JSON or a folder of per-task files."""
    if isinstance(payload, list):
        raise TypeError("expected a task dict or a mapping of task_id -> task")
    if "train" in payload and "test" in payload:
        return {stem or "task": payload}
    return payload
