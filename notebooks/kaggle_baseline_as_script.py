"""ARC-AGI-2 solver v2: hand-written DSL search + rule induction. No ML, stdlib only.

Per task:
  1. Enumerate candidate programs:
       - depth<=2 compositions of whole-grid ops, each optionally followed by a
         learned colour map;
       - learned per-cell rules (cell colour + local context -> output colour);
       - learned per-object recolouring (object property -> colour);
       - symmetry / periodic completion of a masked region;
       - split the grid into parts and combine them with a learned truth table;
       - constant output.
  2. Keep programs that reproduce every train pair exactly.
  3. Rank by simplicity; emit the two best distinct predictions per test input.

The whole file is embedded verbatim into the Kaggle notebook, so keep it
dependency-free.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Callable, Optional

Grid = list[list[int]]
Op = Callable[[Grid], Optional[Grid]]
Pairs = list[tuple[Grid, Grid]]

MAX_SIDE = 30
MAX_VOTERS = 120

# --------------------------------------------------------------------------- basics


def shape(g: Grid) -> tuple[int, int]:
    return (len(g), len(g[0])) if g and g[0] else (0, 0)


def copy_grid(g: Grid) -> Grid:
    return [row[:] for row in g]


def valid(g) -> bool:
    if not isinstance(g, list) or not g or not isinstance(g[0], list) or not g[0]:
        return False
    w = len(g[0])
    if len(g) > MAX_SIDE or w > MAX_SIDE:
        return False
    return all(isinstance(r, list) and len(r) == w for r in g)


def bg_color(g: Grid) -> int:
    return Counter(v for row in g for v in row).most_common(1)[0][0]


def color_counts(g: Grid) -> Counter:
    return Counter(v for row in g for v in row)


def hash_grid(g: Grid) -> tuple:
    return tuple(tuple(r) for r in g)


# --------------------------------------------------------------------------- geometry


def identity(g: Grid) -> Grid:
    return g


def rot90(g: Grid) -> Grid:
    h, w = shape(g)
    return [[g[h - 1 - j][i] for j in range(h)] for i in range(w)]


def rot180(g: Grid) -> Grid:
    return [row[::-1] for row in g[::-1]]


def rot270(g: Grid) -> Grid:
    h, w = shape(g)
    return [[g[j][w - 1 - i] for j in range(h)] for i in range(w)]


def flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]


def flip_v(g: Grid) -> Grid:
    return [row[:] for row in g[::-1]]


def transpose(g: Grid) -> Grid:
    return [list(col) for col in zip(*g)]


def anti_transpose(g: Grid) -> Grid:
    return transpose(rot180(g))


GEOMETRY: list[tuple[str, Op]] = [
    ("identity", identity),
    ("rot90", rot90),
    ("rot180", rot180),
    ("rot270", rot270),
    ("flip_h", flip_h),
    ("flip_v", flip_v),
    ("transpose", transpose),
    ("anti_transpose", anti_transpose),
]

# --------------------------------------------------------------------------- scaling / tiling


def upscale(g: Grid, k: int) -> Grid:
    return [[v for v in row for _ in range(k)] for row in g for _ in range(k)]


def downscale(g: Grid, k: int) -> Optional[Grid]:
    h, w = shape(g)
    if h % k or w % k:
        return None
    out = []
    for bi in range(0, h, k):
        row = []
        for bj in range(0, w, k):
            v = g[bi][bj]
            for di in range(k):
                for dj in range(k):
                    if g[bi + di][bj + dj] != v:
                        return None
            row.append(v)
        out.append(row)
    return out


def downscale_majority(g: Grid, k: int) -> Optional[Grid]:
    h, w = shape(g)
    if h % k or w % k or (h == k and w == k):
        return None
    out = []
    for bi in range(0, h, k):
        row = []
        for bj in range(0, w, k):
            c = Counter(g[bi + di][bj + dj] for di in range(k) for dj in range(k))
            row.append(c.most_common(1)[0][0])
        out.append(row)
    return out


def tile(g: Grid, a: int, b: int) -> Grid:
    return [row * b for _ in range(a) for row in g]


def mirror_h(g: Grid) -> Grid:
    return [row + row[::-1] for row in g]


def mirror_h_rev(g: Grid) -> Grid:
    return [row[::-1] + row for row in g]


def mirror_v(g: Grid) -> Grid:
    return [r[:] for r in g] + [r[:] for r in g[::-1]]


def mirror_v_rev(g: Grid) -> Grid:
    return [r[:] for r in g[::-1]] + [r[:] for r in g]


def mirror_4(g: Grid) -> Grid:
    return mirror_v(mirror_h(g))


def mirror_4_rev(g: Grid) -> Grid:
    return mirror_v_rev(mirror_h_rev(g))


def rot_tile_4(g: Grid) -> Optional[Grid]:
    h, w = shape(g)
    if h != w:
        return None
    top = [a + b for a, b in zip(g, rot90(g))]
    bottom = [a + b for a, b in zip(rot270(g), rot180(g))]
    return top + bottom


def top_half(g: Grid) -> Optional[Grid]:
    h = len(g)
    return None if h % 2 else [r[:] for r in g[: h // 2]]


def bottom_half(g: Grid) -> Optional[Grid]:
    h = len(g)
    return None if h % 2 else [r[:] for r in g[h // 2 :]]


def left_half(g: Grid) -> Optional[Grid]:
    w = len(g[0])
    return None if w % 2 else [r[: w // 2] for r in g]


def right_half(g: Grid) -> Optional[Grid]:
    w = len(g[0])
    return None if w % 2 else [r[w // 2 :] for r in g]


# --------------------------------------------------------------------------- cropping


def crop_cells(g: Grid, cells) -> Optional[Grid]:
    if not cells:
        return None
    r0 = min(i for i, _ in cells)
    r1 = max(i for i, _ in cells)
    c0 = min(j for _, j in cells)
    c1 = max(j for _, j in cells)
    return [row[c0 : c1 + 1] for row in g[r0 : r1 + 1]]


def crop_nonbg(g: Grid) -> Optional[Grid]:
    bg = bg_color(g)
    return crop_cells(g, [(i, j) for i, row in enumerate(g) for j, v in enumerate(row) if v != bg])


def crop_nonzero(g: Grid) -> Optional[Grid]:
    return crop_cells(g, [(i, j) for i, row in enumerate(g) for j, v in enumerate(row) if v != 0])


def crop_color(g: Grid, c: int) -> Optional[Grid]:
    return crop_cells(g, [(i, j) for i, row in enumerate(g) for j, v in enumerate(row) if v == c])


def crop_color_interior(g: Grid, c: int) -> Optional[Grid]:
    box = crop_color(g, c)
    if box is None or len(box) < 3 or len(box[0]) < 3:
        return None
    return [row[1:-1] for row in box[1:-1]]


def compress_bg_lines(g: Grid) -> Optional[Grid]:
    bg = bg_color(g)
    rows = [r for r in g if any(v != bg for v in r)]
    if not rows:
        return None
    keep_cols = [j for j in range(len(g[0])) if any(r[j] != bg for r in rows)]
    return [[r[j] for j in keep_cols] for r in rows]


def compress_dup_lines(g: Grid) -> Grid:
    rows = [g[0][:]] + [g[i][:] for i in range(1, len(g)) if g[i] != g[i - 1]]
    t = transpose(rows)
    cols = [t[0]] + [t[j] for j in range(1, len(t)) if t[j] != t[j - 1]]
    return transpose(cols)


# --------------------------------------------------------------------------- gravity


def gravity(g: Grid, direction: str) -> Grid:
    bg = bg_color(g)
    h, w = shape(g)
    out = [[bg] * w for _ in range(h)]
    if direction in ("down", "up"):
        for j in range(w):
            vals = [g[i][j] for i in range(h) if g[i][j] != bg]
            if direction == "down":
                for k, v in enumerate(vals):
                    out[h - len(vals) + k][j] = v
            else:
                for k, v in enumerate(vals):
                    out[k][j] = v
    else:
        for i in range(h):
            vals = [v for v in g[i] if v != bg]
            if direction == "right":
                out[i][w - len(vals) :] = vals
            else:
                out[i][: len(vals)] = vals
    return out


# --------------------------------------------------------------------------- objects


def components(g: Grid, bg: int, diag: bool, multicolor: bool) -> list[dict]:
    h, w = shape(g)
    seen = [[False] * w for _ in range(h)]
    if diag:
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    else:
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    objs = []
    for i in range(h):
        for j in range(w):
            if seen[i][j] or g[i][j] == bg:
                continue
            color = g[i][j]
            stack = [(i, j)]
            seen[i][j] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in nbrs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc]:
                        v = g[nr][nc]
                        if v != bg and (multicolor or v == color):
                            seen[nr][nc] = True
                            stack.append((nr, nc))
            r0 = min(r for r, _ in cells)
            r1 = max(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            c1 = max(c for _, c in cells)
            cnt = Counter(g[r][c] for r, c in cells)
            objs.append(
                {
                    "cells": cells,
                    "color": cnt.most_common(1)[0][0],
                    "ncolors": len(cnt),
                    "size": len(cells),
                    "bbox": (r0, c0, r1, c1),
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                }
            )
    return objs


def obj_holes(g: Grid, obj: dict, bg: int) -> int:
    """Number of bg cells inside the object's bbox that are not reachable from the bbox border."""
    r0, c0, r1, c1 = obj["bbox"]
    cells = set(obj["cells"])
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inside = [[(r0 + i, c0 + j) not in cells for j in range(w)] for i in range(h)]
    seen = [[False] * w for _ in range(h)]
    stack = []
    for i in range(h):
        for j in (0, w - 1):
            if inside[i][j] and not seen[i][j]:
                seen[i][j] = True
                stack.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if inside[i][j] and not seen[i][j]:
                seen[i][j] = True
                stack.append((i, j))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and inside[nr][nc] and not seen[nr][nc]:
                seen[nr][nc] = True
                stack.append((nr, nc))
    return sum(1 for i in range(h) for j in range(w) if inside[i][j] and not seen[i][j])


def _pick(objs: list[dict], how: str, g: Grid) -> Optional[dict]:
    if not objs:
        return None
    if how == "largest":
        return max(objs, key=lambda o: o["size"])
    if how == "smallest":
        return min(objs, key=lambda o: o["size"])
    if how == "biggest_bbox":
        return max(objs, key=lambda o: o["h"] * o["w"])
    if how == "densest":
        return max(objs, key=lambda o: o["size"] / (o["h"] * o["w"]))
    if how == "sparsest":
        return min(objs, key=lambda o: o["size"] / (o["h"] * o["w"]))
    if how == "most_colors":
        return max(objs, key=lambda o: o["ncolors"])
    if how == "unique_color":
        cnt = Counter(o["color"] for o in objs)
        cands = [o for o in objs if cnt[o["color"]] == 1]
        return cands[0] if len(cands) == 1 else None
    if how == "unique_shape":
        shapes = Counter((o["h"], o["w"], o["size"]) for o in objs)
        cands = [o for o in objs if shapes[(o["h"], o["w"], o["size"])] == 1]
        return cands[0] if len(cands) == 1 else None
    if how == "unique_size":
        sizes = Counter(o["size"] for o in objs)
        cands = [o for o in objs if sizes[o["size"]] == 1]
        return cands[0] if len(cands) == 1 else None
    if how == "top_left":
        return min(objs, key=lambda o: (o["bbox"][0], o["bbox"][1]))
    if how == "bottom_right":
        return max(objs, key=lambda o: (o["bbox"][2], o["bbox"][3]))
    return None


def make_obj_crop(how: str, diag: bool, multicolor: bool) -> Op:
    def op(g: Grid) -> Optional[Grid]:
        bg = bg_color(g)
        objs = components(g, bg, diag, multicolor)
        if len(objs) < 2:
            return None
        o = _pick(objs, how, g)
        if o is None:
            return None
        r0, c0, r1, c1 = o["bbox"]
        return [row[c0 : c1 + 1] for row in g[r0 : r1 + 1]]

    return op


def make_obj_keep(how: str, diag: bool, multicolor: bool) -> Op:
    def op(g: Grid) -> Optional[Grid]:
        bg = bg_color(g)
        objs = components(g, bg, diag, multicolor)
        if len(objs) < 2:
            return None
        o = _pick(objs, how, g)
        if o is None:
            return None
        keep = set(o["cells"])
        return [[v if (i, j) in keep else bg for j, v in enumerate(row)] for i, row in enumerate(g)]

    return op


def make_obj_remove(how: str, diag: bool, multicolor: bool) -> Op:
    def op(g: Grid) -> Optional[Grid]:
        bg = bg_color(g)
        objs = components(g, bg, diag, multicolor)
        if len(objs) < 2:
            return None
        o = _pick(objs, how, g)
        if o is None:
            return None
        drop = set(o["cells"])
        return [[bg if (i, j) in drop else v for j, v in enumerate(row)] for i, row in enumerate(g)]

    return op


# --------------------------------------------------------------------------- fills


def fill_enclosed(g: Grid, fill: int) -> Grid:
    bg = bg_color(g)
    h, w = shape(g)
    seen = [[False] * w for _ in range(h)]
    stack = []
    for i in range(h):
        for j in (0, w - 1):
            if g[i][j] == bg and not seen[i][j]:
                seen[i][j] = True
                stack.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if g[i][j] == bg and not seen[i][j]:
                seen[i][j] = True
                stack.append((i, j))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g[nr][nc] == bg:
                seen[nr][nc] = True
                stack.append((nr, nc))
    return [[fill if (g[i][j] == bg and not seen[i][j]) else g[i][j] for j in range(w)] for i in range(h)]


def outline_objects(g: Grid, c: int) -> Grid:
    """Colour bg cells 4-adjacent to any non-bg cell."""
    bg = bg_color(g)
    h, w = shape(g)
    out = copy_grid(g)
    for i in range(h):
        for j in range(w):
            if g[i][j] != bg:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, q = i + dr, j + dc
                if 0 <= r < h and 0 <= q < w and g[r][q] != bg:
                    out[i][j] = c
                    break
    return out


def shift(g: Grid, di: int, dj: int) -> Grid:
    bg = bg_color(g)
    h, w = shape(g)
    out = [[bg] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            r, c = i + di, j + dj
            if 0 <= r < h and 0 <= c < w:
                out[r][c] = g[i][j]
    return out


def kron_self(g: Grid, invert: bool = False) -> Optional[Grid]:
    """Fractal: replace each non-bg cell (or each bg cell if invert) by a copy of the grid."""
    h, w = shape(g)
    if h * h > MAX_SIDE or w * w > MAX_SIDE:
        return None
    bg = 0 if any(v == 0 for row in g for v in row) else bg_color(g)
    out = [[bg] * (w * w) for _ in range(h * h)]
    for i in range(h):
        for j in range(w):
            on = (g[i][j] != bg) != invert
            if not on:
                continue
            for r in range(h):
                for c in range(w):
                    out[i * h + r][j * w + c] = g[r][c]
    return out


def most_common_1x1(g: Grid) -> Grid:
    return [[bg_color(g)]]


def most_common_nonbg_1x1(g: Grid) -> Optional[Grid]:
    bg = bg_color(g)
    cnt = Counter(v for row in g for v in row if v != bg)
    return [[cnt.most_common(1)[0][0]]] if cnt else None


def least_common_1x1(g: Grid) -> Optional[Grid]:
    cnt = color_counts(g)
    return [[min(cnt, key=lambda c: (cnt[c], c))]] if len(cnt) > 1 else None


def bbox_of_non_bg_color_grid(g: Grid) -> Optional[Grid]:
    """Crop to the bbox of the least common colour."""
    cnt = color_counts(g)
    if len(cnt) < 2:
        return None
    c = min(cnt, key=lambda k: (cnt[k], k))
    return crop_color(g, c)


# --------------------------------------------------------------------------- separator-line grids


def _separator_lines(g: Grid) -> Optional[tuple[int, list[int], list[int]]]:
    """Find full rows/cols of one colour that partition the grid. Returns (sep colour, rows, cols)."""
    h, w = shape(g)
    best = None
    for c in color_counts(g):
        rows = [i for i in range(h) if all(v == c for v in g[i])]
        cols = [j for j in range(w) if all(g[i][j] == c for i in range(h))]
        if not rows and not cols:
            continue
        if len(rows) == h or len(cols) == w:
            continue
        score = len(rows) + len(cols)
        if best is None or score > best[0]:
            best = (score, c, rows, cols)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _blocks(length: int, seps: list[int]) -> list[tuple[int, int]]:
    out = []
    start = 0
    for s in seps + [length]:
        if s > start:
            out.append((start, s))
        start = s + 1
    return out


def meta_grid(g: Grid, mode: str) -> Optional[Grid]:
    """Collapse a grid partitioned by separator lines into one colour per cell."""
    found = _separator_lines(g)
    if found is None:
        return None
    sep, rows, cols = found
    rb = _blocks(len(g), rows)
    cb = _blocks(len(g[0]), cols)
    if len(rb) * len(cb) < 2 or len(rb) > MAX_SIDE or len(cb) > MAX_SIDE:
        return None
    bg = bg_color(g)
    out = []
    for r0, r1 in rb:
        row = []
        for c0, c1 in cb:
            cnt = Counter(g[i][j] for i in range(r0, r1) for j in range(c0, c1))
            if mode == "majority":
                row.append(cnt.most_common(1)[0][0])
            elif mode == "nonbg":
                cnt.pop(bg, None)
                cnt.pop(sep, None)
                row.append(cnt.most_common(1)[0][0] if cnt else bg)
            elif mode == "count":
                row.append(min(9, sum(v for k, v in cnt.items() if k != bg and k != sep)))
            else:
                return None
        out.append(row)
    return out


def meta_cell_pick(g: Grid, how: str) -> Optional[Grid]:
    """Return the sub-grid cell that is unique / has most / fewest non-bg cells."""
    found = _separator_lines(g)
    if found is None:
        return None
    sep, rows, cols = found
    rb = _blocks(len(g), rows)
    cb = _blocks(len(g[0]), cols)
    if len(rb) * len(cb) < 2:
        return None
    bg = bg_color(g)
    cells = []
    for r0, r1 in rb:
        for c0, c1 in cb:
            sub = [row[c0:c1] for row in g[r0:r1]]
            cells.append(sub)
    if len({shape(c) for c in cells}) != 1:
        return None
    if how == "unique":
        cnt = Counter(hash_grid(c) for c in cells)
        uniq = [c for c in cells if cnt[hash_grid(c)] == 1]
        return uniq[0] if len(uniq) == 1 else None
    counts = [sum(1 for row in c for v in row if v != bg) for c in cells]
    if how == "most":
        return cells[counts.index(max(counts))]
    if how == "fewest":
        return cells[counts.index(min(counts))]
    if how == "most_colors":
        nc = [len({v for row in c for v in row if v != bg}) for c in cells]
        return cells[nc.index(max(nc))]
    return None


# --------------------------------------------------------------------------- op catalogue


def build_ops() -> list[tuple[str, Op, float]]:
    ops: list[tuple[str, Op, float]] = []
    for name, fn in GEOMETRY:
        ops.append((name, fn, 0.0 if name == "identity" else 1.0))
    for k in (2, 3, 4):
        ops.append((f"upscale{k}", (lambda k: lambda g: upscale(g, k))(k), 1.0))
    for k in (2, 3):
        ops.append((f"downscale{k}", (lambda k: lambda g: downscale(g, k))(k), 1.0))
        ops.append((f"downscale_maj{k}", (lambda k: lambda g: downscale_majority(g, k))(k), 1.3))
    for a, b in ((1, 2), (2, 1), (2, 2), (3, 3), (1, 3), (3, 1)):
        ops.append((f"tile{a}x{b}", (lambda a, b: lambda g: tile(g, a, b))(a, b), 1.2))
    ops += [
        ("mirror_h", mirror_h, 1.0),
        ("mirror_h_rev", mirror_h_rev, 1.0),
        ("mirror_v", mirror_v, 1.0),
        ("mirror_v_rev", mirror_v_rev, 1.0),
        ("mirror_4", mirror_4, 1.0),
        ("mirror_4_rev", mirror_4_rev, 1.0),
        ("rot_tile_4", rot_tile_4, 1.2),
        ("top_half", top_half, 1.0),
        ("bottom_half", bottom_half, 1.0),
        ("left_half", left_half, 1.0),
        ("right_half", right_half, 1.0),
        ("crop_nonbg", crop_nonbg, 1.0),
        ("crop_nonzero", crop_nonzero, 1.0),
        ("crop_rarest", bbox_of_non_bg_color_grid, 1.2),
        ("compress_bg", compress_bg_lines, 1.0),
        ("compress_dup", compress_dup_lines, 1.0),
        ("most_common_1x1", most_common_1x1, 1.5),
        ("most_common_nonbg_1x1", most_common_nonbg_1x1, 1.5),
        ("least_common_1x1", least_common_1x1, 1.5),
        ("kron_self", lambda g: kron_self(g, False), 1.3),
        ("kron_self_inv", lambda g: kron_self(g, True), 1.4),
        ("meta_majority", lambda g: meta_grid(g, "majority"), 1.3),
        ("meta_nonbg", lambda g: meta_grid(g, "nonbg"), 1.3),
        ("meta_count", lambda g: meta_grid(g, "count"), 1.5),
        ("meta_pick_unique", lambda g: meta_cell_pick(g, "unique"), 1.4),
        ("meta_pick_most", lambda g: meta_cell_pick(g, "most"), 1.4),
        ("meta_pick_fewest", lambda g: meta_cell_pick(g, "fewest"), 1.4),
        ("meta_pick_colors", lambda g: meta_cell_pick(g, "most_colors"), 1.5),
    ]
    for d in ("down", "up", "left", "right"):
        ops.append((f"gravity_{d}", (lambda d: lambda g: gravity(g, d))(d), 1.0))
    for name, di, dj in (("shift_down", 1, 0), ("shift_up", -1, 0), ("shift_right", 0, 1), ("shift_left", 0, -1)):
        ops.append((name, (lambda di, dj: lambda g: shift(g, di, dj))(di, dj), 1.0))
    for c in range(1, 10):
        ops.append((f"crop_color{c}", (lambda c: lambda g: crop_color(g, c))(c), 1.3))
        ops.append((f"crop_interior{c}", (lambda c: lambda g: crop_color_interior(g, c))(c), 1.4))
        ops.append((f"fill_enclosed{c}", (lambda c: lambda g: fill_enclosed(g, c))(c), 1.2))
        ops.append((f"outline{c}", (lambda c: lambda g: outline_objects(g, c))(c), 1.4))
    picks = (
        "largest",
        "smallest",
        "biggest_bbox",
        "densest",
        "sparsest",
        "unique_color",
        "unique_shape",
        "unique_size",
        "most_colors",
        "top_left",
        "bottom_right",
    )
    for diag, multi in ((False, False), (True, True)):
        tag = f"{'d' if diag else 'o'}{'m' if multi else 's'}"
        for how in picks:
            ops.append((f"crop_{how}_{tag}", make_obj_crop(how, diag, multi), 1.4))
            ops.append((f"keep_{how}_{tag}", make_obj_keep(how, diag, multi), 1.5))
            ops.append((f"remove_{how}_{tag}", make_obj_remove(how, diag, multi), 1.5))
    return ops


OPS = build_ops()
OP_BY_NAME = {name: fn for name, fn, _ in OPS}

# --------------------------------------------------------------------------- colour map


def fit_color_map(produced: list[Grid], expected: list[Grid]) -> Optional[dict[int, int]]:
    mapping: dict[int, int] = {}
    for p, e in zip(produced, expected):
        if shape(p) != shape(e):
            return None
        for pr, er in zip(p, e):
            for a, b in zip(pr, er):
                prev = mapping.get(a)
                if prev is None:
                    mapping[a] = b
                elif prev != b:
                    return None
    return mapping


def apply_color_map(g: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(v, v) for v in row] for row in g]


# --------------------------------------------------------------------------- program


class Program:
    __slots__ = ("name", "cost", "fn")

    def __init__(self, name: str, cost: float, fn: Op):
        self.name = name
        self.cost = cost
        self.fn = fn

    def run(self, g: Grid) -> Optional[Grid]:
        try:
            out = self.fn(g)
        except Exception:
            return None
        return out if out is not None and valid(out) else None


def fits(fn: Op, pairs: Pairs) -> bool:
    for inp, out in pairs:
        try:
            got = fn(inp)
        except Exception:
            return False
        if got is None or got != out:
            return False
    return True


# --------------------------------------------------------------------------- whole-grid search


def search_compositions(pairs: Pairs, deadline: float) -> list[Program]:
    inputs = [p for p, _ in pairs]
    expected = [o for _, o in pairs]
    found: list[Program] = []

    def consider(name: str, cost: float, fn: Op, produced: list[Grid]) -> None:
        if all(p == e for p, e in zip(produced, expected)):
            found.append(Program(name, cost, fn))
            return
        mapping = fit_color_map(produced, expected)
        if mapping is None or all(k == v for k, v in mapping.items()):
            return
        found.append(Program(f"{name}+cmap", cost + 0.5, (lambda fn, m: lambda g: (lambda r: None if r is None else apply_color_map(r, m))(fn(g)))(fn, mapping)))

    level1: list[tuple[str, Op, float, list[Grid]]] = []
    for name, fn, cost in OPS:
        produced = []
        ok = True
        for g in inputs:
            try:
                r = fn(g)
            except Exception:
                r = None
            if r is None or not valid(r):
                ok = False
                break
            produced.append(r)
        if not ok:
            continue
        level1.append((name, fn, cost, produced))
        consider(name, cost, fn, produced)

    if time.time() > deadline:
        return found

    exp0 = expected[0]
    sh0 = shape(exp0)
    for name1, fn1, cost1, prod1 in level1:
        if name1 == "identity":
            continue
        for name2, fn2, cost2 in OPS:
            if name2 == "identity":
                continue
            try:
                r0 = fn2(prod1[0])
            except Exception:
                continue
            if r0 is None or not valid(r0) or shape(r0) != sh0:
                continue
            # quick colour-consistency check on the first pair before doing the rest
            if r0 != exp0 and fit_color_map([r0], [exp0]) is None:
                continue
            produced = [r0]
            ok = True
            for g1 in prod1[1:]:
                try:
                    r = fn2(g1)
                except Exception:
                    r = None
                if r is None or not valid(r):
                    ok = False
                    break
                produced.append(r)
            if not ok:
                continue
            composed = (lambda f1, f2: lambda g: (lambda a: None if a is None else f2(a))(f1(g)))(fn1, fn2)
            consider(f"{name1}>{name2}", cost1 + cost2 + 0.5, composed, produced)
        if time.time() > deadline:
            break
    return found


# --------------------------------------------------------------------------- per-cell rules


def cell_context(g: Grid) -> dict:
    """Precompute per-cell features once; feature selectors index into these."""
    h, w = shape(g)
    bg = bg_color(g)
    # rays: first non-bg colour in each of 4 directions (or -1)
    ray_u = [[-1] * w for _ in range(h)]
    ray_d = [[-1] * w for _ in range(h)]
    ray_l = [[-1] * w for _ in range(h)]
    ray_r = [[-1] * w for _ in range(h)]
    for j in range(w):
        last = -1
        for i in range(h):
            ray_u[i][j] = last
            if g[i][j] != bg:
                last = g[i][j]
        last = -1
        for i in range(h - 1, -1, -1):
            ray_d[i][j] = last
            if g[i][j] != bg:
                last = g[i][j]
    for i in range(h):
        last = -1
        for j in range(w):
            ray_l[i][j] = last
            if g[i][j] != bg:
                last = g[i][j]
        last = -1
        for j in range(w - 1, -1, -1):
            ray_r[i][j] = last
            if g[i][j] != bg:
                last = g[i][j]
    row_colors = [tuple(sorted({v for v in row if v != bg})) for row in g]
    col_colors = [tuple(sorted({g[i][j] for i in range(h) if g[i][j] != bg})) for j in range(w)]
    row_uniform = [len(set(row)) == 1 for row in g]
    col_uniform = [len({g[i][j] for i in range(h)}) == 1 for j in range(w)]
    row_cnt = [sum(1 for v in row if v != bg) for row in g]
    col_cnt = [sum(1 for i in range(h) if g[i][j] != bg) for j in range(w)]
    # component sizes (4-conn, same colour, incl. bg components)
    comp_size = [[0] * w for _ in range(h)]
    seen = [[False] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if seen[i][j]:
                continue
            color = g[i][j]
            stack = [(i, j)]
            seen[i][j] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            for r, c in cells:
                comp_size[r][c] = len(cells)
    return {
        "g": g,
        "h": h,
        "w": w,
        "bg": bg,
        "ray": (ray_u, ray_d, ray_l, ray_r),
        "row_colors": row_colors,
        "col_colors": col_colors,
        "row_uniform": row_uniform,
        "col_uniform": col_uniform,
        "row_cnt": row_cnt,
        "col_cnt": col_cnt,
        "comp_size": comp_size,
    }


def _n4(ctx, i, j):
    g, h, w = ctx["g"], ctx["h"], ctx["w"]
    return (
        g[i - 1][j] if i > 0 else -1,
        g[i + 1][j] if i < h - 1 else -1,
        g[i][j - 1] if j > 0 else -1,
        g[i][j + 1] if j < w - 1 else -1,
    )


def _n8(ctx, i, j):
    g, h, w = ctx["g"], ctx["h"], ctx["w"]
    out = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            r, c = i + di, j + dj
            out.append(g[r][c] if 0 <= r < h and 0 <= c < w else -1)
    return tuple(out)


def _rays(ctx, i, j):
    u, d, l, r = ctx["ray"]
    return (u[i][j], d[i][j], l[i][j], r[i][j])


def _n4_count(ctx, i, j):
    bg = ctx["bg"]
    return sum(1 for v in _n4(ctx, i, j) if v != -1 and v != bg)


def _n8_count(ctx, i, j):
    bg = ctx["bg"]
    return sum(1 for v in _n8(ctx, i, j) if v != -1 and v != bg)


def _n4_bg(ctx, i, j):
    bg = ctx["bg"]
    return tuple(-1 if v == -1 else int(v != bg) for v in _n4(ctx, i, j))


def _n8_bg(ctx, i, j):
    bg = ctx["bg"]
    return tuple(-1 if v == -1 else int(v != bg) for v in _n8(ctx, i, j))


def _rays_bg(ctx, i, j):
    return tuple(int(v != -1) for v in _rays(ctx, i, j))


def _isbg(ctx, i, j):
    return int(ctx["g"][i][j] != ctx["bg"])


# (name, feature fn, cost). Features starting with "color" key on the exact cell colour;
# the others are colour-free and can generalise to unseen colours (output may be "keep").
FEATURES: list[tuple[str, Callable, float]] = [
    ("color", lambda c, i, j: (c["g"][i][j],), 0.0),
    ("color+border", lambda c, i, j: (c["g"][i][j], i == 0, j == 0, i == c["h"] - 1, j == c["w"] - 1), 0.2),
    ("color+mod2", lambda c, i, j: (c["g"][i][j], i % 2, j % 2), 0.2),
    ("color+n4cnt", lambda c, i, j: (c["g"][i][j], _n4_count(c, i, j)), 0.2),
    ("color+n8cnt", lambda c, i, j: (c["g"][i][j], _n8_count(c, i, j)), 0.25),
    ("color+n4", lambda c, i, j: (c["g"][i][j],) + _n4(c, i, j), 0.3),
    ("color+rays", lambda c, i, j: (c["g"][i][j],) + _rays(c, i, j), 0.3),
    ("color+rowcol", lambda c, i, j: (c["g"][i][j], c["row_colors"][i], c["col_colors"][j]), 0.35),
    ("color+rowcolhas", lambda c, i, j: (c["g"][i][j], bool(c["row_colors"][i]), bool(c["col_colors"][j])), 0.3),
    ("color+rowcoluniform", lambda c, i, j: (c["g"][i][j], c["row_uniform"][i], c["col_uniform"][j]), 0.3),
    ("color+rowcolcnt", lambda c, i, j: (c["g"][i][j], c["row_cnt"][i], c["col_cnt"][j]), 0.45),
    ("color+compsize", lambda c, i, j: (c["g"][i][j], c["comp_size"][i][j]), 0.3),
    ("color+n8", lambda c, i, j: (c["g"][i][j],) + _n8(c, i, j), 0.4),
    ("color+rays+n4", lambda c, i, j: (c["g"][i][j],) + _rays(c, i, j) + _n4(c, i, j), 0.5),
    ("color+mod3", lambda c, i, j: (c["g"][i][j], i % 3, j % 3), 0.35),
    ("color+row", lambda c, i, j: (c["g"][i][j], i), 0.6),
    ("color+col", lambda c, i, j: (c["g"][i][j], j), 0.6),
    ("color+rowrev", lambda c, i, j: (c["g"][i][j], c["h"] - 1 - i), 0.6),
    ("color+colrev", lambda c, i, j: (c["g"][i][j], c["w"] - 1 - j), 0.6),
    ("color+n8+rays", lambda c, i, j: (c["g"][i][j],) + _n8(c, i, j) + _rays(c, i, j), 0.7),
    # colour-free
    ("isbg", lambda c, i, j: (_isbg(c, i, j),), 0.15),
    ("isbg+border", lambda c, i, j: (_isbg(c, i, j), i == 0, j == 0, i == c["h"] - 1, j == c["w"] - 1), 0.3),
    ("isbg+mod2", lambda c, i, j: (_isbg(c, i, j), i % 2, j % 2), 0.3),
    ("isbg+n4cnt", lambda c, i, j: (_isbg(c, i, j), _n4_count(c, i, j)), 0.3),
    ("isbg+n8cnt", lambda c, i, j: (_isbg(c, i, j), _n8_count(c, i, j)), 0.35),
    ("isbg+n4bg", lambda c, i, j: (_isbg(c, i, j),) + _n4_bg(c, i, j), 0.4),
    ("isbg+n8bg", lambda c, i, j: (_isbg(c, i, j),) + _n8_bg(c, i, j), 0.5),
    ("isbg+raysbg", lambda c, i, j: (_isbg(c, i, j),) + _rays_bg(c, i, j), 0.4),
    ("isbg+rowcoluniform", lambda c, i, j: (_isbg(c, i, j), c["row_uniform"][i], c["col_uniform"][j]), 0.35),
    ("rowcoluniform", lambda c, i, j: (c["row_uniform"][i], c["col_uniform"][j]), 0.3),
    ("isbg+rowcolhas", lambda c, i, j: (_isbg(c, i, j), bool(c["row_colors"][i]), bool(c["col_colors"][j])), 0.35),
    ("isbg+compsize", lambda c, i, j: (_isbg(c, i, j), c["comp_size"][i][j]), 0.4),
    ("isbg+rays", lambda c, i, j: (_isbg(c, i, j),) + _rays(c, i, j), 0.45),
    ("isbg+n4", lambda c, i, j: (_isbg(c, i, j),) + _n4(c, i, j), 0.45),
]

KEEP = ("keep",)


def learn_cell_rules(pairs: Pairs, pre_name: str, pre: Op, pre_cost: float, test_inputs: list[Grid]) -> tuple[list[Program], Optional[Program]]:
    """Return (exact programs, best partial fallback) for same-shape pairs after `pre`."""
    pre_in = []
    for inp, out in pairs:
        try:
            r = pre(inp)
        except Exception:
            return [], None
        if r is None or not valid(r) or shape(r) != shape(out):
            return [], None
        pre_in.append(r)
    ctxs = [cell_context(g) for g in pre_in]
    test_pre = []
    for t in test_inputs:
        try:
            r = pre(t)
        except Exception:
            r = None
        if r is None or not valid(r):
            return [], None
        test_pre.append(r)
    test_ctx = [cell_context(g) for g in test_pre]

    programs: list[Program] = []
    best_partial: Optional[tuple[float, Program]] = None
    for fname, feat, fcost in FEATURES:
        # key -> (set of output colours, all_keep flag)
        obs: dict = {}
        for ctx, (_, out) in zip(ctxs, pairs):
            g = ctx["g"]
            for i in range(ctx["h"]):
                row = out[i]
                grow = g[i]
                for j in range(ctx["w"]):
                    key = feat(ctx, i, j)
                    o = row[j]
                    rec = obs.get(key)
                    if rec is None:
                        obs[key] = [{o}, grow[j] == o]
                    else:
                        rec[0].add(o)
                        rec[1] = rec[1] and grow[j] == o
        table: dict = {}
        ok = True
        for key, (outs, all_keep) in obs.items():
            if len(outs) == 1:
                table[key] = next(iter(outs))
            elif all_keep:
                table[key] = KEEP
            else:
                ok = False
                break
        if not ok:
            continue
        if fname.startswith("isbg") or fname == "rowcoluniform":
            # colour-free features: prefer "keep" whenever it explains the data,
            # so unseen colours pass through instead of being repainted.
            for key, (outs, all_keep) in obs.items():
                if all_keep:
                    table[key] = KEEP
        # coverage on test inputs
        total = 0
        missing = 0
        for ctx in test_ctx:
            for i in range(ctx["h"]):
                for j in range(ctx["w"]):
                    total += 1
                    if feat(ctx, i, j) not in table:
                        missing += 1

        def make(feat=feat, table=table, pre=pre):
            def run(g: Grid) -> Optional[Grid]:
                r = pre(g)
                if r is None:
                    return None
                ctx = cell_context(r)
                out = []
                for i in range(ctx["h"]):
                    row = []
                    rrow = r[i]
                    for j in range(ctx["w"]):
                        t = table.get(feat(ctx, i, j), KEEP)
                        row.append(rrow[j] if t is KEEP else t)
                    out.append(row)
                return out

            return run

        cost = 1.0 + pre_cost + fcost
        name = f"{pre_name}>cells[{fname}]"
        if missing == 0:
            programs.append(Program(name, cost, make()))
        else:
            frac = missing / max(total, 1)
            if frac < 0.5 and (best_partial is None or frac < best_partial[0]):
                best_partial = (frac, Program(name + "~partial", cost + 5 + frac, make()))
    return programs, (best_partial[1] if best_partial else None)


# --------------------------------------------------------------------------- per-object recolour

OBJ_PROPS: list[tuple[str, Callable]] = [
    ("size", lambda o, objs, g, bg: o["size"]),
    ("color", lambda o, objs, g, bg: o["color"]),
    ("rank_desc", lambda o, objs, g, bg: sorted({x["size"] for x in objs}, reverse=True).index(o["size"])),
    ("rank_asc", lambda o, objs, g, bg: sorted({x["size"] for x in objs}).index(o["size"])),
    ("is_max", lambda o, objs, g, bg: o["size"] == max(x["size"] for x in objs)),
    ("is_min", lambda o, objs, g, bg: o["size"] == min(x["size"] for x in objs)),
    ("hw", lambda o, objs, g, bg: (o["h"], o["w"])),
    ("holes", lambda o, objs, g, bg: obj_holes(g, o, bg)),
    ("color+size", lambda o, objs, g, bg: (o["color"], o["size"])),
    ("touch_border", lambda o, objs, g, bg: o["bbox"][0] == 0 or o["bbox"][1] == 0 or o["bbox"][2] == len(g) - 1 or o["bbox"][3] == len(g[0]) - 1),
    ("shape", lambda o, objs, g, bg: tuple(sorted((r - o["bbox"][0], c - o["bbox"][1]) for r, c in o["cells"]))),
    ("count_same_color", lambda o, objs, g, bg: sum(1 for x in objs if x["color"] == o["color"])),
    ("is_square_full", lambda o, objs, g, bg: o["h"] == o["w"] and o["size"] == o["h"] * o["w"]),
    ("is_rect_full", lambda o, objs, g, bg: o["size"] == o["h"] * o["w"]),
]


def learn_object_recolor(pairs: Pairs, pre_name: str, pre: Op, pre_cost: float, test_inputs: list[Grid]) -> list[Program]:
    pre_in = []
    for inp, out in pairs:
        try:
            r = pre(inp)
        except Exception:
            return []
        if r is None or not valid(r) or shape(r) != shape(out):
            return []
        pre_in.append(r)
    programs: list[Program] = []
    for diag in (False, True):
        for multi in (False, True):
            per_pair = []
            ok = True
            for g, (_, out) in zip(pre_in, pairs):
                bg = bg_color(g)
                objs = components(g, bg, diag, multi)
                if not objs:
                    ok = False
                    break
                in_obj = set()
                for o in objs:
                    in_obj.update(o["cells"])
                # cells outside objects must be unchanged
                for i, row in enumerate(g):
                    for j, v in enumerate(row):
                        if (i, j) not in in_obj and out[i][j] != v:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
                targets = []
                for o in objs:
                    cols = {out[r][c] for r, c in o["cells"]}
                    if len(cols) != 1:
                        ok = False
                        break
                    targets.append(cols.pop())
                if not ok:
                    break
                per_pair.append((g, bg, objs, targets))
            if not ok:
                continue
            for pname, prop in OBJ_PROPS:
                table: dict = {}
                good = True
                for g, bg, objs, targets in per_pair:
                    for o, t in zip(objs, targets):
                        try:
                            key = prop(o, objs, g, bg)
                        except Exception:
                            good = False
                            break
                        prev = table.get(key)
                        if prev is None:
                            table[key] = t
                        elif prev != t:
                            good = False
                            break
                    if not good:
                        break
                if not good:
                    continue
                if all(k == v for k, v in table.items() if isinstance(k, int)) and pname == "color":
                    continue  # identity recolour, useless

                def make(prop=prop, table=table, pre=pre, diag=diag, multi=multi):
                    def run(g: Grid) -> Optional[Grid]:
                        r = pre(g)
                        if r is None:
                            return None
                        bg = bg_color(r)
                        objs = components(r, bg, diag, multi)
                        out = copy_grid(r)
                        for o in objs:
                            key = prop(o, objs, r, bg)
                            if key not in table:
                                return None
                            t = table[key]
                            for rr, cc in o["cells"]:
                                out[rr][cc] = t
                        return out

                    return run

                fn = make()
                # must be defined on all test inputs
                if all(fn(t) is not None for t in test_inputs):
                    programs.append(Program(f"{pre_name}>objrecolor[{pname},{'d' if diag else 'o'}{'m' if multi else 's'}]", 1.3 + pre_cost, fn))
    return programs


# --------------------------------------------------------------------------- symmetry / periodic completion


def _sym_candidates(h: int, w: int):
    """Yield (name, mapper) where mapper(i, j) -> (i', j') or None."""
    for a in range(1, 2 * w - 2):
        yield f"fh{a}", (lambda a: lambda i, j: (i, a - j) if 0 <= a - j < w else None)(a)
    for b in range(1, 2 * h - 2):
        yield f"fv{b}", (lambda b: lambda i, j: (b - i, j) if 0 <= b - i < h else None)(b)
    for d in range(-(w - 1), h):
        yield f"tr{d}", (lambda d: lambda i, j: (j + d, i - d) if 0 <= j + d < h and 0 <= i - d < w else None)(d)
    for s in range(0, h + w - 1):
        yield f"at{s}", (lambda s: lambda i, j: (s - j, s - i) if 0 <= s - j < h and 0 <= s - i < w else None)(s)
    yield "r180", lambda i, j: (h - 1 - i, w - 1 - j)
    if h == w:
        yield "r90", lambda i, j: (j, w - 1 - i)


def symmetry_fill(g: Grid, mask: int) -> Optional[Grid]:
    h, w = shape(g)
    masked = [(i, j) for i in range(h) for j in range(w) if g[i][j] == mask]
    if not masked:
        return None
    n_known = h * w - len(masked)
    accepted = []
    for name, mp in _sym_candidates(h, w):
        agree = 0
        bad = False
        for i in range(h):
            row = g[i]
            for j in range(w):
                v = row[j]
                if v == mask:
                    continue
                q = mp(i, j)
                if q is None:
                    continue
                u = g[q[0]][q[1]]
                if u == mask:
                    continue
                if u != v:
                    bad = True
                    break
                agree += 1
            if bad:
                break
        if not bad and agree >= max(4, 0.5 * n_known):
            accepted.append(mp)
    if not accepted:
        return None
    out = copy_grid(g)
    remaining = set(masked)
    changed = True
    while remaining and changed:
        changed = False
        for i, j in list(remaining):
            for mp in accepted:
                q = mp(i, j)
                if q is None:
                    continue
                u = out[q[0]][q[1]]
                if u != mask:
                    out[i][j] = u
                    remaining.discard((i, j))
                    changed = True
                    break
    if remaining:
        return None
    return out


def periodic_fill(g: Grid, mask: int) -> Optional[Grid]:
    h, w = shape(g)
    masked = [(i, j) for i in range(h) for j in range(w) if g[i][j] == mask]
    if not masked:
        return None
    periods = sorted(((py, px) for py in range(1, h + 1) for px in range(1, w + 1) if py * px < h * w), key=lambda p: (p[0] * p[1], p))
    for py, px in periods:
        table: dict = {}
        ok = True
        for i in range(h):
            row = g[i]
            for j in range(w):
                v = row[j]
                if v == mask:
                    continue
                key = (i % py, j % px)
                prev = table.get(key)
                if prev is None:
                    table[key] = v
                elif prev != v:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        out = copy_grid(g)
        for i, j in masked:
            v = table.get((i % py, j % px))
            if v is None:
                ok = False
                break
            out[i][j] = v
        if ok:
            return out
    return None


def mask_bbox_crop(g: Grid, mask: int, filled: Grid) -> Optional[Grid]:
    cells = [(i, j) for i, row in enumerate(g) for j, v in enumerate(row) if v == mask]
    return crop_cells(filled, cells)


def learn_completion(pairs: Pairs, test_inputs: list[Grid]) -> list[Program]:
    programs: list[Program] = []
    same = all(shape(i) == shape(o) for i, o in pairs)
    # candidate mask colours: for same-shape, colour of changed cells; else any colour in every input
    if same:
        masks = set()
        for inp, out in pairs:
            diff = {inp[i][j] for i in range(len(inp)) for j in range(len(inp[0])) if inp[i][j] != out[i][j]}
            if len(diff) != 1:
                return []
            masks |= diff
        if len(masks) != 1:
            return []
        mask = masks.pop()
        for fname, filler in (("sym", symmetry_fill), ("period", periodic_fill)):
            fn = (lambda f, m: lambda g: f(g, m))(filler, mask)
            if fits(fn, pairs) and all(fn(t) is not None for t in test_inputs):
                programs.append(Program(f"{fname}_fill[{mask}]", 1.5, fn))
    else:
        common = None
        for inp, _ in pairs:
            cs = set(v for row in inp for v in row)
            common = cs if common is None else common & cs
        for mask in sorted(common or ()):
            for fname, filler in (("sym", symmetry_fill), ("period", periodic_fill)):
                def fn(g, f=filler, m=mask):
                    filled = f(g, m)
                    return None if filled is None else mask_bbox_crop(g, m, filled)

                if fits(fn, pairs) and all(fn(t) is not None for t in test_inputs):
                    programs.append(Program(f"{fname}_patch[{mask}]", 1.7, fn))
    return programs


# --------------------------------------------------------------------------- split & combine


def split_parts(g: Grid, n: int, axis: int) -> Optional[list[Grid]]:
    h, w = shape(g)
    length = h if axis == 0 else w
    for sep in (0, 1):
        total_sep = sep * (n - 1)
        if (length - total_sep) % n or length - total_sep <= 0:
            continue
        size = (length - total_sep) // n
        parts = []
        ok = True
        for k in range(n):
            start = k * (size + sep)
            if axis == 0:
                part = [r[:] for r in g[start : start + size]]
                if sep and k < n - 1:
                    line = g[start + size]
                    if len(set(line)) != 1:
                        ok = False
                        break
            else:
                part = [r[start : start + size] for r in g]
                if sep and k < n - 1:
                    line = [r[start + size] for r in g]
                    if len(set(line)) != 1:
                        ok = False
                        break
            parts.append(part)
        if ok:
            return parts
    return None


def learn_split_combine(pairs: Pairs, test_inputs: list[Grid]) -> list[Program]:
    programs: list[Program] = []
    for n in (2, 3, 4):
        for axis in (0, 1):
            per = []
            ok = True
            for inp, out in pairs:
                parts = split_parts(inp, n, axis)
                if parts is None or shape(parts[0]) != shape(out):
                    ok = False
                    break
                per.append((inp, parts, out))
            if not ok:
                continue
            for mode in ("bool", "color"):
                table: dict = {}
                good = True
                for inp, parts, out in per:
                    bg = bg_color(inp)
                    ph, pw = shape(parts[0])
                    for i in range(ph):
                        for j in range(pw):
                            if mode == "bool":
                                key = tuple(p[i][j] != bg for p in parts)
                            else:
                                key = tuple(p[i][j] for p in parts)
                            prev = table.get(key)
                            if prev is None:
                                table[key] = out[i][j]
                            elif prev != out[i][j]:
                                good = False
                                break
                        if not good:
                            break
                    if not good:
                        break
                if not good:
                    continue

                def make(n=n, axis=axis, mode=mode, table=table):
                    def run(g: Grid) -> Optional[Grid]:
                        parts = split_parts(g, n, axis)
                        if parts is None:
                            return None
                        bg = bg_color(g)
                        ph, pw = shape(parts[0])
                        out = []
                        for i in range(ph):
                            row = []
                            for j in range(pw):
                                key = tuple(p[i][j] != bg for p in parts) if mode == "bool" else tuple(p[i][j] for p in parts)
                                if key not in table:
                                    return None
                                row.append(table[key])
                            out.append(row)
                        return out

                    return run

                fn = make()
                if all(fn(t) is not None for t in test_inputs):
                    programs.append(Program(f"split{n}{'v' if axis else 'h'}[{mode}]", 1.5 if mode == "bool" else 1.8, fn))
    return programs


# --------------------------------------------------------------------------- transform tiling


def learn_tile_transforms(pairs: Pairs, test_inputs: list[Grid]) -> list[Program]:
    """Output is an a x b layout of blocks, each block a fixed geometric transform of the input (or blank)."""
    layout = None
    for inp, out in pairs:
        ih, iw = shape(inp)
        oh, ow = shape(out)
        if ih == 0 or oh % ih or ow % iw:
            return []
        a, b = oh // ih, ow // iw
        if a * b < 2 or a > 4 or b > 4:
            return []
        if layout is None:
            layout = (a, b)
        elif layout != (a, b):
            return []
    if layout is None:
        return []
    a, b = layout
    assignment: dict = {}
    for bi in range(a):
        for bj in range(b):
            chosen = None
            for name, fn in GEOMETRY + [("blank", None)]:
                ok = True
                for inp, out in pairs:
                    ih, iw = shape(inp)
                    block = [row[bj * iw : (bj + 1) * iw] for row in out[bi * ih : (bi + 1) * ih]]
                    if fn is None:
                        bg = bg_color(inp)
                        if any(v != bg for row in block for v in row):
                            ok = False
                            break
                    else:
                        t = fn(inp)
                        if shape(t) != shape(block) or t != block:
                            ok = False
                            break
                if ok:
                    chosen = (name, fn)
                    break
            if chosen is None:
                return []
            assignment[(bi, bj)] = chosen
    if all(name == "identity" for name, _ in assignment.values()):
        return []  # plain tiling is already an op

    def run(g: Grid) -> Optional[Grid]:
        h, w = shape(g)
        if h * a > MAX_SIDE or w * b > MAX_SIDE:
            return None
        bg = bg_color(g)
        out = [[bg] * (w * b) for _ in range(h * a)]
        for (bi, bj), (name, fn) in assignment.items():
            if fn is None:
                continue
            t = fn(g)
            if shape(t) != (h, w):
                return None
            for i in range(h):
                out[bi * h + i][bj * w : (bj + 1) * w] = t[i]
        return out

    if all(run(t) is not None for t in test_inputs):
        return [Program(f"tile_transforms{a}x{b}", 1.4, run)]
    return []


# --------------------------------------------------------------------------- driver


def train_pairs_of(task: dict) -> Pairs:
    return [(p["input"], p["output"]) for p in task["train"]]


PRE_OPS_FOR_RULES = [
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "transpose",
    "anti_transpose",
    "upscale2",
    "upscale3",
    "downscale2",
    "downscale3",
    "crop_nonbg",
    "compress_bg",
    "tile1x2",
    "tile2x1",
    "tile2x2",
    "mirror_h",
    "mirror_v",
    "mirror_4",
    "top_half",
    "left_half",
    "gravity_down",
    "meta_majority",
    "meta_nonbg",
    "meta_pick_unique",
]


def solve_task(task: dict, time_limit: float = 20.0) -> dict:
    """Return {"attempts": [[a1, a2], ...] per test input, "programs": [names]}."""
    start = time.time()
    deadline = start + time_limit
    pairs = train_pairs_of(task)
    test_inputs = [t["input"] for t in task["test"]]
    programs: list[Program] = []
    partial: Optional[Program] = None

    # constant output
    outs = [o for _, o in pairs]
    if all(o == outs[0] for o in outs):
        const = copy_grid(outs[0])
        programs.append(Program("constant", 0.5, lambda g, c=const: copy_grid(c)))

    programs += search_compositions(pairs, deadline - time_limit * 0.4)

    if time.time() < deadline:
        programs += learn_completion(pairs, test_inputs)
    if time.time() < deadline:
        programs += learn_split_combine(pairs, test_inputs)
    if time.time() < deadline:
        programs += learn_tile_transforms(pairs, test_inputs)

    for pre_name in PRE_OPS_FOR_RULES:
        if time.time() > deadline:
            break
        pre = OP_BY_NAME[pre_name]
        pre_cost = 0.0 if pre_name == "identity" else 1.0
        exact, part = learn_cell_rules(pairs, pre_name, pre, pre_cost, test_inputs)
        programs += exact
        if part is not None and (partial is None or part.cost < partial.cost):
            partial = part
        programs += learn_object_recolor(pairs, pre_name, pre, pre_cost, test_inputs)

    programs.sort(key=lambda p: p.cost)
    voters = programs[:MAX_VOTERS]

    attempts = []
    for t in test_inputs:
        # consensus: every consistent program votes for its prediction, weighted by simplicity
        votes: dict = {}
        for p in voters:
            out = p.run(t)
            if out is None:
                continue
            hsh = hash_grid(out)
            rec = votes.get(hsh)
            weight = 1.0 / (1.0 + p.cost)
            if rec is None:
                votes[hsh] = [weight, out, p.name, p.cost]
            else:
                rec[0] += weight
                if p.cost < rec[3]:
                    rec[2], rec[3] = p.name, p.cost
        ranked = sorted(votes.values(), key=lambda r: (-r[0], r[3]))
        preds: list[Grid] = [r[1] for r in ranked[:2]]
        names: list[str] = [f"{r[2]}(v={r[0]:.2f})" for r in ranked[:2]]
        seen = {hash_grid(p) for p in preds}
        if len(preds) < 2 and partial is not None:
            out = partial.run(t)
            if out is not None and hash_grid(out) not in seen:
                preds.append(out)
                names.append(partial.name)
                seen.add(hash_grid(out))
        if not preds:
            preds.append(copy_grid(t))
            names.append("identity_fallback")
        if len(preds) < 2:
            # second guess: keep it cheap but different when possible
            alt = flip_h(preds[0]) if shape(preds[0])[1] > 1 else copy_grid(preds[0])
            if hash_grid(alt) == hash_grid(preds[0]):
                alt = copy_grid(preds[0])
            preds.append(alt)
            names.append("alt_" + names[0])
        attempts.append((preds[0], preds[1], names))
    return {
        "attempts": [(a, b) for a, b, _ in attempts],
        "names": [n for _, _, n in attempts],
        "seconds": time.time() - start,
        "n_programs": len(programs),
    }


def predict_task(task: dict, time_limit: float = 20.0) -> list[dict]:
    try:
        res = solve_task(task, time_limit)
        return [{"attempt_1": a, "attempt_2": b} for a, b in res["attempts"]]
    except Exception:
        return [{"attempt_1": copy_grid(t["input"]), "attempt_2": copy_grid(t["input"])} for t in task["test"]]


def score_task(task: dict, solutions: Optional[list[Grid]] = None, time_limit: float = 20.0) -> dict:
    res = solve_task(task, time_limit)
    per_test = []
    solved = True
    for k, (test_pair, (a1, a2)) in enumerate(zip(task["test"], res["attempts"])):
        truth = test_pair.get("output")
        if truth is None and solutions is not None and k < len(solutions):
            truth = solutions[k]
        hit = truth is not None and (a1 == truth or a2 == truth)
        solved = solved and hit
        per_test.append(
            {
                "hit": hit,
                "names": res["names"][k],
                "attempt_1": a1,
                "attempt_2": a2,
                "truth": truth,
                "input": test_pair["input"],
            }
        )
    return {"solved": solved, "per_test": per_test, "seconds": res["seconds"], "n_programs": res["n_programs"]}


def build_submission(challenges: dict[str, dict], time_limit: float = 20.0, log_every: int = 0) -> dict[str, list[dict]]:
    submission = {}
    for n, (task_id, task) in enumerate(challenges.items(), start=1):
        submission[task_id] = predict_task(task, time_limit)
        if log_every and n % log_every == 0:
            print(f"{n}/{len(challenges)} tasks")
    return submission


def flatten_challenges(payload: dict | list, stem: str | None = None) -> dict[str, dict]:
    if isinstance(payload, list):
        raise TypeError("expected a task dict or a mapping of task_id -> task")
    if "train" in payload and "test" in payload:
        return {stem or "task": payload}
    return payload


# ---- notebook driver ----
import json
import os
import time
from pathlib import Path

TIME_LIMIT_PER_TASK = float(os.environ.get("ARC_TIME_LIMIT", "30"))

# Kaggle mounts the competition data under /kaggle/input/<competition>/.
# ARC_INPUT_DIR lets the same notebook run locally against data/kaggle/.
input_dir = Path(os.environ.get("ARC_INPUT_DIR", "/kaggle/input"))
candidates = []
if input_dir.exists():
    # Kaggle has used both /kaggle/input/<slug>/ and /kaggle/input/competitions/<slug>/; search recursively.
    candidates.extend(p for p in input_dir.rglob("*.json") if "test" in p.name and "challenges" in p.name)
repo_examples = Path.cwd() / "data" / "examples"


def load_challenges():
    for path in sorted(candidates):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data and "train" not in data:
            print("using", path, "tasks", len(data))
            return data
    tasks = {}
    if repo_examples.exists():
        for file in sorted(repo_examples.glob("*.json")):
            tasks[file.stem] = json.loads(file.read_text())
        print("using local examples", repo_examples, "n=", len(tasks))
        return tasks
    if input_dir.exists():
        for p in sorted(input_dir.rglob("*"))[:200]:
            print("  ", p)
    raise FileNotFoundError("no ARC challenges file found")


challenges = load_challenges()
t0 = time.time()
submission = {}
for n, (task_id, task) in enumerate(challenges.items(), start=1):
    submission[task_id] = predict_task(task, time_limit=TIME_LIMIT_PER_TASK)
    if n % 20 == 0 or n == len(challenges):
        print(f"{n}/{len(challenges)} tasks, {time.time() - t0:.0f}s", flush=True)

# Every task id must be present with both attempts for every test input.
for task_id, task in challenges.items():
    preds = submission.setdefault(task_id, [])
    while len(preds) < len(task["test"]):
        g = task["test"][len(preds)]["input"]
        preds.append({"attempt_1": [row[:] for row in g], "attempt_2": [row[:] for row in g]})
    for k, p in enumerate(preds):
        for key in ("attempt_1", "attempt_2"):
            if not valid(p.get(key)):
                p[key] = [row[:] for row in task["test"][k]["input"]]

out = Path(os.environ.get("ARC_OUTPUT", "/kaggle/working/submission.json"))
if not out.parent.exists():
    out = Path("demo/output/kaggle_submission.json")
    out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(submission, separators=(",", ":")))
print("wrote", out, "tasks", len(submission), f"total {time.time() - t0:.0f}s")
