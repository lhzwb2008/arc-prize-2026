#!/usr/bin/env python3
"""Helpers for the "state the rule, then draw the grid" format.

Pure Python (no torch) so it can be unit-tested anywhere.

Answer layout used by SFT and TTT in --rule-mode:

    Concepts: gravity, collision          <- optional line
    Rule: Every coloured pixel falls ...  <- one or two sentences
    Output:
    0123
    4567

Natural-language rules do not commute with our augmentations:
  * colour permutation changes the colour *names* in the text,
  * dihedral transforms change direction words (left/right, rows/columns, ...).
`rewrite_rule` fixes what can be fixed and refuses (returns None) otherwise, so
callers can restrict augmentation instead of training on a wrong rule.
"""

from __future__ import annotations

import re
from typing import Iterable

# ---------------------------------------------------------------------------
# Answer block
# ---------------------------------------------------------------------------
OUTPUT_MARK = "Output:"
RULE_REQUEST = "State the rule, then the output."
MAX_RULE_CHARS = 420


def clean_text(s: str, max_chars: int = MAX_RULE_CHARS) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    # end on a sentence boundary if there is one in the last third
    m = max(cut.rfind(". "), cut.rfind("; "))
    if m > max_chars * 0.6:
        cut = cut[: m + 1]
    return cut.strip()


def format_rule_block(rule: str, concepts: Iterable[str] | None = None) -> str:
    """Text placed at the start of the final assistant turn (ends with 'Output:\\n')."""
    lines = []
    concepts = [c.strip() for c in (concepts or []) if c and c.strip()]
    if concepts:
        lines.append("Concepts: " + ", ".join(concepts))
    lines.append("Rule: " + clean_text(rule))
    lines.append(OUTPUT_MARK)
    return "\n".join(lines) + "\n"


# 'Output:' on its own line, but tolerate a grid starting on the same line
_OUT_RE = re.compile(r"^[ \t]*Output\s*:[ \t]*", re.I | re.M)


def split_rule_output(text: str) -> tuple[str, str]:
    """Split generated text into (rule_part, grid_part).

    The grid part is what should go to parse_grid(); rule part may be ''.
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    matches = list(_OUT_RE.finditer(text))
    if not matches:
        return "", text
    m = matches[-1]
    return text[: m.start()].strip(), text[m.end():]


def parse_rule_block(rule_part: str) -> tuple[list[str], str]:
    """'Concepts: a, b\\nRule: xyz' -> (['a','b'], 'xyz')."""
    concepts: list[str] = []
    rule = rule_part.strip()
    m = re.search(r"Concepts?\s*:\s*(.*)", rule_part, re.I)
    if m:
        concepts = [c.strip() for c in m.group(1).split(",") if c.strip()]
    m = re.search(r"Rule\s*:\s*(.*)", rule_part, re.I | re.S)
    if m:
        rule = clean_text(m.group(1))
    return concepts, rule


# ---------------------------------------------------------------------------
# BARC-style sources ("# concepts:" / "# description:" comment header)
# ---------------------------------------------------------------------------
_CONCEPTS_RE = re.compile(r"#\s*concepts?\s*:\s*\n((?:[ \t]*#.*\n)+)", re.I)
_DESC_RE = re.compile(r"#\s*description\s*:\s*\n((?:[ \t]*#.*\n)+)", re.I)


def _strip_comment_lines(block: str) -> str:
    return " ".join(re.sub(r"^\s*#\s?", "", ln).strip() for ln in block.splitlines()).strip()


def parse_barc_source(source: str) -> tuple[list[str], str]:
    concepts: list[str] = []
    m = _CONCEPTS_RE.search(source)
    if m:
        concepts = [c.strip().lower() for c in _strip_comment_lines(m.group(1)).split(",") if c.strip()]
    desc = ""
    m = _DESC_RE.search(source)
    if m:
        desc = _strip_comment_lines(m.group(1))
    return concepts, clean_text(desc)


# ---------------------------------------------------------------------------
# Colour names (BARC canonical names + common synonyms humans use in LARC)
# ---------------------------------------------------------------------------
COLOR_CANON = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "grey", 6: "pink", 7: "orange", 8: "teal", 9: "maroon",
}
COLOR_SYNONYMS = {
    "black": 0,
    "blue": 1, "dark blue": 1,
    "red": 2,
    "green": 3,
    "yellow": 4,
    "grey": 5, "gray": 5, "silver": 5,
    "pink": 6, "magenta": 6, "purple": 6, "fuchsia": 6,
    "orange": 7,
    "teal": 8, "cyan": 8, "azure": 8, "light blue": 8, "sky blue": 8, "turquoise": 8,
    "maroon": 9, "brown": 9, "dark red": 9, "burgundy": 9,
}
# longest names first so "light blue" beats "blue"
_COLOR_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(k) for k in COLOR_SYNONYMS), key=len, reverse=True)) + r")(s?)\b",
    re.I,
)


def _match_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def rewrite_colors(text: str, table) -> str:
    """Apply a colour permutation table (new = table[old]) to colour words."""
    if not text:
        return text

    def sub(m):
        word, plural = m.group(1), m.group(2)
        old = COLOR_SYNONYMS[word.lower()]
        new = table[old]
        return _match_case(word, COLOR_CANON[new]) + plural

    return _COLOR_RE.sub(sub, text)


def has_color_words(text: str) -> bool:
    return bool(_COLOR_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Direction words vs. dihedral transforms
# ---------------------------------------------------------------------------
# swappable pairs (word-level mirror)
LR_PAIRS = [("left", "right"), ("leftmost", "rightmost"), ("leftward", "rightward"),
            ("leftwards", "rightwards"), ("east", "west"), ("eastward", "westward")]
UD_PAIRS = [("up", "down"), ("top", "bottom"), ("topmost", "bottommost"), ("above", "below"),
            ("upper", "lower"), ("upward", "downward"), ("upwards", "downwards"),
            ("north", "south"), ("uppermost", "lowermost"), ("ascending", "descending")]
ROT_PAIRS = [("clockwise", "counterclockwise"), ("clockwise", "counter-clockwise"),
             ("clockwise", "anticlockwise"), ("clockwise", "anti-clockwise")]
# vertical-only verbs that cannot be mirrored by word swap
UD_FIXED = {"fall", "falls", "falling", "fell", "fallen", "drop", "drops", "dropping", "dropped",
            "rise", "rises", "rising", "rose", "risen", "sink", "sinks", "sinking", "sank",
            "gravity", "float", "floats", "floating", "hang", "hangs", "hanging"}
# words that only survive transforms preserving the row/column axes
HV_WORDS = {"horizontal", "horizontally", "vertical", "vertically", "row", "rows", "column",
            "columns", "width", "height", "wide", "wider", "widest", "tall", "taller", "tallest",
            "widths", "heights"}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]*")


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def _groups(text: str) -> dict[str, bool]:
    ws = _words(text)
    lr = {w for p in LR_PAIRS for w in p}
    ud = {w for p in UD_PAIRS for w in p}
    rot = {w for p in ROT_PAIRS for w in p}
    return {
        "LR": bool(ws & lr),
        "UD": bool(ws & ud),
        "UD_FIXED": bool(ws & UD_FIXED),
        "HV": bool(ws & HV_WORDS),
        "ROT": bool(ws & rot),
    }


def has_direction_words(text: str) -> bool:
    return any(_groups(text).values())


def _swap_words(text: str, pairs) -> str:
    table = {}
    for a, b in pairs:
        table[a] = b
        table[b] = a

    def sub(m):
        w = m.group(0)
        r = table.get(w.lower())
        return _match_case(w, r) if r else w

    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, table), key=len, reverse=True)) + r")\b", re.I)
    return pat.sub(sub, text)


def rewrite_directions(text: str, geom: str) -> str | None:
    """Rewrite direction words for a dihedral transform, or None if not expressible.

    geom names follow ttt_qwen.GEOM: id r90 r180 r270 fh fv tr atr.
    fh = mirror left/right, fv = mirror up/down.
    """
    g = _groups(text)
    if geom == "id":
        return text
    any_axis = g["LR"] or g["UD"] or g["UD_FIXED"] or g["HV"]
    if geom == "fh":
        out = _swap_words(text, LR_PAIRS) if g["LR"] else text
        return _swap_words(out, ROT_PAIRS) if g["ROT"] else out
    if geom == "fv":
        if g["UD_FIXED"]:
            return None
        out = _swap_words(text, UD_PAIRS) if g["UD"] else text
        return _swap_words(out, ROT_PAIRS) if g["ROT"] else out
    if geom == "r180":
        if g["UD_FIXED"]:
            return None
        out = _swap_words(text, LR_PAIRS) if g["LR"] else text
        return _swap_words(out, UD_PAIRS) if g["UD"] else out
    if geom in ("r90", "r270"):
        return None if any_axis else text  # rotations keep clockwise-ness
    if geom in ("tr", "atr"):
        if any_axis:
            return None
        return _swap_words(text, ROT_PAIRS) if g["ROT"] else text
    return None


def allowed_geoms(text: str, names: Iterable[str] = ("id", "r90", "r180", "r270", "fh", "fv", "tr", "atr")):
    return [n for n in names if rewrite_directions(text, n) is not None]


def rewrite_rule(text: str, geom: str = "id", color_table=None) -> str | None:
    """Full rewrite for one augmented view; None if the direction words cannot follow."""
    out = rewrite_directions(text, geom)
    if out is None:
        return None
    if color_table is not None:
        out = rewrite_colors(out, color_table)
    return out


# ---------------------------------------------------------------------------
# LARC free-text descriptions
# ---------------------------------------------------------------------------
_LARC_TEMPLATES = [
    r"to make the output,?\s*you (have|need) to\s*\.*",
    r"(in )?the input,?\s*you should see\s*\.*",
    r"the output grid size\s*(is|should be)?\s*\.*",
]


def clean_larc(text: str) -> str:
    """Strip the LARC form templates ('To make the output, you have to...') and tidy."""
    t = (text or "").strip()
    for p in _LARC_TEMPLATES:
        t = re.sub(p, " ", t, flags=re.I)
    t = re.sub(r"\.{2,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .,;:")
    if t:
        t = t[0].upper() + t[1:]
        if not t.endswith((".", "!", "?")):
            t += "."
    return clean_text(t)


def larc_usable(rule: str, min_words: int = 6) -> bool:
    """Reject descriptions that are too terse or that are input descriptions typed in the wrong box."""
    if len(rule.split()) < min_words:
        return False
    low = rule.lower()
    return not re.search(r"\b(you should see|you will see|the input (has|contains|shows))\b", low)


if __name__ == "__main__":  # tiny self-test
    blk = format_rule_block("The blue pixels fall down until they hit the bottom.", ["gravity", "collision"])
    print(blk)
    r, g = split_rule_output(blk + "012\n345")
    print(parse_rule_block(r), repr(g))
    t = [0, 3, 1, 2, 4, 5, 6, 7, 8, 9]
    print(rewrite_rule("Blue pixels fall down; the red bar stays.", "fh", t))
    print(rewrite_rule("Blue pixels fall down; the red bar stays.", "fv", t))
    print(rewrite_rule("Move every object to the left; the top row is kept.", "r180"))
    print(allowed_geoms("Move every object to the left; the top row is kept."))
    print(allowed_geoms("Recolor each object by its size."))
    print(allowed_geoms("Rotate the shape clockwise."))
    print(clean_larc("To make the output, you have to...fill the black 2x2 square with the same color"))
