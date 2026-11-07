"""
fast_equity.py - vectorised (numpy) Monte Carlo equity, several times faster
than the pure-Python loop, so the bot gets more simulations per decision.

Trick: every 5-card hand is "2 hole cards + 3 board cards". We precompute a
table PT[pair_id * 7140 + triple_id] = hand score for all disjoint pairs and
triples, so scoring thousands of hands is a single numpy gather + max.

Opponent ranges are applied as per-sample weights (instead of rejection
sampling): equity = sum(weight * win) / sum(weight).

If numpy is missing, AVAILABLE is False and agent.py uses its Python loop.
"""

from __future__ import annotations

from itertools import combinations

try:
    import numpy as np
    AVAILABLE = True
except Exception:  # pragma: no cover - numpy is in the tournament image
    np = None
    AVAILABLE = False

import bot_utils as bu
from evaluator import get_table

N_TRIPLES = 7140  # C(36, 3)

_PT = None
_PRE = None
_B = None          # _B[k][c] = C(c, k)
_PAIRS5 = None     # (10, 2) index patterns
_TRIS5 = None      # (10, 3) index patterns
_CATW = None


def _pid(a, b):
    """Pair id for a < b."""
    return a + b * (b - 1) // 2


def _tid(a, b, c):
    """Triple id for a < b < c."""
    return a + b * (b - 1) // 2 + c * (c - 1) * (c - 2) // 6


def init(cat_weight) -> bool:
    """Build the lookup tables (about 1 s). Returns True when ready."""
    global _PT, _PRE, _B, _PAIRS5, _TRIS5, _CATW
    if not AVAILABLE:
        return False
    if _PT is not None:
        return True
    table = get_table()
    combos = np.array(list(combinations(range(36), 5)), dtype=np.int64)  # sorted rows
    scores = np.fromiter(
        (table[(1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e)] for a, b, c, d, e in combos.tolist()),
        dtype=np.int64, count=len(combos))
    pt = np.zeros(630 * N_TRIPLES, dtype=np.int32)
    for i, j in combinations(range(5), 2):
        rest = [k for k in range(5) if k not in (i, j)]
        p = _pid(combos[:, i], combos[:, j])
        t = _tid(combos[:, rest[0]], combos[:, rest[1]], combos[:, rest[2]])
        pt[p * N_TRIPLES + t] = scores
    _PT = pt
    pre = bu.load_preflop()
    _PRE = np.frombuffer(pre.tobytes(), dtype=np.uint16).astype(np.int32) if pre is not None else None
    c = np.arange(37, dtype=np.int64)
    _B = [np.ones(37, dtype=np.int64), c,
          c * (c - 1) // 2, c * (c - 1) * (c - 2) // 6,
          c * (c - 1) * (c - 2) * (c - 3) // 24,
          c * (c - 1) * (c - 2) * (c - 3) * (c - 4) // 120]
    _PAIRS5 = np.array(list(combinations(range(5), 2)))
    _TRIS5 = np.array(list(combinations(range(5), 3)))
    _CATW = np.array(cat_weight, dtype=np.float64)
    return True


def _pair_ids(sorted_hands):
    """(N, 5) sorted cards -> (N, 10) pair ids."""
    a = sorted_hands[:, _PAIRS5[:, 0]]
    b = sorted_hands[:, _PAIRS5[:, 1]]
    return a + _B[2][b]


def _triple_ids(sorted_boards):
    """(N, 5) sorted cards -> (N, 10) triple ids."""
    a = sorted_boards[:, _TRIS5[:, 0]]
    b = sorted_boards[:, _TRIS5[:, 1]]
    c = sorted_boards[:, _TRIS5[:, 2]]
    return a + _B[2][b] + _B[3][c]


def _visible_triple_ids(board):
    """Triple ids of a visible 3- or 4-card board."""
    b = sorted(board)
    return np.array([_tid(x, y, z) for x, y, z in combinations(b, 3)], dtype=np.int64)


def equity(hole, board, n, rng, pre_thresh=0, post_w=0.0, chk_keep=1.0, want_detail=False):
    """Equity of `hole` vs a weighted opponent range, from n random samples.

    pre_thresh: opponent hands with preflop table value below this get weight 0.
    post_w:     exponent on the made-hand category weights (opponent aggression).
    chk_keep:   weight for opponent straights-or-better after it checked.
    """
    used = set(hole) | set(board)
    rest = np.array([c for c in range(36) if c not in used], dtype=np.int64)
    need = 5 - len(board)
    k = 5 + need
    keys = rng.random((n, len(rest)))
    # argsort (not argpartition) so the k chosen cards come in random order;
    # argpartition leaves them in a card-correlated order and biases the deal.
    idx = np.argsort(keys, axis=1)[:, :k]
    cards = rest[idx]
    opp = np.sort(cards[:, :5], axis=1)
    if need:
        full = np.concatenate([np.broadcast_to(np.array(board, dtype=np.int64), (n, len(board))),
                               cards[:, 5:]], axis=1)
    else:
        full = np.broadcast_to(np.array(board, dtype=np.int64), (n, 5))
    full = np.sort(full, axis=1)
    tid = _triple_ids(full) if need else _triple_ids(full[:1])
    hs = np.array(sorted(hole), dtype=np.int64)[None, :]
    my_pid = _pair_ids(hs)[0]                                   # (10,)
    opp_pid = _pair_ids(opp)                                     # (n, 10)
    if need:
        mine = _PT[my_pid[None, :, None] * N_TRIPLES + tid[:, None, :]].max(axis=(1, 2))
        theirs = _PT[opp_pid[:, :, None] * N_TRIPLES + tid[:, None, :]].max(axis=(1, 2))
    else:
        mine = _PT[my_pid[:, None] * N_TRIPLES + tid[0][None, :]].max()
        theirs = _PT[opp_pid[:, :, None] * N_TRIPLES + tid[0][None, None, :]].max(axis=(1, 2))
    win = (mine > theirs).astype(np.float64) + 0.5 * (mine == theirs)

    w = None
    if pre_thresh > 0 and _PRE is not None:
        hidx = _B[1][opp[:, 0]] + _B[2][opp[:, 1]] + _B[3][opp[:, 2]] + _B[4][opp[:, 3]] + _B[5][opp[:, 4]]
        w = (_PRE[hidx] >= pre_thresh).astype(np.float64)
    if len(board) >= 3 and (post_w > 0.01 or chk_keep < 1.0):
        vt = _visible_triple_ids(board)
        made = _PT[opp_pid[:, :, None] * N_TRIPLES + vt[None, None, :]].max(axis=(1, 2))
        cat = made >> 20
        cw = _CATW[cat] ** post_w if post_w > 0.01 else np.ones(n)
        if chk_keep < 1.0:
            cw = cw * np.where(cat >= 3, chk_keep, 1.0)
        w = cw if w is None else w * cw
    if w is not None:
        total = w.sum()
        if total >= 8.0:  # enough effective samples; otherwise fall back to unweighted
            return float((w * win).sum() / total)
    return float(win.mean())
