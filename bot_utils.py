"""
bot_utils.py - fast helpers used by agent.py.

* hand_index(cards)   -> a unique int 0..376991 for any 5 sorted cards
* preflop_equity(...) -> equity of 5 hole cards vs a random hand (from a
                         precomputed table in data/preflop_equity.bin)
* best_score(...)     -> omaha_score with masks precomputed (for simulations)
* made_score(...)     -> best hand right now on a 3- or 4-card board

The preflop table is built offline by scripts/build_preflop_table.py.
"""

from __future__ import annotations

import os
from array import array
from itertools import combinations
from math import comb

from evaluator import NUM_CARDS, get_table

HERE = os.path.dirname(os.path.abspath(__file__))
PREFLOP_PATH = os.path.join(HERE, "data", "preflop_equity.bin")

NUM_HANDS = comb(NUM_CARDS, 5)  # 376,992

# _BINOM[k][c] = C(c, k+1): combinadic ranking of a sorted 5-card hand.
_BINOM = [[comb(c, k + 1) for c in range(NUM_CARDS)] for k in range(5)]
_B0, _B1, _B2, _B3, _B4 = _BINOM

PAIRS5 = tuple(combinations(range(5), 2))
TRIPLES3 = ((0, 1, 2),)
TRIPLES4 = tuple(combinations(range(4), 3))
TRIPLES5 = tuple(combinations(range(5), 3))


def hand_index(cards) -> int:
    """Unique index of 5 distinct cards (any order) in 0..NUM_HANDS-1."""
    a, b, c, d, e = sorted(cards)
    return _B0[a] + _B1[b] + _B2[c] + _B3[d] + _B4[e]


# ---------------------------------------------------------------------------
# Preflop table
# ---------------------------------------------------------------------------

_PREFLOP: array | None = None
_SORTED_EQ: list | None = None


def load_preflop() -> array | None:
    """Load the preflop equity table (uint16, equity * 65535). None if missing."""
    global _PREFLOP, _SORTED_EQ
    if _PREFLOP is None and os.path.isfile(PREFLOP_PATH):
        arr = array("H")
        with open(PREFLOP_PATH, "rb") as f:
            arr.frombytes(f.read())
        if len(arr) == NUM_HANDS:
            _PREFLOP = arr
            # 1-in-37 sample is plenty for percentile lookups.
            _SORTED_EQ = sorted(arr[::37])
    return _PREFLOP


def preflop_equity(cards) -> float:
    """Equity (0..1) of 5 hole cards against a random 5-card hand."""
    table = _PREFLOP if _PREFLOP is not None else load_preflop()
    if table is None:
        return 0.5
    return table[hand_index(cards)] / 65535.0


def equity_at_percentile(p: float) -> float:
    """The preflop equity such that a fraction p of all hands are below it."""
    if _SORTED_EQ is None:
        load_preflop()
    if not _SORTED_EQ:
        return 0.5
    i = min(len(_SORTED_EQ) - 1, max(0, int(p * len(_SORTED_EQ))))
    return _SORTED_EQ[i] / 65535.0


def preflop_raw_threshold(p: float) -> int:
    """Like equity_at_percentile, but as the raw uint16 value in the table."""
    if _SORTED_EQ is None:
        load_preflop()
    if not _SORTED_EQ:
        return 0
    i = min(len(_SORTED_EQ) - 1, max(0, int(p * len(_SORTED_EQ))))
    return _SORTED_EQ[i]


# ---------------------------------------------------------------------------
# Fast scoring
# ---------------------------------------------------------------------------

def hole_pair_masks(hole) -> list:
    """The 10 two-card masks of 5 hole cards."""
    bits = [1 << c for c in hole]
    return [bits[i] | bits[j] for i, j in PAIRS5]


def board_triple_masks(board) -> list:
    """All three-card masks from a 3, 4 or 5 card board."""
    bits = [1 << c for c in board]
    n = len(bits)
    triples = TRIPLES5 if n == 5 else TRIPLES4 if n == 4 else TRIPLES3
    return [bits[i] | bits[j] | bits[k] for i, j, k in triples]


def best_score(pair_masks, triple_masks, table=None) -> int:
    """Best 2+3 score given precomputed masks."""
    t = table if table is not None else get_table()
    return max([t[h | b] for h in pair_masks for b in triple_masks])


def made_score(hole, board) -> int:
    """Best score right now using 2 hole cards + 3 of the visible board cards."""
    return best_score(hole_pair_masks(hole), board_triple_masks(board))
