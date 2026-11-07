"""
common.py - shared pieces for the practice opponents in this folder.

These bots are only for local testing (scripts/simulate.py, scripts/league.py).
They are not part of the submitted bot.
"""

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bot_utils as bu  # noqa: E402
from evaluator import get_table  # noqa: E402
from game import Action, Street  # noqa: E402

bu.load_preflop()
TABLE = get_table()
RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def pick(obs, *prefs):
    for a in prefs:
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL


def call_or_fold(obs, ok):
    if obs.to_call == 0 or ok:
        return Action.CHECK_CALL
    return pick(obs, Action.FOLD)


def pct(hole):
    """Percentile of a starting hand (0 = worst, 1 = best)."""
    from bisect import bisect_left
    eq = bu.preflop_equity(hole)
    return bisect_left(bu._SORTED_EQ, int(eq * 65535)) / len(bu._SORTED_EQ)


def heuristic_preflop(hole):
    """A typical hand-written preflop score (no table), roughly 0..1.

    Rewards pairs, suited cards, connected ranks and high cards, and punishes
    trips - the kind of rule most teams write from the guide.
    """
    ranks = sorted((c >> 2 for c in hole), reverse=True)
    suits = [c & 3 for c in hole]
    score = sum(ranks) / 40.0
    counts = {r: ranks.count(r) for r in ranks}
    pairs = sum(1 for v in counts.values() if v == 2)
    trips = sum(1 for v in counts.values() if v >= 3)
    score += 0.12 * pairs + 0.04 * sum(r for r, v in counts.items() if v == 2) / 8.0
    score -= 0.15 * trips
    sc = sorted((suits.count(s) for s in set(suits)), reverse=True)
    if sc[0] >= 2:
        score += 0.08
    if len(sc) > 1 and sc[1] >= 2:
        score += 0.06
    if sc[0] >= 4:
        score -= 0.05
    uniq = sorted(set(ranks))
    gaps = sum(1 for a, b in zip(uniq, uniq[1:]) if b - a <= 2)
    score += 0.03 * gaps
    return max(0.0, min(1.0, score / 1.1))


def mc_equity(hole, board, iters, rng):
    """Equity vs one RANDOM hand (no range), like the guide suggests."""
    hole = list(hole)
    board = list(board)
    used = set(hole) | set(board)
    rest = [c for c in range(36) if c not in used]
    need = 5 - len(board)
    mine = bu.hole_pair_masks(hole)
    bb = [1 << c for c in board]
    wins = 0.0
    for _ in range(iters):
        s = rng.sample(rest, 5 + need)
        bits = bb + [1 << c for c in s[5:]]
        tri = [bits[i] | bits[j] | bits[k] for i, j, k in bu.TRIPLES5]
        ob = [1 << c for c in s[:5]]
        om = [ob[i] | ob[j] for i, j in bu.PAIRS5]
        a = max([TABLE[h | b] for h in mine for b in tri])
        o = max([TABLE[h | b] for h in om for b in tri])
        wins += 1.0 if a > o else 0.5 if a == o else 0.0
    return wins / iters


def opp_raised(obs, street=None):
    return any((not a.is_me) and a.action in RAISES and (street is None or a.street == street)
               for a in obs.action_history)


def new_rng(memory, seed):
    memory["rng"] = random.Random(seed)
