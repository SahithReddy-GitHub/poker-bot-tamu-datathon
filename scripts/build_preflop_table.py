"""
build_preflop_table.py - precompute preflop equity for every 5-card hand.

Run once (offline) from the repo root:
    python scripts/build_preflop_table.py [--trials 2000]

Hands that differ only by a suit relabelling have the same equity, so we
simulate one hand per suit-class and copy the value to all its members.
Writes data/preflop_equity.bin: 376,992 uint16 values (equity * 65535),
indexed by bot_utils.hand_index().
"""

import argparse
import os
import random
import sys
import time
from array import array
from itertools import combinations
from multiprocessing import Pool, cpu_count

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bot_utils import NUM_HANDS, PREFLOP_PATH, TRIPLES5, hand_index, hole_pair_masks  # noqa: E402
from evaluator import get_table  # noqa: E402


def canonical(hand):
    """Suit-independent key: the sorted per-suit rank sets."""
    by_suit = [[], [], [], []]
    for c in hand:
        by_suit[c & 3].append(c >> 2)
    return tuple(sorted(tuple(sorted(s)) for s in by_suit))


def equity(args):
    hand, trials, seed = args
    table = get_table()
    rng = random.Random(seed)
    rest = [c for c in range(36) if c not in hand]
    mine = hole_pair_masks(hand)
    total = 0.0
    for _ in range(trials):
        s = rng.sample(rest, 10)
        opp = hole_pair_masks(s[:5])
        bits = [1 << c for c in s[5:]]
        tri = [bits[i] | bits[j] | bits[k] for i, j, k in TRIPLES5]
        a = max([table[h | b] for h in mine for b in tri])
        o = max([table[h | b] for h in opp for b in tri])
        total += 1.0 if a > o else 0.5 if a == o else 0.0
    return total / trials


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=2000)
    args = p.parse_args()

    t0 = time.time()
    classes = {}
    for hand in combinations(range(36), 5):
        classes.setdefault(canonical(hand), hand)
    reps = list(classes.items())
    print(f"{len(reps)} suit classes; simulating {args.trials} trials each on {cpu_count()} cores")

    get_table()
    jobs = [(hand, args.trials, i) for i, (_, hand) in enumerate(reps)]
    with Pool(cpu_count()) as pool:
        eqs = pool.map(equity, jobs, chunksize=64)
    eq_by_key = {key: e for (key, _), e in zip(reps, eqs)}
    print(f"simulated in {time.time() - t0:.0f}s")

    out = array("H", [0]) * NUM_HANDS
    for hand in combinations(range(36), 5):
        out[hand_index(hand)] = int(round(eq_by_key[canonical(hand)] * 65535))
    os.makedirs(os.path.dirname(PREFLOP_PATH), exist_ok=True)
    with open(PREFLOP_PATH, "wb") as f:
        f.write(out.tobytes())
    print(f"wrote {PREFLOP_PATH} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
