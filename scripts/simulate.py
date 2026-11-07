"""
simulate.py - fast in-process duplicate matches (no HTTP), for tuning.

    python scripts/simulate.py agent.py baseline-bots/baseline_random.py --deals 1000 --seed 1

Plays the same 2-half duplicate format as the judge and prints bb/deal for
the first bot with its 95% confidence interval, plus each bot's compute time
per half. Errors in either bot are counted and printed (default action used).
"""

import argparse
import importlib.util
import math
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "practice-bots"))

from game import BIG_BLIND, AgentConfig, Game, button_for_deal, default_action, make_deal  # noqa: E402

_count = [0]


def load(path, params=None):
    _count[0] += 1
    spec = importlib.util.spec_from_file_location(f"bot_{_count[0]}", os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if params:
        mod.P.update(params)  # setting overrides for tuning
    return mod


def play_half(paths, seed, deals, half, params=(None, None)):
    """paths[0] sits in seat 0. Returns (per-deal nets for seat 0, times, errors)."""
    bots = [load(p, q) for p, q in zip(paths, params)]
    mems = [{}, {}]
    times = [0.0, 0.0]
    errors = [0, 0]
    for i in (0, 1):
        bots[i].new_half(AgentConfig(num_hands=deals), mems[i])
    nets = []
    for d in range(deals):
        g = Game(make_deal(seed, d), button_for_deal(d), hand_index=d)
        while not g.is_terminal:
            seat = g.to_act
            obs = g.observation(seat)
            t = time.perf_counter()
            try:
                a = bots[seat].choose_action(obs, mems[seat])
                if a not in obs.legal_actions:
                    raise ValueError(f"illegal {a!r}")
            except Exception:
                errors[seat] += 1
                if errors[seat] <= 2:
                    traceback.print_exc()
                a = default_action(obs.to_call)
            times[seat] += time.perf_counter() - t
            g.apply(a)
        for i in (0, 1):
            t = time.perf_counter()
            try:
                bots[i].on_hand_end(g.hand_result(i), mems[i])
            except Exception:
                errors[i] += 1
            times[i] += time.perf_counter() - t
        nets.append(g.result().nets[0])
    return nets, times, errors


def match(bot1, bot2, seed, deals, params1=None, params2=None):
    """Full duplicate match. Returns (mean bb/deal for bot1, lo, hi, times, errors)."""
    n1, t1, e1 = play_half([bot1, bot2], seed, deals, 1, (params1, params2))
    n2, t2, e2 = play_half([bot2, bot1], seed, deals, 2, (params2, params1))
    per_deal = [(x - y) / BIG_BLIND for x, y in zip(n1, n2)]  # bot1's bb per deal
    n = len(per_deal)
    mean = sum(per_deal) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in per_deal) / (n - 1))
    half = 1.96 * sd / math.sqrt(n)
    return mean, mean - half, mean + half, (t1, t2), (e1, e2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bot1")
    p.add_argument("bot2")
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--seed", default="1")
    a = p.parse_args()

    mean, lo, hi, (t1, t2), (e1, e2) = match(a.bot1, a.bot2, a.seed, a.deals)
    verdict = "BOT1_WIN" if lo > 0 else "BOT2_WIN" if hi < 0 else "CHOP"
    print(f"{a.bot1} vs {a.bot2} seed={a.seed}: {mean:+.3f} bb/deal  CI [{lo:+.3f}, {hi:+.3f}]  {verdict}")
    print(f"  bot1 time {t1[0]:.2f}s/{t2[1]:.2f}s errors {e1[0]}/{e2[1]} | "
          f"bot2 time {t1[1]:.2f}s/{t2[0]:.2f}s errors {e1[1]}/{e2[0]}")


if __name__ == "__main__":
    main()
