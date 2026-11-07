"""
league.py - play one bot against a pool of opponents on several seeds, in parallel.

    python scripts/league.py                       # agent.py vs the default pool
    python scripts/league.py --params '{"open_pct": 0.35}' --seeds 1,2,3
    python scripts/league.py --variants variants.json   # many setting variants

Prints bb/deal per opponent (averaged over seeds), how many matches were
won/chopped/lost, and an overall score.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from simulate import match  # noqa: E402

POOL = [
    "practice-bots/standard_bot.py",
    "practice-bots/tag_bot.py",
    "practice-bots/lag_bot.py",
    "practice-bots/nit_bot.py",
    "practice-bots/v1_agent.py",
    "baseline-bots/baseline_random.py",
    "baseline-bots/baseline_calling_station.py",
    "baseline-bots/baseline_always_pot.py",
]


def job(args):
    name, bot, params, opp, seed, deals = args
    mean, lo, hi, times, errors = match(bot, opp, seed, deals, params)
    err = errors[0][0] + errors[1][1]
    t = max(times[0][0], times[1][1])
    return name, opp, seed, mean, lo, hi, err, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bot", default="agent.py")
    p.add_argument("--params", default="{}")
    p.add_argument("--variants", help="JSON file: {name: {param: value}}")
    p.add_argument("--pool", default=",".join(POOL))
    p.add_argument("--seeds", default="1,2")
    p.add_argument("--deals", type=int, default=1000)
    a = p.parse_args()

    variants = json.load(open(a.variants)) if a.variants else {"base": json.loads(a.params)}
    pool = a.pool.split(",")
    seeds = a.seeds.split(",")
    jobs = [(n, a.bot, v, o, s, a.deals) for n, v in variants.items() for o in pool for s in seeds]
    with Pool(min(cpu_count(), len(jobs))) as pl:
        res = pl.map(job, jobs, chunksize=1)

    for name in variants:
        rows = [r for r in res if r[0] == name]
        print(f"=== {name}: {json.dumps(variants[name])}")
        total = 0.0
        w = c = l = 0
        for o in pool:
            rs = [r for r in rows if r[1] == o]
            m = sum(r[3] for r in rs) / len(rs)
            total += m
            for r in rs:
                w += r[4] > 0
                l += r[5] < 0
                c += r[4] <= 0 <= r[5]
            errs = sum(r[6] for r in rs)
            tmax = max(r[7] for r in rs)
            print(f"  {os.path.basename(o):32s} {m:+7.3f} bb/deal  "
                  + " ".join(f"{r[3]:+.2f}" for r in rs) + f"  err={errs} t={tmax:.1f}s")
        print(f"  SCORE: W{w} C{c} L{l}  points={w + 0.5 * c:.1f}  avg={total / len(pool):+.3f}")


if __name__ == "__main__":
    main()
