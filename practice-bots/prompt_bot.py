"""prompt_bot.py - the kind of bot a team gets from a typical "elite poker AI"
LLM prompt: tiered heuristic preflop (pairs, double-suited, connectivity,
aces/kings), ~100 Monte Carlo rollouts vs a random hand, pot odds plus implied
odds for draws, and aggression-factor / fold-to-raise exploits with 5-10%
semi-bluffs. A realistic mid-strength opponent for tuning."""

import bot_utils as bu
from common import RAISES, Action, Street, call_or_fold, mc_equity, new_rng, pick

STRONG_RANKS = 7  # king or ace


def _wrap(ranks):
    """Most distinct ranks inside any 5-rank window (ace also plays low)."""
    rs = set(ranks)
    if 8 in rs:
        rs = rs | {-1}
    return max(sum(1 for r in rs if lo <= r < lo + 5) for lo in range(-1, 5))


def tier(hole):
    ranks = [c >> 2 for c in hole]
    suits = [c & 3 for c in hole]
    counts = {r: ranks.count(r) for r in set(ranks)}
    score = 0.0
    for r, v in counts.items():
        if v == 2:
            score += 3 if r >= STRONG_RANKS else 2 if r >= 5 else 1
        elif v >= 3:
            score -= 2
    if sum(1 for v in counts.values() if v == 2) >= 2:
        score += 1
    sc = sorted((suits.count(s) for s in set(suits)), reverse=True)
    if len(sc) > 1 and sc[0] >= 2 and sc[1] >= 2:
        score += 2
    elif sc[0] >= 2:
        score += 1
    if sc[0] >= 4:
        score -= 1
    w = _wrap(ranks)
    score += 2 if w >= 4 else 1 if w == 3 else 0
    score += 0.5 * sum(1 for r in ranks if r >= STRONG_RANKS)
    return 1 if score >= 7 else 2 if score >= 5 else 3 if score >= 3 else 4


def new_half(config, memory):
    new_rng(memory, 505)
    memory.update(opp_raise=0, opp_call=0, faced=0, folded=0, last=None)


def _stats(memory):
    af = (memory["opp_raise"] + 1) / (memory["opp_call"] + 1)
    ftr = (memory["folded"] + 2) / (memory["faced"] + 5)
    return af, ftr


def choose_action(obs, memory):
    memory["last"] = (obs.hand_index, obs.action_history, None)
    a = _choose(obs, memory)
    memory["last"] = (obs.hand_index, obs.action_history, a)
    return a


def _choose(obs, memory):
    rng = memory["rng"]
    af, ftr = _stats(memory)
    if obs.street == Street.PREFLOP:
        t = tier(obs.hole_cards)
        if t == 1:
            return pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
        if t == 2:
            if obs.raises_this_street < 2:
                return pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
            return Action.CHECK_CALL
        if t == 3:
            return call_or_fold(obs, obs.to_call <= 4)
        return call_or_fold(obs, False)

    eq = mc_equity(obs.hole_cards, obs.board, 100, rng)
    odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    cat = bu.made_score(obs.hole_cards, obs.board) >> 20
    drawing = obs.street != Street.RIVER and cat <= 1 and eq > 0.35
    if drawing:
        odds -= 0.05  # implied odds
    if af > 2.0:
        odds += 0.05  # tighten vs aggressive opponents

    if obs.to_call == 0:
        if eq > 0.80 and af > 2.0 and not obs.is_button:
            return Action.CHECK_CALL  # trap: let them bluff, then raise
        if eq > 0.70:
            return pick(obs, Action.RAISE_POT)
        if eq > (0.55 if af < 0.7 else 0.60):
            return pick(obs, Action.RAISE_HALF)
        if drawing and ftr > 0.5 and rng.random() < 0.08:
            return pick(obs, Action.RAISE_HALF)
        return Action.CHECK_CALL
    if eq > 0.80:
        return pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
    return call_or_fold(obs, eq > odds)


def on_hand_end(result, memory):
    last = memory.get("last")
    if not last or last[0] != result.hand_index:
        return
    _, hist, mine = last
    prev_raise = False
    for a in hist:
        if a.is_me:
            prev_raise = a.action in RAISES
            continue
        if a.action in RAISES:
            memory["opp_raise"] += 1
        elif a.action == Action.CHECK_CALL and a.amount > 0:
            memory["opp_call"] += 1
        if prev_raise:
            memory["faced"] += 1
        prev_raise = False
    if mine in RAISES:
        memory["faced"] += 1
        if not result.went_to_showdown and result.my_net > 0:
            memory["folded"] += 1
