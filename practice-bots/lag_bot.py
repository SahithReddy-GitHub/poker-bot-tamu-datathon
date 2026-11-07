"""lag_bot.py - loose-aggressive: plays most hands, raises and bluffs a lot,
calls light. Tests whether we can stand up to pressure."""

from common import Action, Street, call_or_fold, mc_equity, new_rng, opp_raised, pct, pick


def new_half(config, memory):
    new_rng(memory, 303)


def choose_action(obs, memory):
    rng = memory["rng"]
    if obs.street == Street.PREFLOP:
        p = pct(obs.hole_cards)
        if not opp_raised(obs) or obs.raises_this_street < 2:
            if p > 0.25 or rng.random() < 0.3:
                return pick(obs, Action.RAISE_POT)
            return Action.CHECK_CALL
        return call_or_fold(obs, p > 0.35)
    eq = mc_equity(obs.hole_cards, obs.board, 50, rng)
    odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if eq > 0.60 or (rng.random() < 0.35 and obs.raises_this_street < 2):
        return pick(obs, Action.RAISE_POT if eq > 0.6 else Action.RAISE_HALF)
    return call_or_fold(obs, eq > odds - 0.05)


def on_hand_end(result, memory):
    pass
