"""standard_bot.py - what a typical team builds from the guide: heuristic
preflop, Monte Carlo equity vs a random hand, pot odds. No opponent model."""

from common import Action, Street, call_or_fold, heuristic_preflop, mc_equity, new_rng, opp_raised, pick


def new_half(config, memory):
    new_rng(memory, 101)


def choose_action(obs, memory):
    if obs.street == Street.PREFLOP:
        s = heuristic_preflop(obs.hole_cards)
        if s > 0.62:
            return pick(obs, Action.RAISE_POT)
        if opp_raised(obs) and obs.to_call > 6:
            return call_or_fold(obs, s > 0.50)
        if s > 0.40:
            return Action.CHECK_CALL
        return call_or_fold(obs, False)
    eq = mc_equity(obs.hole_cards, obs.board, 60, memory["rng"])
    odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if eq > 0.70:
        return pick(obs, Action.RAISE_POT)
    if obs.to_call == 0 and eq > 0.55:
        return pick(obs, Action.RAISE_HALF)
    return call_or_fold(obs, eq > odds)


def on_hand_end(result, memory):
    pass
