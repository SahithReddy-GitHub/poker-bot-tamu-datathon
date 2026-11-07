"""nit_bot.py - very tight and passive-ish: only strong hands, never bluffs,
folds to pressure unless strong. Tests whether we steal enough pots."""

from common import Action, Street, call_or_fold, mc_equity, new_rng, opp_raised, pct, pick


def new_half(config, memory):
    new_rng(memory, 404)


def choose_action(obs, memory):
    if obs.street == Street.PREFLOP:
        p = pct(obs.hole_cards)
        if p > 0.80:
            return pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
        if opp_raised(obs):
            return call_or_fold(obs, p > 0.70)
        return call_or_fold(obs, p > 0.55)
    eq = mc_equity(obs.hole_cards, obs.board, 70, memory["rng"])
    odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if eq > 0.75:
        return pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
    return call_or_fold(obs, eq > odds + 0.12)


def on_hand_end(result, memory):
    pass
