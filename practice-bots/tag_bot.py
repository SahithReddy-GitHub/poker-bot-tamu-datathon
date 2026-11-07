"""tag_bot.py - tight-aggressive: good preflop table, plays fewer hands,
bets them hard, some continuation-bet bluffs."""

from common import Action, Street, call_or_fold, mc_equity, new_rng, opp_raised, pct, pick


def new_half(config, memory):
    new_rng(memory, 202)


def choose_action(obs, memory):
    rng = memory["rng"]
    if obs.street == Street.PREFLOP:
        p = pct(obs.hole_cards)
        if not opp_raised(obs):
            if p > 0.45:
                return pick(obs, Action.RAISE_POT)
            return call_or_fold(obs, p > 0.25)
        if p > 0.85:
            return pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
        return call_or_fold(obs, p > 0.55 or (obs.to_call <= 4 and p > 0.4))
    eq = mc_equity(obs.hole_cards, obs.board, 70, rng)
    odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if obs.to_call == 0:
        if eq > 0.62:
            return pick(obs, Action.RAISE_POT)
        if obs.street == Street.FLOP and rng.random() < 0.35:
            return pick(obs, Action.RAISE_HALF)
        return Action.CHECK_CALL
    if eq > 0.78:
        return pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
    return call_or_fold(obs, eq > odds + 0.08)


def on_hand_end(result, memory):
    pass
