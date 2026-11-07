"""checkfold_bot.py - the starter template's strategy (check if free, else fold).
This is also how any bot plays after it forfeits or runs out of time bank."""

from common import Action


def new_half(config, memory):
    pass


def choose_action(obs, memory):
    return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD


def on_hand_end(result, memory):
    pass
