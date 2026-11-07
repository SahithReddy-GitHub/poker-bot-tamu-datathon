"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "Royal Flush"
AGENT_NAME = "RangeEquityBot"


# ===== YOUR CODE HERE =====

import random
import time
from bisect import bisect_left

import bot_utils as bu
from evaluator import get_table

bu.load_preflop()          # load the preflop table at startup (fast, ~1 MB)
_TABLE = get_table()

# --- tunables ---------------------------------------------------------------
TIME_BUDGET = 5.5          # seconds of our own compute per half (bank is 8 s)
MAX_DECISION_TIME = 0.012  # never spend more than this on one decision
MIN_TRIALS = 24
MAX_TRIALS = 400

# Chance an opponent bets/raises with a hand of each made-hand category,
# relative to a straight-or-better (used to narrow its range postflop).
_CAT_WEIGHT = (0.30, 0.45, 0.70, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

# Strategy settings (tuned offline with scripts/league.py).
P = {
    "open_pct": 0.40,          # button opens (raises) hands above this percentile
    "open_pct_vs_folder": 0.30,  # ... and wider vs opponents who fold a lot
    "limp_pct": 0.12,          # button calls (limps) above this percentile
    "bb_raise_pct": 0.65,      # big blind raises a limp above this percentile
    "pf_reraise_eq": 0.6,     # re-raise preflop with this equity vs range
    "pf_call_margin": 0.09,    # extra equity needed to call a preflop raise
    "bet_pot_eq": 0.62,        # bet pot with this equity when checked to
    "bet_half_eq": 0.52,       # bet half pot with this equity
    "bet_half_eq_station": 0.55,  # ... vs opponents who rarely fold
    "ip_bonus": 0.03,           # lower betting thresholds by this in position
    "bluff_fold_thresh": 0.45, # only bluff vs fold-to-raise above this
    "bluff_base": 0.15,        # base bluff frequency
    "raise_eq_hi": 0.85,       # always raise a bet with this equity
    "raise_eq": 0.75,          # raise a bet with this equity (early in the street)
    "call_margin": 0.1,       # extra equity needed to call before the river
    "river_call_margin": 0.05,  # extra equity needed to call on the river
    "check_strong_keep": 1.0,  # after opp checks, keep its strong hands with this prob
    "time_budget": TIME_BUDGET,  # seconds of our own compute per half
    "steal_ftr": 0.80,         # vs opponents folding this often to raises: raise everything
}


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# --- opponent model ---------------------------------------------------------

def _stats(memory):
    s = memory["stats"]
    pf_raise = (s["pf_raise"] + 6) / (s["pf_act"] + 20)          # prior 0.30
    pf_cont = (s["pf_cont"] + 14) / (s["pf_act"] + 20)            # prior 0.70
    post_raise = (s["post_raise"] + 6) / (s["post_act"] + 20)     # prior 0.30
    fold_to_raise = (s["fold_vs_raise"] + 4) / (s["faced_raise"] + 10)  # prior 0.40
    # How much the opponent's raises actually mean strength (1 = tight, 0 = random).
    sig_pre = _clamp((0.70 - pf_raise) / 0.45, 0.0, 1.0)
    sig_post = _clamp((0.70 - post_raise) / 0.45, 0.0, 1.0)
    return pf_raise, pf_cont, post_raise, fold_to_raise, sig_pre, sig_post


def _opp_range(obs, memory):
    """(preflop raw threshold, postflop aggression weight) for the opponent's range."""
    pf_raise, pf_cont, _, _, sig_pre, sig_post = _stats(memory)
    keep = 1.0
    post_k = 0.0
    for a in obs.action_history:
        if a.is_me:
            continue
        if a.street == Street.PREFLOP:
            if a.action in RAISES:
                keep *= 1.0 - sig_pre * (1.0 - _clamp(pf_raise, 0.12, 1.0))
            elif a.action == Action.CHECK_CALL and a.amount > 0:
                keep *= 1.0 - 0.5 * sig_pre * (1.0 - _clamp(pf_cont, 0.3, 1.0))
        else:
            if a.action in RAISES:
                post_k += 1.0
            elif a.action == Action.CHECK_CALL and a.amount > 0:
                post_k += 0.4
    keep = _clamp(keep, 0.06, 1.0)
    thresh = bu.preflop_raw_threshold(1.0 - keep) if keep < 0.999 else 0
    return thresh, post_k * sig_post


def _opp_checked(obs):
    """True if the opponent checked on this street (a sign of a weaker hand)."""
    for a in obs.action_history:
        if a.street == obs.street and not a.is_me and a.action == Action.CHECK_CALL and a.amount == 0:
            return True
    return False


# --- equity -----------------------------------------------------------------

def _equity(obs, memory, allowance):
    """Monte Carlo equity vs the modelled opponent range, cached per situation."""
    thresh, post_w = _opp_range(obs, memory)
    chk_keep = P["check_strong_keep"] if (obs.street != Street.PREFLOP and _opp_checked(obs)) else 1.0
    key = (obs.hand_index, len(obs.board), thresh, round(post_w, 2), chk_keep)
    cache = memory["eq_cache"]
    if key in cache:
        return cache[key]

    deadline = time.perf_counter() + allowance
    rng = memory["rng"]
    table = _TABLE
    pre = bu._PREFLOP
    hole = list(obs.hole_cards)
    board = list(obs.board)
    used = set(hole) | set(board)
    rest = [c for c in range(36) if c not in used]
    need = 5 - len(board)
    mine = bu.hole_pair_masks(hole)
    board_bits = [1 << c for c in board]
    vis_tri = bu.board_triple_masks(board) if len(board) >= 3 else None
    river_tri = vis_tri if need == 0 else None
    my_river = max([table[h | b] for h in mine for b in river_tri]) if river_tri else 0
    filter_pre = pre is not None and thresh > 0
    filter_post = vis_tri is not None and (post_w > 0.01 or chk_keep < 1.0)
    hidx = bu.hand_index
    triples5 = bu.TRIPLES5
    pairs5 = bu.PAIRS5
    sample = rng.sample
    rnd = rng.random

    wins = 0.0
    n = 0
    while n < MAX_TRIALS:
        if n >= MIN_TRIALS and (n & 7) == 0 and time.perf_counter() > deadline:
            break
        for _ in range(40):  # rejection-sample an opponent hand from its range
            s = sample(rest, 5 + need)
            opp = s[:5]
            if filter_pre and pre[hidx(opp)] < thresh:
                continue
            if filter_post:
                ob = [1 << c for c in opp]
                om = [ob[i] | ob[j] for i, j in pairs5]
                cat = max([table[h | b] for h in om for b in vis_tri]) >> 20
                if rnd() > _CAT_WEIGHT[cat] ** post_w:
                    continue
                if cat >= 3 and chk_keep < 1.0 and rnd() > chk_keep:
                    continue
            break
        ob = [1 << c for c in opp]
        om = [ob[i] | ob[j] for i, j in pairs5]
        if need:
            bits = board_bits + [1 << c for c in s[5:]]
            tri = [bits[i] | bits[j] | bits[k] for i, j, k in triples5]
            a = max([table[h | b] for h in mine for b in tri])
        else:
            tri = river_tri
            a = my_river
        o = max([table[h | b] for h in om for b in tri])
        wins += 1.0 if a > o else 0.5 if a == o else 0.0
        n += 1
    eq = wins / n if n else 0.5
    cache[key] = eq
    return eq


def _allowance(obs, memory):
    """Seconds we may spend on this decision, from what is left of our budget."""
    left = P["time_budget"] - memory["t_used"]
    hands_left = max(1, memory["num_hands"] - obs.hand_index)
    dph = max(1.2, memory["decisions"] / max(1, obs.hand_index + 1))
    per = left / (hands_left * dph)
    return _clamp(per, 0.0, MAX_DECISION_TIME)


# --- decisions --------------------------------------------------------------

def _pick(obs, *prefs):
    """First preferred action that is legal; else check/call."""
    for a in prefs:
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL


def _call_or_fold(obs, ok):
    if obs.to_call == 0:
        return Action.CHECK_CALL
    return Action.CHECK_CALL if ok else _pick(obs, Action.FOLD)


def _preflop(obs, memory, allowance):
    pf_eq = bu.preflop_equity(obs.hole_cards)
    pct = bisect_left(bu._SORTED_EQ, int(pf_eq * 65535)) / len(bu._SORTED_EQ) if bu._SORTED_EQ else 0.5
    opp_raised = any((not a.is_me) and a.action in RAISES for a in obs.action_history)
    _, _, _, fold_to_raise, _, _ = _stats(memory)

    steal = fold_to_raise >= P["steal_ftr"] and memory["stats"]["faced_raise"] >= 20
    if not opp_raised:
        if steal:
            return _pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
        if obs.to_call > 0:
            # Button, first to act (only the blinds are in).
            open_pct = P["open_pct"] if fold_to_raise < 0.5 else P["open_pct_vs_folder"]
            if pct >= open_pct:
                return _pick(obs, Action.RAISE_POT)
            return _call_or_fold(obs, pct >= P["limp_pct"])
        # Big blind after a limp: raise good hands, check the rest.
        if pct >= P["bb_raise_pct"]:
            return _pick(obs, Action.RAISE_POT)
        return Action.CHECK_CALL

    # Facing a raise: compare equity vs the opponent's range to the price.
    if allowance > 0.0004:
        eq = _equity(obs, memory, allowance)
    else:
        eq = pf_eq - 0.03
    odds = obs.to_call / (obs.pot + obs.to_call)
    if eq >= P["pf_reraise_eq"] and obs.raises_this_street < 4:
        return _pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
    margin = 0.0 if obs.my_stack <= obs.to_call else P["pf_call_margin"]
    return _call_or_fold(obs, eq >= odds + margin)


def _postflop(obs, memory, allowance):
    if allowance > 0.0004:
        eq = _equity(obs, memory, allowance)
    else:  # out of time: crude made-hand strength
        eq = 0.15 + 0.12 * (bu.made_score(obs.hole_cards, obs.board) >> 20)
    _, _, _, fold_to_raise, _, _ = _stats(memory)
    rng = memory["rng"]
    river = obs.street == Street.RIVER

    if obs.to_call == 0:
        ip = P["ip_bonus"] if obs.is_button else 0.0
        if eq >= P["bet_pot_eq"] - ip:
            return _pick(obs, Action.RAISE_POT)
        if eq >= P["bet_half_eq"] - ip or (eq >= P["bet_half_eq_station"] - ip and fold_to_raise < 0.3):
            return _pick(obs, Action.RAISE_HALF)
        if fold_to_raise >= P["steal_ftr"] and memory["stats"]["faced_raise"] >= 20:
            return _pick(obs, Action.RAISE_HALF)
        # Bluff / semi-bluff only against opponents who fold enough.
        if fold_to_raise > P["bluff_fold_thresh"]:
            p = P["bluff_base"] + (fold_to_raise - P["bluff_fold_thresh"])
            if (not river and eq >= 0.40) or eq < 0.25:
                if rng.random() < p:
                    return _pick(obs, Action.RAISE_HALF)
        return Action.CHECK_CALL

    odds = obs.to_call / (obs.pot + obs.to_call)
    all_in_call = obs.my_stack <= obs.to_call
    if eq >= P["raise_eq_hi"] or (eq >= P["raise_eq"] and obs.raises_this_street < 2):
        return _pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
    if all_in_call:
        margin = 0.0
    else:
        margin = P["river_call_margin"] if river else P["call_margin"]
    return _call_or_fold(obs, eq >= odds + margin)


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["num_hands"] = getattr(config, "num_hands", 1000) or 1000
    memory["t_used"] = 0.0
    memory["decisions"] = 0
    memory["rng"] = random.Random(20261107)
    memory["eq_cache"] = {}
    memory["cur"] = None  # (hand_index, last history seen, my last action)
    memory["stats"] = {k: 0 for k in (
        "pf_act", "pf_raise", "pf_cont", "post_act", "post_raise",
        "faced_raise", "fold_vs_raise")}


def _safe_default(obs):
    if Action.CHECK_CALL in obs.legal_actions and (obs.to_call == 0 or Action.FOLD not in obs.legal_actions):
        return Action.CHECK_CALL
    return Action.FOLD if Action.FOLD in obs.legal_actions else obs.legal_actions[0]


def choose_action(obs: Observation, memory: dict) -> Action:
    t0 = time.perf_counter()
    action = None
    try:
        if "stats" not in memory:
            new_half(AgentConfig(), memory)
        if len(memory["eq_cache"]) > 64:
            memory["eq_cache"].clear()
        allowance = _allowance(obs, memory)
        if obs.street == Street.PREFLOP:
            action = _preflop(obs, memory, allowance)
        else:
            action = _postflop(obs, memory, allowance)
        memory["cur"] = (obs.hand_index, obs.action_history, action, obs.street)
    except Exception:
        action = None
    if action not in obs.legal_actions:
        action = _safe_default(obs)
    try:
        memory["decisions"] += 1
        memory["t_used"] += time.perf_counter() - t0
    except Exception:
        pass
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    t0 = time.perf_counter()
    try:
        s = memory["stats"]
        cur = memory.get("cur")
        if cur is None or cur[0] != result.hand_index:
            # We never acted: the button folded straight away.
            if not result.went_to_showdown and result.my_net > 0:
                s["pf_act"] += 1
            return
        _, history, my_last, my_street = cur
        prev_mine = None
        for a in history:
            if a.is_me:
                prev_mine = a.action
                continue
            pre = a.street == Street.PREFLOP
            s["pf_act" if pre else "post_act"] += 1
            if a.action in RAISES:
                s["pf_raise" if pre else "post_raise"] += 1
            if pre and a.action != Action.FOLD:
                s["pf_cont"] += 1
            if prev_mine in RAISES:
                s["faced_raise"] += 1
            prev_mine = None
        # The opponent's response to our last action, which we never saw.
        if my_last in RAISES:
            s["faced_raise"] += 1
            key = "pf_act" if my_street == Street.PREFLOP else "post_act"
            s[key] += 1
            if not result.went_to_showdown and result.my_net > 0:
                s["fold_vs_raise"] += 1
            elif my_street == Street.PREFLOP:
                s["pf_cont"] += 1
    except Exception:
        pass
    finally:
        try:
            memory["t_used"] += time.perf_counter() - t0
        except Exception:
            pass

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
