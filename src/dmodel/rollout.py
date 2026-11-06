"""one episode of play, and the move functions the entry points share."""

import math

from . import actions as A
from . import readout
from .env import kill_delta
from .teacher import Teacher


def student_move(model, tok, ids_t, letters, nav):
    # the model picks an option; index 11 hands this step to the patrol controller
    def move(state, env, moved, last_nav, t):
        idx, _ = readout.choose(model, tok, ids_t, readout.prompt(state, A.OPTIONS12, letters))
        if idx != A.NAV:
            return A.ACTIONS12[idx], A.TICS12[idx], False
        v = env.vars()
        a, n = nav.act(v.get("POSITION_X", 0.0), v.get("POSITION_Y", 0.0),
                       v.get("ANGLE", 0.0), moved if last_nav else None)
        return a, n or 2, True
    return move


def teacher_move(nav):
    teacher = Teacher(nav)

    def move(state, env, moved, last_nav, t):
        v = env.vars()
        a, n, tr = teacher.act(env, t, v.get("HEALTH", 0.0), v.get("KILLCOUNT", 0.0))
        return a, n, tr == "nav"
    return move


def play(env, move, cap):
    # move(state, env, moved, used_nav, t) -> (row, tics, used_nav)
    env.reset()
    env.state()
    t = 0
    prev = (0.0, 0.0)
    pk = taken = 0.0
    lastpos = None
    last_nav = False
    trace = []
    st = {"KILLCOUNT": 0.0, "DAMAGECOUNT": 0.0, "HEALTH": 0.0}
    while not env.game.is_episode_finished() and t < cap:
        state = env.state()
        v = env.vars()
        px, py = v.get("POSITION_X", 0.0), v.get("POSITION_Y", 0.0)
        hp = v.get("HEALTH", 0.0)
        moved = None if lastpos is None else math.hypot(px - lastpos[0], py - lastpos[1])
        a, n, last_nav = move(state, env, moved, last_nav, t)
        n = int(n)
        trace.append({"t": t, "opt": a, "dt": n})
        env.step(a, n)
        t += n
        lastpos = (px, py)
        st = env.stats()
        taken += max(0.0, hp - st["HEALTH"])
        pk += kill_delta(prev, (st["KILLCOUNT"], st["DAMAGECOUNT"]))
        prev = (st["KILLCOUNT"], st["DAMAGECOUNT"])
    return {"tics": t, "kills": st["KILLCOUNT"], "damage": st["DAMAGECOUNT"],
            "player_kills": pk, "damage_taken": taken, "dead": env.dead, "trace": trace}
