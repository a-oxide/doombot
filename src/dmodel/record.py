"""record a corpus from the scripted teacher."""

import argparse
import json

from . import actions as A
from .env import Arena
from .nav import Nav, load
from .teacher import Teacher


def label_for(action, tics):
    # index of the option that executes exactly this row and hold
    for i, (a, t) in enumerate(zip(A.ACTIONS11, A.TICS11)):
        if a == action and t == tics:
            return i
    same = [(abs(t - tics), i) for i, (a, t) in enumerate(zip(A.ACTIONS11, A.TICS11)) if a == action]
    return min(same)[1] if same else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", default="data/deathmatch-lowpress.wad")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--seeds", default="1-20")
    ap.add_argument("--cap", type=int, default=6300)
    ap.add_argument("--out", default="corpus/key12.jsonl")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.seeds.split("-"))
    grid = load(args.wad)
    f = open(args.out, "w")
    rows = miss = 0
    for seed in range(lo, hi + 1):
        env = Arena(cfg=args.cfg, tics=args.cap, seed=seed)
        env.reset()
        env.state()
        teacher = Teacher(Nav(grid))
        t = ep = 0
        while not env.game.is_episode_finished() and t < args.cap:
            v = env.vars()
            a, n, tr = teacher.act(env, t, v.get("HEALTH", 0.0), v.get("KILLCOUNT", 0.0))
            label = A.NAV if tr == "nav" else label_for(a, n)
            if label is None:
                miss += 1
                label = 1
            f.write(json.dumps({"state": env.state(), "options": A.OPTIONS12,
                                "label": int(label)}) + "\n")
            rows += 1
            ep += 1
            env.step(a, n)
            t += n
            env.state()
        print("seed %2d rows %3d tics %4d kills %2.0f" % (seed, ep, t, env.stats()["KILLCOUNT"]),
              flush=True)
        env.close()
    f.close()
    print("rows %d, unlabelled %d -> %s" % (rows, miss, args.out))


if __name__ == "__main__":
    main()
