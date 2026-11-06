"""closed-loop evaluation of the student or the teacher."""

import argparse
import json

from . import actions as A
from .env import Arena, summarise
from .nav import Nav, load
from .rollout import play, student_move, teacher_move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", default="data/deathmatch-lowpress.wad")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--policy", default="student", choices=["student", "teacher"])
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--seeds", default="21-40")
    ap.add_argument("--cap", type=int, default=2100)
    ap.add_argument("--json", default=None)
    ap.add_argument("--trace", default=None)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.seeds.split("-"))
    grid = load(args.wad)
    rows = []

    if args.policy == "teacher":
        build = teacher_move
    else:
        import torch
        from . import readout
        letters = readout.LETTERS[:len(A.OPTIONS12)]
        model, tok = readout.load(args.model, args.adapter)
        ids_t = torch.tensor(readout.letter_ids(tok, letters)[0], device="cuda")

        def build(nav):
            return student_move(model, tok, ids_t, letters, nav)

    for seed in range(lo, hi + 1):
        env = Arena(cfg=args.cfg, tics=args.cap, seed=seed)
        r = play(env, build(Nav(grid)), args.cap)
        env.close()
        r["seed"] = seed
        rows.append(r)
        print("seed %d tics %d kills %.0f damage %.0f pk %.0f dead %d"
              % (seed, r["tics"], r["kills"], r["damage"], r["player_kills"], r["dead"]),
              flush=True)

    s = summarise(rows)
    print("n %d | kills %.2f | player kills %.2f | damage %.1f | damage/1000 tics %.1f | "
          "tics %.0f | deaths %d/%d" % (s["episodes"], s["kills"], s["player_kills"], s["damage"],
                                        s["damage_per_1000"], s["tics"], s["deaths"], s["episodes"]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f)
    if args.trace:
        with open(args.trace, "w") as f:
            for r in rows:
                for d in r["trace"]:
                    f.write(json.dumps(dict(d, seed=r["seed"])) + "\n")


if __name__ == "__main__":
    main()
