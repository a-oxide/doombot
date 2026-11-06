"""what the student looks at: final-token activations, field ablation, decisiveness."""

import argparse
import json
import os
import re

import numpy as np
import torch

from . import actions as A
from . import readout
from .env import Arena
from .nav import Nav, load

FIELDS = ["enemies", "enemy", "health item", "threat", "walls",
          "health", "armor", "ammo", "kills", "weapon"]

NEUTRAL = {
    "health": ("health (-?\\d+)", "health 100"),
    "armor": ("armor (-?\\d+)", "armor 0"),
    "ammo": ("ammo (-?\\d+)", "ammo 50"),
    "kills": ("kills (\\d+)", "kills 0"),
    "weapon": ("weapon (-?\\d+)", "weapon 1"),
    "enemies": ("enemies [^|]*", "enemies none"),
    "enemy": ("enemy [^|]*", "enemy none"),
    "health item": ("health item [^|]*", "health item none"),
    "threat": ("threat [^|]*", "threat none"),
    "walls": ("walls [^|]*", "walls front:open front-right:open right:open back-right:open "
                             "back:open back-left:open left:open front-left:open"),
}


def ablate(text, field):
    pat, rep = NEUTRAL[field]
    return re.sub(pat, rep, text)


@torch.no_grad()
def forward(model, tok, ids, text, hidden=False):
    enc = tok([text], return_tensors="pt", truncation=True, max_length=1024)
    enc = {k: v.to("cuda") for k, v in enc.items()}
    try:
        out = model(**enc, logits_to_keep=1, output_hidden_states=hidden)
    except TypeError:
        out = model(**enc, output_hidden_states=hidden)
    p = torch.softmax(out.logits[:, -1, :].float()[:, ids], dim=-1)[0]
    hs = None
    if hidden and out.hidden_states is not None:
        hs = torch.stack([h[0, -1, :] for h in out.hidden_states[1:]]).float().cpu()
    return p, hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", default="data/deathmatch-lowpress.wad")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--cap", type=int, default=2100)
    ap.add_argument("--ablate", type=int, default=60, help="decisions to ablate")
    ap.add_argument("--out", default="runs/introspect")
    args = ap.parse_args()

    letters = readout.LETTERS[:len(A.OPTIONS12)]
    model, tok = readout.load(args.model, args.adapter)
    ids = torch.tensor(readout.letter_ids(tok, letters)[0], device="cuda")

    env = Arena(cfg=args.cfg, tics=args.cap, seed=args.seed)
    env.reset()
    env.state()
    nav = Nav(load(args.wad))
    t = 0
    lastpos = None
    last_nav = False
    probs, acts, texts = [], [], []
    while not env.game.is_episode_finished() and t < args.cap:
        state = env.state()
        v = env.vars()
        px, py = v.get("POSITION_X", 0.0), v.get("POSITION_Y", 0.0)
        moved = None if lastpos is None else ((px - lastpos[0]) ** 2 + (py - lastpos[1]) ** 2) ** 0.5
        text = readout.prompt(state, A.OPTIONS12, letters)
        p, hs = forward(model, tok, ids, text, hidden=True)
        idx = int(p.argmax())
        probs.append(p.cpu().numpy())
        acts.append(hs.numpy())
        texts.append(text)
        if idx == A.NAV:
            a, n = nav.act(px, py, v.get("ANGLE", 0.0), moved if last_nav else None)
            last_nav = True
        else:
            a, n, last_nav = A.ACTIONS12[idx], A.TICS12[idx], False
        env.step(a, n)
        t += n
        lastpos = (px, py)
    env.close()

    p = np.stack(probs)
    h = np.stack(acts)                      # (decisions, layers, hidden)
    top = np.sort(p, axis=1)[:, -2:]
    decisive = top[:, 1] - top[:, 0]        # margin over the runner-up
    var = h.var(axis=0).mean(axis=0)
    dims = np.sort(np.argsort(-var)[:32])

    kl = np.zeros((min(args.ablate, len(texts)), len(FIELDS)))
    for i in range(len(kl)):
        base = np.clip(p[i], 1e-12, 1.0)
        base = base / base.sum()
        for j, field in enumerate(FIELDS):
            q, _ = forward(model, tok, ids, ablate(texts[i], field))
            q = np.clip(q.cpu().numpy().astype(np.float64), 1e-12, 1.0)
            q = q / q.sum()
            kl[i, j] = float(np.sum(base * (np.log(base) - np.log(q))))

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(args.out + "/act.npz", probs=p, norms=np.linalg.norm(h, axis=2),
                        dims=dims, dims_values=h[:, :, dims].astype(np.float16),
                        decisive=decisive, kl=kl)
    meta = {"seed": args.seed, "decisions": len(p), "tics": t,
            "layers": int(h.shape[1]), "hidden": int(h.shape[2]),
            "fields": FIELDS, "dims": [int(d) for d in dims],
            "mean_decisive": float(decisive.mean()),
            "mean_kl": {f: float(kl[:, j].mean()) for j, f in enumerate(FIELDS)}}
    with open(args.out + "/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("decisions %d layers %d mean margin %.3f" % (len(p), h.shape[1], decisive.mean()))


if __name__ == "__main__":
    main()
