"""held-out calibration and the escalation threshold, on unseen episodes."""

import argparse
import json

import numpy as np
import torch

from . import readout


def split_episodes(path, holdout):
    """the corpus is written in play order and the tic counter restarts each episode."""
    rows = [json.loads(line) for line in open(path)]
    with open(path + ".t") as f:
        tics = [int(line) for line in f]

    eps, cur = [], []
    for r, t in zip(rows, tics):
        if cur and t <= cur[-1][1]:
            eps.append(cur)
            cur = []
        cur.append((r, t))
    if cur:
        eps.append(cur)

    cut = len(eps) - max(1, int(round(len(eps) * holdout)))
    return [r for e in eps[:cut] for r, _ in e], [r for e in eps[cut:] for r, _ in e], len(eps)


def gather(model, tok, ids, rows, letters, batch):
    """logits over the option letters, before softmax: temperature needs raw scores."""
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        texts = [readout.prompt(r["state"], r["options"], letters) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        out.append(readout.last_logits(model, enc)[:, ids].cpu().numpy())
    return np.concatenate(out)


def softmax(z, t=1.0):
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(z, y):
    """one scalar, fitted by nll on rows disjoint from the ones being reported."""
    best, best_t = None, 1.0
    for t in np.arange(0.4, 3.01, 0.05):
        p = softmax(z, t)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean()
        if best is None or nll < best:
            best, best_t = nll, float(t)
    return best_t


def stats(p, y):
    pmax, pred = p.max(1), p.argmax(1)
    correct = pred == y
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1

    edges = np.linspace(0.0, 1.0, 11)
    which = np.clip(np.digitize(pmax, edges[1:-1]), 0, 9)
    table, ece = [], 0.0
    for b in range(10):
        m = which == b
        if m.sum():
            table.append((float(pmax[m].mean()), float(correct[m].mean()), int(m.sum())))
            ece += abs(correct[m].mean() - pmax[m].mean()) * m.sum() / len(y)

    brier = float(((p - onehot) ** 2).sum(1).mean())
    return {"acc": float(correct.mean()), "brier": brier, "ece": float(ece),
            "conf": float(pmax.mean()), "correct": correct, "pmax": pmax, "table": table}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/key12.jsonl")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--fit-rows", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    # episodes, not rows: consecutive decisions inside one episode are near duplicates,
    # so a row split reports a skill the policy does not have
    train, test, n = split_episodes(args.corpus, args.holdout)
    letters = readout.LETTERS[:len(test[0]["options"])]
    model, tok = readout.load(args.model, args.adapter)
    ids = torch.tensor(readout.letter_ids(tok, letters)[0], device="cuda")

    stride = max(1, len(train) // args.fit_rows)
    fit = train[::stride]
    z_fit = gather(model, tok, ids, fit, letters, args.batch)
    y_fit = np.array([r["label"] for r in fit])
    z = gather(model, tok, ids, test, letters, args.batch)
    y = np.array([r["label"] for r in test])

    t = fit_temperature(z_fit, y_fit)
    raw, cal = stats(softmax(z), y), stats(softmax(z, t), y)

    print("episodes %d | held out %d | rows %d train / %d test | temperature %.2f"
          % (n, round(n * args.holdout), len(train), len(test), t))
    print("accuracy %.3f | brier %.4f -> %.4f | ece %.4f -> %.4f | mean confidence %.3f -> %.3f"
          % (raw["acc"], raw["brier"], cal["brier"], raw["ece"], cal["ece"], raw["conf"], cal["conf"]))
    print()
    print("  confidence   accuracy   decisions")
    for conf, acc, n_b in cal["table"]:
        print("  %.2f         %.3f      %d" % (conf, acc, n_b))

    errors = int((~cal["correct"]).sum())
    print()
    print("escalate when p <= threshold; %d errors in %d decisions" % (errors, len(y)))
    print("  threshold   kept    acc(kept)   escalated   acc(escalated)   errors caught")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        kept = cal["pmax"] > thr
        rest = ~kept
        caught = int((~cal["correct"] & rest).sum())
        print("  %.2f        %5.1f%%  %-10.3f  %5.1f%%      %-14s  %5.1f%%"
              % (thr, 100 * kept.mean(),
                 cal["correct"][kept].mean() if kept.any() else float("nan"),
                 100 * rest.mean(),
                 "%.3f" % cal["correct"][rest].mean() if rest.any() else "-",
                 100 * caught / max(1, errors)))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"episodes": n, "test_rows": len(test), "temperature": t,
                       "raw": {k: raw[k] for k in ("acc", "brier", "ece", "conf")},
                       "calibrated": {k: cal[k] for k in ("acc", "brier", "ece", "conf")},
                       "bins": [{"conf": c, "acc": a, "n": k} for c, a, k in cal["table"]]}, f, indent=1)


if __name__ == "__main__":
    main()
