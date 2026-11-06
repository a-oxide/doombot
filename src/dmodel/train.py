"""lora training on a recorded corpus, with closed-loop checkpoint selection."""

import argparse
import json
import random

import torch
import torch.nn.functional as F

from . import actions as A
from . import readout
from .env import Arena
from .nav import Nav, load
from .rollout import play, student_move

TARGETS = "all-linear"


def corpus(path, n):
    items, opts = [], None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            opts = opts or r["options"]
            items.append({"text": r["state"], "label": int(r["label"])})
    return items[:n] if n else items, opts


def dev_kills(model, tok, ids_t, letters, grid, seeds, cap):
    """played-out kills per episode; this is the selection signal, not val loss."""
    out = []
    for seed in seeds:
        env = Arena(tics=cap, seed=seed)
        out.append(play(env, student_move(model, tok, ids_t, letters, Nav(grid)), cap)["kills"])
        env.close()
    return sum(out) / len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/key12.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--out", default="adapters/dm-key12")
    ap.add_argument("--steps", type=int, default=350)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--brier", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-seeds", default="44,46,48")
    ap.add_argument("--eval-cap", type=int, default=2100)
    ap.add_argument("--wad", default="data/deathmatch-lowpress.wad")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    items, opts = corpus(args.corpus, args.steps * args.batch * args.accum)
    letters = readout.LETTERS[:len(opts)]
    model, tok = readout.load(args.model)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                                             target_modules=TARGETS, bias="none",
                                             task_type="CAUSAL_LM"))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.03, anneal_strategy="cos",
                                                div_factor=10, final_div_factor=10)
    ids_t = torch.tensor(readout.letter_ids(tok, letters)[0], device="cuda")
    grid = load(args.wad)
    dev = [int(x) for x in args.eval_seeds.split(",")]

    step, best = 0, -1.0
    opt.zero_grad()
    while step < args.steps:
        random.Random(step).shuffle(items)
        for b in range(0, len(items) - args.batch + 1, args.batch):
            batch = items[b:b + args.batch]
            prompts = [readout.prompt(it["text"], opts, letters) for it in batch]
            gold = torch.tensor([it["label"] for it in batch], device="cuda")
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            enc = {k: v.to("cuda") for k, v in enc.items()}
            p = torch.softmax(readout.last_logits(model, enc)[:, ids_t], dim=-1)

            smooth = torch.full_like(p, args.smooth / p.shape[1])
            smooth.scatter_(1, gold.unsqueeze(1), 1.0 - args.smooth + args.smooth / p.shape[1])
            ce = -(smooth * torch.log(p.clamp_min(1e-9))).sum(-1).mean()
            brier = ((p - F.one_hot(gold, p.shape[1]).float()) ** 2).sum(-1).mean()
            ((ce + args.brier * brier) / args.accum).backward()

            if (b // args.batch + 1) % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 20 == 0:
                print("step %d ce %.4f brier %.4f acc %.3f"
                      % (step, ce.item(), brier.item(), (p.argmax(-1) == gold).float().mean().item()),
                      flush=True)
            if step % args.eval_every == 0:
                model.eval()
                k = dev_kills(model, tok, ids_t, letters, grid, dev, args.eval_cap)
                model.train()
                print("step %d dev kills %.2f (best %.2f)" % (step, k, best), flush=True)
                if k > best:
                    best = k
                    model.save_pretrained(args.out)
            if step >= args.steps:
                break
    print("trained %d steps, best dev kills %.2f -> %s" % (step, best, args.out))


if __name__ == "__main__":
    main()
