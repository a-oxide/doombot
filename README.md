# dmodel

A 2B language model plays deathmatch on a modified Doom map by reading a text state
and picking one of twelve options. It is trained by imitation: a scripted teacher plays,
the decisions are recorded, and a LoRA is fit to the 12-way letter readout.

## result

20 seeds (21-40), same arena, same starting state. player-attributed kills / damage,
mean per episode:

| cap | teacher | student |
|-----|---------|---------|
| 2,100 tics | 20.75 / 272 | 19.70 / 254 |
| 6,300 tics | 34.90 / 422 | 31.65 / 382 |

Damage per 1,000 tics is ~140 for both, so the student loses nothing per unit of
time played; it wins fewer long episodes because it dies earlier (see metrics).

## method

The teacher is a hand-written policy: turn onto the largest visible enemy, fire inside
1.8 degrees, hold the bearing for a few tics after the target leaves view, collect
health below 70, otherwise walk the patrol controller to the next spawn zone. Its
decisions are recorded as `{state, options, label}` rows, one row per decision, with
the state text that produced them.

The student reads the same text with the twelve option strings appended as lettered
choices, and the loss is over the softmax of those twelve letter tokens at the final
position: label-smoothed cross entropy plus a Brier term. Checkpoints are chosen by
playing episodes, not by validation loss — a checkpoint with better held-out accuracy
routinely plays worse.

Two things make this work at all. The option text carries its own tic count, so
"turn left 8 tics, 28 deg" cannot drift from what the harness executes. And navigation
is an option (`navigate: patrol-controller step`, index 11): the model chooses *when* to
hand the controls to the BFS patrol, not the route. With eleven motor options alone it
collapsed onto one of them and stopped deciding.

## reproducing

    python scripts/make_arena.py                 # data/deathmatch-lowpress.wad + arena.cfg
    PYTHONPATH=src python -m dmodel.record --out corpus/key12.jsonl
    PYTHONPATH=src python -m dmodel.train --corpus corpus/key12.jsonl --out adapters/dm-key12
    PYTHONPATH=src python -m dmodel.evaluate --adapter adapters/dm-key12 --seeds 21-40

`make_arena.py` needs a stock `deathmatch.wad` from the vizdoom scenarios directory;
everything else is in this repo. Recording 20 episodes at cap 6,300 is about 12k rows
and takes under a minute; training is ~2.5 s per optimizer step on one GPU.

There is an `introspect` entry point that replays an episode and dumps per-layer
activation norms, the top-variance hidden dimensions, and a field-ablation table (which
parts of the state text move the decision). It is how the figures in the write-up were
produced, not part of the training path.

## metrics

Damage dealt (`DAMAGECOUNT`) is the headline, not kills. `KILLCOUNT` also counts
monsters that kill each other, and on this map they do it often enough to move a
20-seed mean by several kills. Player kills are attributed by requiring damage to
have risen in the same step, and both numbers are printed.

Episodes end when the player dies or the cap is reached, so the number of tics played
differs per episode and per policy. Anything compared across policies is therefore
quoted per 1,000 tics; raw per-episode totals favour whichever policy survives longer.
Deaths are taken from the game's own death state at the end of the episode, not from
health crossing zero.

## layout

    src/dmodel/actions.py      option tables and tic counts
    src/dmodel/observation.py  state text
    src/dmodel/env.py          vizdoom wrapper, metrics
    src/dmodel/nav.py          walkability grid, bfs, patrol
    src/dmodel/teacher.py      scripted policy
    src/dmodel/readout.py      prompt, letter readout, model loading
    src/dmodel/rollout.py      one episode of play, shared move functions
    src/dmodel/{record,train,evaluate,introspect}.py   entry points

`python -m pytest tests` if you want the two invariant checks. no GPU needed;
the metrics test does need vizdoom importable.

MIT licensed, see LICENSE.
