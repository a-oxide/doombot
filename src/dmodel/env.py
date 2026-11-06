"""vizdoom wrapper for the arena: carry-hold stepping and the metric readout."""

import os
import pathlib

import numpy as np
import vizdoom as vzd

from .observation import ENEMIES, ITEMS, text

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG = os.environ.get("DM_CFG") or str(ROOT / "data" / "arena.cfg")
TIMEOUT = 2100
VARS = ("ANGLE", "POSITION_X", "POSITION_Y", "FRAGCOUNT", "DEATHCOUNT")


class Arena:
    def __init__(self, cfg=None, tics=TIMEOUT, seed=None):
        g = vzd.DoomGame()
        g.load_config(cfg or CFG)
        g.set_doom_map("map01")
        g.set_window_visible(False)
        g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        g.set_screen_format(vzd.ScreenFormat.GRAY8)
        g.set_depth_buffer_enabled(True)
        g.set_labels_buffer_enabled(True)
        g.set_objects_info_enabled(True)
        g.set_automap_buffer_enabled(True)
        g.set_automap_mode(vzd.AutomapMode.OBJECTS_WITH_SIZE)
        g.set_episode_timeout(tics)
        have = {v.name for v in g.get_available_game_variables()}
        for name in VARS:
            if name not in have:
                g.add_available_game_variable(getattr(vzd.GameVariable, name))
        if seed is not None:
            g.set_seed(seed)
        g.init()
        self.game = g

        names = [b.name for b in g.get_available_buttons()]
        ix = {n: i for i, n in enumerate(names)}
        F, L, R = ix["MOVE_FORWARD"], ix["MOVE_LEFT"], ix["MOVE_RIGHT"]
        TL, TR, AT = ix["TURN_LEFT"], ix["TURN_RIGHT"], ix["ATTACK"]
        run = ix.get("SPEED")

        def combo(*bs):
            row = [0.0] * len(names)
            for b in bs:
                row[b] = 1.0
            if run is not None and any(b in (F, L, R) for b in bs):
                row[run] = 1.0      # always sprint while moving; the teacher was recorded that way
            return row

        self.rows = [combo(F, AT), combo(L, AT), combo(R, AT), combo(TL), combo(TR),
                     combo(F), combo(AT)]
        self.attack = AT
        self.labels = []
        self.enemy = None
        self.item = None
        self.contact = None
        self.prof = [0.0] * 8
        self._vars = {}

    def reset(self):
        self.game.new_episode()

    def close(self):
        self.game.close()

    @property
    def dead(self):
        return bool(self.game.is_player_dead())

    def vars(self):
        st = self.game.get_state()
        if st is not None:
            names = [a.name for a in self.game.get_available_game_variables()]
            self._vars = {k: float(v) for k, v in zip(names, st.game_variables)}
        return self._vars

    def contact_from_automap(self):
        # nearest bright blob on the automap: 360 degrees, but blind to what it is
        st = self.game.get_state()
        if st is None or st.automap_buffer is None:
            return None
        am = st.automap_buffer
        ys, xs = np.nonzero(am >= 230)
        if ys.size == 0:
            return None
        cy, cx = am.shape[0] / 2.0, am.shape[1] / 2.0
        dist = np.hypot(ys - cy, xs - cx)
        keep = dist >= 12          # the player's own blip
        if not keep.any():
            return None
        ys, xs, dist = ys[keep], xs[keep], dist[keep]
        j = int(np.argmin(dist))
        ang = np.degrees(np.arctan2(float(xs[j] - cx), -float(ys[j] - cy)))
        rel = (ang - self.vars().get("ANGLE", 0.0) + 540.0) % 360.0 - 180.0
        return rel, float(dist[j])

    def perceive(self):
        st = self.game.get_state()
        self.labels = [] if st is None or not st.labels else st.labels
        self.enemy = _largest(self.labels, ENEMIES)
        self.item = _largest(self.labels, ITEMS)
        self.contact = self.contact_from_automap()
        if st is not None and st.depth_buffer is not None:
            d = st.depth_buffer.astype(np.float32) / 255.0
            h, w = d.shape
            rows = d[int(h * 0.35):int(h * 0.75)]
            self.prof = [float(rows[:, int(w * i / 8):int(w * (i + 1) / 8)].max(axis=0).mean())
                         for i in range(8)]
        return self.enemy, self.item

    def state(self):
        self.perceive()
        st = self.game.get_state()
        return text(self.vars(), self.labels, self.prof, self.contact,
                    None if st is None else st.objects)

    def step(self, action, tics):
        # carry-hold: one button row held for the whole interval
        return self.game.make_action(self.rows[int(action)], int(tics))

    def stats(self):
        # read straight from the game: the cached state is stale once the episode ends,
        # and the last kill happens on the fatal tic
        out = {}
        for name in ("KILLCOUNT", "DAMAGECOUNT", "HEALTH"):
            try:
                out[name] = float(self.game.get_game_variable(getattr(vzd.GameVariable, name)))
            except Exception:
                out[name] = self._vars.get(name, 0.0)
        return out


def _largest(labels, wanted):
    best = None
    for lb in labels:
        if lb.object_name not in wanted:
            continue
        area = float(lb.width * lb.height)
        if best is None or area > best[2]:
            best = (lb.object_name, lb.x + lb.width / 2.0, area)
    return best


def kill_delta(prev, cur):
    # a kill that did not come with damage on the same step was two monsters
    # fighting, not ours
    dk = cur[0] - prev[0]
    return dk if dk > 0 and cur[1] > prev[1] else 0.0


def summarise(rows):
    tics = sum(r["tics"] for r in rows)
    dmg = sum(r["damage"] for r in rows)
    n = max(1, len(rows))
    return {"episodes": len(rows),
            "tics": tics,
            "kills": sum(r["kills"] for r in rows) / n,
            "player_kills": sum(r["player_kills"] for r in rows) / n,
            "damage": dmg / n,
            "damage_per_1000": 1000.0 * dmg / max(1, tics),
            "deaths": sum(1 for r in rows if r["dead"]),
            "damage_taken": sum(r["damage_taken"] for r in rows) / n}
