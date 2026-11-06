"""scripted policy: aim and fire on the nearest visible enemy, patrol when there is none."""

from .actions import DPT, FOV

FIRE_BAND = 0.04        # ~1.8 degrees, tight enough that the shot connects
LOW_HEALTH = 70.0
LOS_GRACE = 20          # tics of invisibility before a hold is dropped
HOLD_TIMEOUT = 300      # tics without a kill before a hold is dropped
AIM_DEADBAND = 0.10


def tics_for(off, cap=14):
    return int(max(1, min(cap, round(abs(off) * FOV / DPT))))


def tics_deg(deg, cap=14):
    return int(max(1, min(cap, round(abs(deg) / DPT))))


class Teacher:
    def __init__(self, nav, low_health=LOW_HEALTH, fire_band=FIRE_BAND):
        self.nav = nav
        self.low_health = low_health
        self.fire_band = fire_band
        self.hold = False           # still turning onto where the enemy was
        self.start = 0
        self.kill0 = 0.0
        self.last_off = 0.0
        self.seen = 0
        self.last = None
        self.last_tr = None

    def act(self, env, tics, health, kills):
        v = env.vars()
        px, py, ang = (v.get("POSITION_X", 0.0), v.get("POSITION_Y", 0.0), v.get("ANGLE", 0.0))
        moved = (None if self.last is None or self.last_tr != "nav"
                 else ((px - self.last[0]) ** 2 + (py - self.last[1]) ** 2) ** 0.5)
        enemy = env.enemy
        if enemy is not None:
            off = (enemy[1] - 160.0) / 160.0
            if not self.hold:
                self.hold, self.start, self.kill0 = True, tics, kills
            self.last_off, self.seen = off, tics
            if abs(off) <= self.fire_band:
                a, n, tr = 0, 2, "fire"
            else:
                a, n, tr = (4 if off > 0 else 3), tics_for(off), "aim"
        elif self.hold:
            gone = tics - self.seen
            if kills > self.kill0 or gone > LOS_GRACE or tics - self.start > HOLD_TIMEOUT:
                self.hold = False
                a, n, tr = self._tail(env, health)
            elif gone <= 4 and abs(self.last_off) > self.fire_band:
                a, n, tr = (4 if self.last_off > 0 else 3), tics_for(self.last_off, 4), "hold"
            else:
                a, n, tr = 5, 2, "hold"
        else:
            a, n, tr = self._tail(env, health)

        # hand the controls to the patrol only when nothing is visible to shoot
        if tr in ("explore", "orient") and enemy is None and not self.hold:
            c = env.contact
            if tr == "explore" or c is None or c[1] < 90:
                a, n = self.nav.act(px, py, ang, moved)
                tr = "nav"
        self.last, self.last_tr = (px, py), tr
        return a, n or 2, tr

    def _tail(self, env, health):
        item = env.item
        if health < self.low_health and item is not None:
            off = (item[1] - 160.0) / 160.0
            if abs(off) < AIM_DEADBAND:
                return 5, 2, "collect"
            return (4 if off > 0 else 3), tics_for(off), "collect"
        c = env.contact
        if c is not None and c[1] < 90 and abs(c[0]) > 10:
            return (4 if c[0] > 0 else 3), tics_deg(c[0]), "orient"
        if env.prof[0] > 0.97:
            return 4, 6, "explore"      # wall ahead: turn out of the corner
        return 5, 2, "explore"
