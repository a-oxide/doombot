"""walkability grid from the wad's TEXTMAP, bfs over it, and the spawn-zone patrol."""

import collections
import math
import re
import struct

CELL = 16.0
BODY = 32.0          # player width; the grid is grown by half of this
STEP_UP = 24.0
MIN_GAP = 56.0
N8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

FWD, TL, TR = 5, 3, 4        # rows of the button table

# the eight acs spawn zones, most visited first
SPAWNS = [(838.1, 294.3), (835.9, 819.8), (194.6, 504.2), (561.7, 489.7),
          (213.4, 789.8), (511.0, 848.8), (218.4, 206.3), (586.2, 116.6)]


def lumps(path):
    # {name: bytes} for a wad directory
    data = open(path, "rb").read()
    n, off = struct.unpack_from("<ii", data, 4)
    out = {}
    for i in range(n):
        lo, size, name = struct.unpack_from("<ii8s", data, off + 16 * i)
        out[name.rstrip(b"\0").decode("latin1")] = data[lo:lo + size]
    return out


def _blocks(txt, kind):
    out = []
    for m in re.finditer(r"(?m)^\s*" + kind + r"\s*(?://[^\n]*)?\s*\{", txt):
        i, depth, j = m.end(), 1, m.end()
        while depth and j < len(txt):
            if txt[j] == "{":
                depth += 1
            elif txt[j] == "}":
                depth -= 1
            j += 1
        body = txt[i:j - 1]
        out.append(dict(re.findall(r"(?m)^\s*(\w+)\s*=\s*(-?\d+|\"[^\"]*\"|[A-Za-z_]\w*)", body)))
    return out


def build(wad):
    # udmf map -> cell grid; a cell is solid if a blocking line touches it
    txt = lumps(wad)["TEXTMAP"].decode("utf-8", "ignore")
    verts = [(int(v["x"]), int(v["y"])) for v in _blocks(txt, "vertex")]
    sides = _blocks(txt, "sidedef")
    secs = _blocks(txt, "sector")
    lines = _blocks(txt, "linedef")

    def sec_h(si):
        s = secs[int(si)]
        return int(s.get("heightfloor", 0)), int(s.get("heightceiling", 192))

    def blocked(L):
        if int(L.get("special", 0)) != 0:
            return False
        sf, sb = L.get("sidefront"), L.get("sideback")
        if L.get("blocking", "false").lower() == "true":
            return True
        if sb is None or sb == "-1" or sf is None:
            return True
        fa, ca = sec_h(sides[int(sf)]["sector"])
        fb, cb = sec_h(sides[int(sb)]["sector"])
        return abs(fa - fb) > STEP_UP or (min(ca, cb) - max(fa, fb)) < MIN_GAP

    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    x0, y0 = min(xs) - 64, min(ys) - 64
    w = int((max(xs) + 64 - x0) / CELL) + 1
    h = int((max(ys) + 64 - y0) / CELL) + 1
    solid = bytearray(w * h)
    for L in lines:
        if not blocked(L):
            continue
        ax, ay = verts[int(L["v1"])]
        bx, by = verts[int(L["v2"])]
        n = int(max(abs(bx - ax), abs(by - ay)) / 4) + 1
        for k in range(n + 1):
            px, py = ax + (bx - ax) * k / n, ay + (by - ay) * k / n
            cx, cy = int((px - x0) / CELL), int((py - y0) / CELL)
            if 0 <= cx < w and 0 <= cy < h:
                solid[cy * w + cx] = 1
    pad = int(BODY / 2 / CELL)
    return {"w": w, "h": h, "x0": x0, "y0": y0, "raw": solid,
            "solid": _grow(solid, w, h, pad), "pad": pad}


def _grow(sol, w, h, r):
    out = bytearray(sol)
    for cy in range(h):
        for cx in range(w):
            if not sol[cy * w + cx]:
                continue
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= nx < w and 0 <= ny < h:
                        out[ny * w + nx] = 1
    return out


class Nav:
    # follows a bfs path to the next spawn zone, replanning when it stalls

    ARRIVE = 56.0        # radius that counts as reaching a zone
    TOL = 5.0            # degrees of heading error we accept before turning
    TMIN, TMAX, DEG = 3, 8, 4.4
    PATIENCE = 8         # nav steps without 8 units of progress before a replan

    def __init__(self, grid, spawns=SPAWNS):
        self.g = grid
        self.w, self.h = grid["w"], grid["h"]
        self.sol = grid["solid"]
        self.spawns = spawns
        self.rot = 0
        self.blocked = set()
        self.pts = None
        self.tgtw = None
        self.best = 1e18
        self.since = 0
        self.stalls = 0
        self.arrivals = 0

    def c2w(self, c):
        return self.g["x0"] + (c[0] + 0.5) * CELL, self.g["y0"] + (c[1] + 0.5) * CELL

    def w2c(self, x, y):
        return int((x - self.g["x0"]) / CELL), int((y - self.g["y0"]) / CELL)

    def free(self, c):
        return (0 <= c[0] < self.w and 0 <= c[1] < self.h
                and not self.sol[c[1] * self.w + c[0]] and c not in self.blocked)

    def nearest(self, c):
        if self.free(c):
            return c
        for r in range(1, 20):
            cand = []
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    n = (c[0] + dx, c[1] + dy)
                    if self.free(n):
                        cand.append((dx * dx + dy * dy, n))
            if cand:
                return min(cand)[1]
        return None

    def bfs(self, src, dst):
        if src is None or dst is None:
            return None
        prev = {src: None}
        dq = collections.deque([src])
        while dq:
            c = dq.popleft()
            if c == dst:
                break
            for dx, dy in N8:
                n = (c[0] + dx, c[1] + dy)
                if n in prev or not self.free(n):
                    continue
                if dx and dy and not (self.free((c[0] + dx, c[1])) and self.free((c[0], c[1] + dy))):
                    continue        # no corner cutting
                prev[n] = c
                dq.append(n)
        if dst not in prev:
            return None
        out = []
        c = dst
        while c is not None:
            out.append(c)
            c = prev[c]
        return out[::-1]

    def los(self, a, b):
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 12))
        for k in range(n + 1):
            c = self.w2c(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
            if not self.free(c):
                return False
        return True

    def straighten(self, i0, win=25):
        seg = self.pts[i0:i0 + win]
        if len(seg) < 3:
            return seg
        out = [seg[0]]
        i, n = 0, len(seg)
        while i < n - 1:
            j = n - 1
            while j > i and not self.los(seg[i], seg[j]):
                j -= 1
            if j <= i:
                j = i + 1
            out.append(seg[j])
            i = j
        return out

    def plan(self, x, y, rotate=False):
        if rotate:
            self.rot += 1
        for _ in range(len(self.spawns)):
            tx, ty = self.spawns[self.rot % len(self.spawns)]
            cells = self.bfs(self.nearest(self.w2c(x, y)), self.nearest(self.w2c(tx, ty)))
            if cells is not None and len(cells) > 1:
                self.pts = [self.c2w(c) for c in cells]
                self.tgtw = self.c2w(self.nearest(self.w2c(tx, ty)))
                self.best = 1e18
                self.since = 0
                return True
            self.rot += 1
        self.pts = None
        return False

    def project(self, px, py):
        # nearest point on the path, and how far along it that is
        best, s = (1e18, 0.0), 0.0
        for i in range(len(self.pts) - 1):
            ax, ay = self.pts[i]
            bx, by = self.pts[i + 1]
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy)
            t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (L * L)))
            d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
            if d < best[0]:
                best = (d, s + t * L)
            s += L
        return best

    def at_arc(self, pts, dist):
        s = 0.0
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            L = math.hypot(bx - ax, by - ay)
            if s + L >= dist and L > 0:
                t = (dist - s) / L
                return ax + t * (bx - ax), ay + t * (by - ay)
            s += L
        return pts[-1]

    def act(self, x, y, ang, moved):
        # moved is the world displacement since the last nav step, or None when the
        # previous decision was not one: a combat step is not a stall
        if self.pts is None and not self.plan(x, y):
            return FWD, 4
        dc = math.hypot(self.tgtw[0] - x, self.tgtw[1] - y)
        if dc < self.ARRIVE:
            self.arrivals += 1
            self.plan(x, y, rotate=True)
            if self.pts is None:
                return FWD, 4
            dc = math.hypot(self.tgtw[0] - x, self.tgtw[1] - y)
        if moved is not None:
            if dc < self.best - 8.0:
                self.best, self.since = dc, 0
            else:
                self.since += 1
        if self.since >= self.PATIENCE:
            self.stalls += 1
            c = self.w2c(x, y)
            if self.free(c):
                self.blocked.add(c)
            self.plan(x, y, rotate=self.stalls % 3 == 0)
            if self.pts is None:
                return FWD, 4
            dc = math.hypot(self.tgtw[0] - x, self.tgtw[1] - y)
        _, s = self.project(x, y)
        ahead = max(24.0, min(96.0, dc - 32.0))
        front = self.straighten(max(0, int(s / 16) - 2))
        wx, wy = self.at_arc(front if len(front) > 1 else self.pts, ahead)
        err = (math.degrees(math.atan2(wy - y, wx - x)) - ang + 540.0) % 360.0 - 180.0
        if abs(err) <= self.TOL:
            return FWD, 6
        return (TL if err > 0 else TR), int(max(self.TMIN, min(self.TMAX, round(abs(err) / self.DEG) + 1)))


def load(wad):
    # grid for the arena, with an extra cell of clearance if that keeps the map connected
    g = build(wad)
    wide = dict(g, solid=_grow(g["raw"], g["w"], g["h"], g["pad"] + 1))
    nav = Nav(wide)
    src = nav.nearest(nav.w2c(*SPAWNS[0]))
    ok = sum(1 for x, y in SPAWNS[1:] if nav.bfs(src, nav.nearest(nav.w2c(x, y))))
    return wide if ok == len(SPAWNS) - 1 else g
