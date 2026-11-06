"""state text: pipe-separated fields, always ending in the unit legend."""

import math

from .nav import SPAWNS

ENEMIES = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "Demon", "DoomImp", "Cacodemon", "BaronOfHell",
           # the wad replaces these five with low-press copies, which keep their own names
           "LowPressZombieman", "LowPressShotgunGuy", "LowPressChaingunGuy",
           "LowPressDemon", "LowPressHellKnight",
           # wad-only marine; it saws at 32-54 units and is the one enemy that closes fast
           "MarineChainsawVzd"}
ITEMS = {"Medikit", "Stimpack", "HealthBonus", "Soulsphere", "Megasphere",
         "GreenArmor", "BlueArmor", "ArmorBonus"}

ZONES = [(round(x), round(y)) for x, y in SPAWNS]
CLUSTER = 600.0      # radius of the local spawn-zone census
CENSUS = 1600.0      # radius of the 360-degree enemy census
REAR = 90.0

BANDS = ["front", "front-right", "right", "back-right",
         "back", "back-left", "left", "front-left"]

UNITS = ("units: map units, body 32 wide; contact<64 close<256 medium<768 far beyond; "
         "bearing 0 ahead, positive left, negative right, +-90 side, 180 behind")


def band(closeness):
    if closeness >= 0.97:
        return "contact"
    if closeness >= 0.93:
        return "close"
    if closeness >= 0.88:
        return "medium"
    return "open"


def rel(x, y, px, py, ang):
    # distance and bearing to a world point, matching the label convention
    d = math.hypot(x - px, y - py)
    a = math.degrees(math.atan2(x - px, -(y - py)))
    return d, (a - ang + 540.0) % 360.0 - 180.0


def spawn_block(px, py, ang):
    rows = sorted((rel(x, y, px, py, ang) + (x, y, i) for i, (x, y) in enumerate(ZONES)))
    (d1, b1, x1, y1, i1), (d2, b2, x2, y2, i2) = rows[0], rows[1]
    near = sum(1 for r in rows if r[0] <= CLUSTER)
    return ("spawn nearest (%d,%d) zone%d dist %d bearing %+.0fdeg | "
            "second (%d,%d) zone%d dist %d bearing %+.0fdeg | zones_in_600 %d"
            % (x1, y1, i1 + 1, round(d1), b1, x2, y2, i2 + 1, round(d2), b2, near))


def contact_block(blobs):
    # 360-degree enemy picture from objects, not the automap: the automap
    # cannot tell a monster from a stimpack
    if not blobs:
        return ("rear contact none | rear_contacts 0 | contacts_360 0 | "
                "contact_in_fov no | nearest contact none")
    (d0, b0) = blobs[0]
    rear = [b for b in blobs if abs(b[1]) > REAR]
    head = ("rear contact nearest dist %d bearing %+.0fdeg" % (round(rear[0][0]), rear[0][1])
            if rear else "rear contact none")
    return ("%s | rear_contacts %d | contacts_360 %d | contact_in_fov %s | "
            "nearest contact dist %d bearing %+.0fdeg"
            % (head, len(rear), sum(1 for b in blobs if b[0] <= CENSUS),
               "yes" if abs(b0) <= 45.0 else "no", round(d0), b0))


def picture(objects, px, py, ang):
    # (distance, bearing) for every enemy in the level, nearest first
    return sorted(rel(o.position_x, o.position_y, px, py, ang)
                  for o in objects or () if o.name in ENEMIES)


def text(v, labels, prof, contact, objects):
    hp, ar = v.get("HEALTH", 0.0), v.get("ARMOR", 0.0)
    am = v.get("SELECTED_WEAPON_AMMO", -1.0)
    # field order and the armour/ammo repeat are fixed: the adapter was trained on this exact string
    parts = ["health %.0f" % hp, "armor %.0f" % ar, "ammo %.0f" % am,
             "kills %.0f" % v.get("KILLCOUNT", 0.0),
             "weapon %.0f ammo %.0f armor %.0f" % (v.get("SELECTED_WEAPON", -1.0), am, ar)]

    seen = sorted((l for l in labels if l.object_name in ENEMIES),
                  key=lambda l: -(l.width * l.height))
    if seen:
        parts.append("enemies " + " ".join(
            "%+.2f/%d" % ((l.x + l.width / 2 - 160.0) / 160.0, int(l.width * l.height))
            for l in seen[:3]))
    else:
        parts.append("enemies none")

    near = max((l for l in labels if l.object_name in ENEMIES),
               key=lambda l: l.width * l.height, default=None)
    parts.append("enemy none" if near is None else
                 "enemy %s offset %+.3f size %.0f" % (near.object_name,
                                                      (near.x + near.width / 2 - 160.0) / 160.0,
                                                      near.width * near.height))

    item = max((l for l in labels if l.object_name in ITEMS),
               key=lambda l: l.width * l.height, default=None)
    parts.append("health item none" if item is None else
                 "health item %s offset %+.3f size %.0f" % (item.object_name,
                                                            (item.x + item.width / 2 - 160.0) / 160.0,
                                                            item.width * item.height))

    parts.append("threat none" if contact is None else
                 "threat bearing %+.0fdeg dist %.0f" % contact)
    parts.append("walls " + " ".join("%s:%s" % (n, band(prof[i])) for i, n in enumerate(BANDS)))

    px, py, ang = v.get("POSITION_X", 0.0), v.get("POSITION_Y", 0.0), v.get("ANGLE", 0.0)
    parts.append(spawn_block(px, py, ang))
    parts.append(contact_block(picture(objects, px, py, ang)))
    parts.append(UNITS)
    return " | ".join(parts)
