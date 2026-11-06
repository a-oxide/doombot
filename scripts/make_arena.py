"""build the arena wad and its cfg from a stock deathmatch.wad.

the monsters keep vanilla behaviour but drop to 10 health, so a pistol kill takes
a couple of hits and the fight is winnable with the starting weapon. the chainsaw
marine is in the same wad and already counted as a kill; it just needed the lower
health too.
"""

import argparse
import pathlib
import struct

DECORATE = """ACTOR MarineChainsawVzd: MarineChainsaw
{
  +COUNTKILL
  Health 10
}
ACTOR LowPressZombieman : Zombieman replaces Zombieman
{
  Health 10
}
ACTOR LowPressShotgunGuy : ShotgunGuy replaces ShotgunGuy
{
  Health 10
}
ACTOR LowPressChaingunGuy : ChaingunGuy replaces ChaingunGuy
{
  Health 10
}
ACTOR LowPressDemon : Demon replaces Demon
{
  Health 10
}
ACTOR LowPressHellKnight : HellKnight replaces HellKnight
{
  Health 10
}
"""

CFG = """doom_scenario_path = %s
doom_skill = 3
screen_resolution = RES_320X240
render_hud = false
render_screen_flashes = false
episode_start_time = 1
episode_timeout = 4200
available_buttons =
    {
        ATTACK
        SPEED
        STRAFE
        MOVE_RIGHT
        MOVE_LEFT
        MOVE_BACKWARD
        MOVE_FORWARD
        TURN_RIGHT
        TURN_LEFT
        SELECT_WEAPON1
        SELECT_WEAPON2
        SELECT_WEAPON3
        SELECT_WEAPON4
        SELECT_WEAPON5
        SELECT_WEAPON6
        SELECT_NEXT_WEAPON
        SELECT_PREV_WEAPON
        LOOK_UP_DOWN_DELTA
        TURN_LEFT_RIGHT_DELTA
        MOVE_LEFT_RIGHT_DELTA
    }
available_game_variables =
    {
        KILLCOUNT
        HEALTH
        ARMOR
        SELECTED_WEAPON
        SELECTED_WEAPON_AMMO
    }
mode = PLAYER
"""


def read(path):
    data = pathlib.Path(path).read_bytes()
    n, off = struct.unpack_from("<ii", data, 4)
    out = []
    for i in range(n):
        lo, size, name = struct.unpack_from("<ii8s", data, off + 16 * i)
        out.append((name.rstrip(b"\0").decode("latin1"), data[lo:lo + size]))
    return out


def write(path, lumps):
    """data first, directory last, the way every doom iwad is laid out."""
    body, table = bytearray(), bytearray()
    for name, data in lumps:
        table += struct.pack("<ii8s", 12 + len(body), len(data), name.encode().ljust(8, b"\0"))
        body += data
    pathlib.Path(path).write_bytes(
        b"PWAD" + struct.pack("<ii", len(lumps), 12 + len(body)) + body + table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="stock deathmatch.wad (default: vizdoom scenarios)")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    base = args.base
    if base is None:
        import vizdoom
        base = str(pathlib.Path(vizdoom.scenarios_path) / "deathmatch.wad")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wad = out / "deathmatch-lowpress.wad"
    write(wad, [(n, DECORATE.encode("latin1") if n == "DECORATE" else d) for n, d in read(base)])
    (out / "arena.cfg").write_text(CFG % wad.name)
    print("%s (%d bytes)\n%s" % (wad, wad.stat().st_size, out / "arena.cfg"))


if __name__ == "__main__":
    main()
