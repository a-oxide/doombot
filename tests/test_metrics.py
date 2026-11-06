"""kill attribution and the aggregate that the report quotes."""

from dmodel.env import kill_delta, summarise


def test_infight_kills_are_not_ours():
    assert kill_delta((4, 100), (6, 100)) == 0.0          # two monsters, no new damage
    assert kill_delta((4, 100), (6, 160)) == 2.0
    assert kill_delta((4, 100), (4, 160)) == 0.0


def test_deaths_come_from_the_death_flag():
    rows = [{"tics": 100, "kills": 2, "player_kills": 2.0, "damage": 20, "damage_taken": 50,
             "dead": True},
            {"tics": 300, "kills": 0, "player_kills": 0.0, "damage": 0, "damage_taken": 0,
             "dead": False}]
    s = summarise(rows)
    assert s["deaths"] == 1
    assert s["tics"] == 400
    assert abs(s["damage_per_1000"] - 50.0) < 1e-9
    assert abs(s["damage"] - 10.0) < 1e-9


if __name__ == "__main__":
    test_infight_kills_are_not_ours()
    test_deaths_come_from_the_death_flag()
    print("ok")
