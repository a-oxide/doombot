"""the option text and the executed hold must agree."""

import re

from dmodel.actions import NAV, OPTIONS11, OPTIONS12, ACTIONS12, TICS11, TICS12, row_tic

DEG = 3.5      # degrees per tic the texts were written against


def test_text_carries_its_tic_count():
    for text, tics in zip(OPTIONS11, TICS11):
        assert row_tic(text) == tics, text


def test_turn_text_agrees_with_its_degrees():
    for text, tics in zip(OPTIONS11, TICS11):
        m = re.search(r"([\d.]+) deg", text)
        if m:
            assert abs(float(m.group(1)) - tics * DEG) <= 1.0, text


def test_nav_option_is_a_sentinel():
    assert len(OPTIONS12) == len(TICS12) == len(ACTIONS12) == 12
    assert NAV == 11
    assert ACTIONS12[NAV] == -1
    assert row_tic(OPTIONS12[NAV]) is None


if __name__ == "__main__":
    test_text_carries_its_tic_count()
    test_turn_text_agrees_with_its_degrees()
    test_nav_option_is_a_sentinel()
    print("ok")
