"""twelve-way action table: eleven combat controls plus one navigation sentinel."""

import re

DPT = 3.076          # degrees per tic, measured in-engine
FOV = 45.0           # degrees per unit of normalised screen offset

# each text carries its own tic count so the prompt cannot drift from what the
# harness executes; tests/test_actions.py enforces the match
OPTIONS11 = ["advance and fire, 6 tics",
             "advance, 6 tics",
             "fire, 4 tics",
             "turn left 1 tic, 3.5 deg",
             "turn left 4 tics, 14 deg",
             "turn left 8 tics, 28 deg",
             "turn left 14 tics, 49 deg",
             "turn right 1 tic, 3.5 deg",
             "turn right 4 tics, 14 deg",
             "turn right 8 tics, 28 deg",
             "turn right 14 tics, 49 deg"]
ACTIONS11 = [0, 5, 6, 3, 3, 3, 3, 4, 4, 4, 4]      # row in the doom button table
TICS11 = [6, 6, 4, 1, 4, 8, 14, 1, 4, 8, 14]       # hold length per row

NAV = 11
NAV_TEXT = "navigate: patrol-controller step"
OPTIONS12 = OPTIONS11 + [NAV_TEXT]
ACTIONS12 = ACTIONS11 + [-1]        # -1 never reaches vizdoom; the harness runs the patrol
TICS12 = TICS11 + [0]

FIRE = 6                            # action row that holds the trigger standing still


def row_tic(text):
    # tic count baked into the option text; the nav option has none
    m = re.search(r"(\d+) tics?\b", text)
    return int(m.group(1)) if m else None
