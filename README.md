# Sailing Simulator

A PyGame sailing school. The boat stays in the middle of the screen, the wind
changes gently, and the physics model tracks trim, tack, point of sail, speed,
heel, capsize risk, traffic, and simplified right-of-way rules.

The game is organized into seven levels that teach sailing from the basics to
advanced maneuvers, selected from a menu:

1. **First Reach** — steering and sail trim on a beam reach
2. **Points of Sail** — retrimming for beam, broad, close-hauled legs
3. **Tack the Ladder** — beating upwind through a zig-zag of gates
4. **Gybe Slalom** — controlled gybes downwind, first rocks appear
5. **Rock Channel** — precise handling through a rock-lined channel
6. **Harbor Traffic** — right-of-way rules against moving traffic
7. **Regatta Day** — full course with gusty wind, rocks and busy traffic

Rocks are solid: hitting one wrecks the boat. Capsizing or colliding with
another vessel does too. After a failure a three-second countdown runs and the
boat respawns at the last rounded buoy, pointed on a sailable course.

Traffic vessels patrol waypoint routes and follow simplified COLREGS: the
give-way vessel alters course to starboard and slows down, the stand-on vessel
holds its course. The right-of-way trainer checks, in order:

- Overtaking vessel keeps clear (regardless of sail or power)
- Power gives way to sail
- Port tack gives way to starboard tack
- Windward boat keeps clear of leeward boat on the same tack
- Power/power: head-on both alter to starboard, crossing gives way to starboard

A red ring marks a vessel you must give way to; passing too close as the
give-way boat gives a red penalty flash and logs the violated rule.

## Run

```bash
.venv/bin/python -m sailing_simulator.game
```

## Controls

Menu:

- Up / Down or mouse: select level
- Enter, Space, click or 1-7: start level
- Esc: quit

On the water:

- Left / A: tiller to port
- Right / D: tiller to starboard
- Up / W: trim in the main sheet
- Down / S: ease the main sheet
- Mouse: drag the main sheet lever
- N: show or hide the red no-sail-zone overlay
- Space: center the tiller
- Esc: back to the menu

## Test

```bash
.venv/bin/python -m pytest
```
