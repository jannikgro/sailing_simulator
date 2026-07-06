# Sailing Simulator

A small PyGame sailing simulator. The boat stays in the middle of the screen,
the wind changes gently, and the physics model tracks trim, tack, point of
sail, speed, heel, capsize risk, traffic, and simplified right-of-way rules.
The buoy course starts with an upwind zig-zag leg that requires tacking, then
turns into a downwind run back toward the finish.

Traffic includes motorboats and sailboats. The right-of-way trainer checks:

- Overtaking vessel keeps clear
- Sail generally stands on over power
- Port tack gives way to starboard tack
- Windward boat keeps clear of leeward boat on the same tack

If you are the give-way boat and pass too close, the game gives a red penalty
flash and logs the violated rule.

## Run

```bash
.venv/bin/python -m sailing_simulator.game
```

## Controls

- Left / A: tiller to port
- Right / D: tiller to starboard
- Up / W: trim in the main sheet
- Down / S: ease the main sheet
- Mouse: drag the main sheet lever
- N: show or hide the red no-sail-zone overlay
- Space: center the tiller
- R: right the boat after a capsize
- Esc: quit

## Test

```bash
.venv/bin/python -m pytest
```
