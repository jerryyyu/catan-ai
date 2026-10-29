# Experiments

`heuristic-v-random.json` is the first DEV screen. Its pass rule was declared as:
all 100 games complete and the candidate win-rate Wilson 95% lower bound is at
least 0.35.

The 2026-10-29 implementation check completed 100/100 games in 0.85 seconds.
The heuristic won 44 (44.0%, Wilson 95% interval 34.67%–53.77%), so the screen
**failed** its declared lower-bound bar by 0.33 percentage points. This establishes
that the runner, seat rotation, truncation guard, and fail-closed pass rule work; it
does not promote the heuristic. Improve the policy, preregister a new seed block,
and write a new config for the next comparison.
