# Colonist move advisor (read-only)

Watches a game you are playing on Colonist and prints the served planner's
suggested move. It never clicks, types or sends anything to Colonist. Use it
only in private games whose players have agreed to it; Colonist's terms may
forbid assistance in public or ranked games.

Pipeline: screenshot (`screencapture`) → Claude reads the board into
`schema.Position` → `reconstruct.to_game` rebuilds it in the engine →
`planner_unified` chooses → the move is printed in board terms.

```sh
# One screenshot on disk
uv run --with anthropic python -m tools.colonist_advisor.advisor --image shot.png
# Watch the screen (or a region x,y,w,h) and advise whenever it changes; keep reads for review
uv run --with anthropic python -m tools.colonist_advisor.advisor --watch 5 --region 0,80,1400,900 --save runs/colonist-reads
# A position typed or corrected by hand (no API call)
uv run python -m tools.colonist_advisor.advisor --state position.json
```

Limits of this first version:

- The planner sees no game history, so it has no card-counting estimate of
  opponents' hands (robber, steal and trade advice is weaker than in our own games).
- Four-player base game only. The read is refused if terrain or number counts
  do not match the base game, or if it does not find exactly four players.
- Screen-read accuracy on Colonist is not yet measured. Check `uncertain`
  and the saved reads before trusting a suggestion.
