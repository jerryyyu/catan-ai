# Catan AI agent workflow

Build and measure the real engine-to-policy-to-player path before adding research
orchestration. The Python engine is authoritative; browser and simulation clients
submit the same actions and may not reimplement legality.

- Every run is seeded. Persist the config, source revision, per-game outcomes,
  truncation count, uncertainty interval, and elapsed time.
- Give a new policy idea a cheap DEV comparison and a numeric pass bar before it
  runs. A failed bar is a result; do not tune the bar after seeing it.
- Rotate candidate policies through all seats on the same seed block. Separate
  gameplay-strength, correctness, latency, and calibration claims.
- Never expose hidden resources through player-facing state. Add a boundary test
  whenever the private state grows.
- Preserve completed results. A new producer or changed config requires a new run;
  documentation or receipt changes do not invalidate game outcomes.
- Keep the initial rules exclusions explicit. Add development cards, trading,
  ports, and bonus scoring as engine semantics with tests before teaching bots to
  use them.

This is an unofficial research project. Use original visual assets and do not copy
commercial game art, rulebook text, or branding.
