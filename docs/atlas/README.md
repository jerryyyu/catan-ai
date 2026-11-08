# Catan Bot Atlas

`registry.json` is the only place a bot, screen, external result or model is entered. `build.py` renders it to
`atlas.html`, which is built locally (gitignored) and published as an artifact; never hand-edit the HTML.

```sh
cd docs/atlas
python3 build.py            # write atlas.html
python3 build.py --check    # exit 1 if an invariant breaks or a built atlas.html is stale; prints CONSISTENT otherwise
python3 -m pytest -q test_build.py
```

Every number must come from a named receipt (see docs/AI_POLICIES.md for receipt names). The evaluation-strategy
section is marked `evolving` while the owner settles how candidates are judged.
