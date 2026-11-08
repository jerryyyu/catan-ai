"""Render the Catan atlas from registry.json (the only source of truth; never hand-edit atlas.html).

    python3 build.py           # write atlas.html next to the registry
    python3 build.py --check   # exit 1 if the registry breaks an invariant or a built atlas.html is stale
"""
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / 'registry.json'
OUT = HERE / 'atlas.html'
TABLES = ('tactical', 'mixed', 'self')
VERDICTS = {'PASS', 'FAIL', 'measurement'}


def check_registry(reg: dict) -> list[str]:
    problems = []
    if reg.get('evaluation', {}).get('status') != 'evolving' and not reg.get('evaluation', {}).get('current_rules'):
        problems.append('evaluation section needs current_rules')
    ids = set()
    for s in reg['screens']:
        if s['id'] in ids:
            problems.append(f"duplicate screen id {s['id']}")
        ids.add(s['id'])
        if s.get('verdict') not in VERDICTS:
            problems.append(f"{s['id']}: verdict {s.get('verdict')!r} not in {sorted(VERDICTS)}")
        if not s.get('tables'):
            problems.append(f"{s['id']}: no tables")
        for t, v in s.get('tables', {}).items():
            if t not in TABLES:
                problems.append(f"{s['id']}: unknown table {t}")
            point, lo, hi = v
            if (lo is None) != (hi is None):
                problems.append(f"{s['id']}/{t}: half an interval")
            if lo is not None and not (lo <= point <= hi):
                problems.append(f"{s['id']}/{t}: point {point} outside [{lo}, {hi}]")
    for e in reg['external']:
        lo, hi = e['ci']
        if not (0 <= lo <= e['win'] <= hi <= 1):
            problems.append(f"{e['receipt']}: win {e['win']} outside [{lo}, {hi}]")
        if not e.get('receipt'):
            problems.append(f"{e['opponent']}/{e['bot']}: no receipt")
    return problems


def esc(x) -> str:
    return html.escape(str(x))


def fmt(v, signed=True) -> str:
    if v is None:
        return '—'
    return f'{v:+.3f}' if signed else f'{v:.3f}'


def forest(reg: dict) -> str:
    """One forest plot per table: paired lift vs served, with 95% interval, zero line and +5 pp bar."""
    rows = [s for s in reg['screens'] if s['vs'] == reg['production']['served']]
    lo_x, hi_x = -0.10, 0.18
    w, left, right, row_h, top = 640, 210, 20, 22, 28
    span = w - left - right

    def x(v):
        return left + (v - lo_x) / (hi_x - lo_x) * span
    out = []
    for table in TABLES:
        entries = [(s, s['tables'][table]) for s in rows if table in s['tables']]
        h = top + row_h * len(entries) + 26
        parts = [f'<svg class="forest" viewBox="0 0 {w} {h}" role="img" aria-label="Paired lift vs served, {table} table">']
        for tick in (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15):
            cls = 'zero' if tick == 0 else ('bar' if abs(tick - 0.05) < 1e-9 else 'grid')
            parts.append(f'<line class="{cls}" x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top - 8}" y2="{h - 22}"/>')
            parts.append(f'<text class="tick" x="{x(tick):.1f}" y="{h - 8}" text-anchor="middle">{tick * 100:+.0f}</text>')
        parts.append(f'<text class="axis" x="{x(0.05) + 4:.1f}" y="{top - 12}">+5 pp bar</text>')
        for i, (s, (p, lo, hi)) in enumerate(entries):
            y = top + i * row_h + row_h / 2
            kind = 'pass' if (lo is not None and lo > 0) else ('neg' if p < 0 else 'flat')
            parts.append(f'<text class="label" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{esc(s["bot"])} '
                         f'<tspan class="dim">{s["seeds"]}</tspan></text>')
            if lo is not None:
                parts.append(f'<line class="ci {kind}" x1="{x(max(lo, lo_x)):.1f}" x2="{x(min(hi, hi_x)):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
            parts.append(f'<circle class="pt {kind}" cx="{x(max(min(p, hi_x), lo_x)):.1f}" cy="{y:.1f}" r="4.5"/>')
        parts.append('</svg>')
        out.append(f'<figure class="plot"><figcaption>{table.capitalize()} table</figcaption>{"".join(parts)}</figure>')
    return '\n'.join(out)


def ladder(reg: dict) -> str:
    entries = reg['external']
    w, left, right, row_h, top = 640, 300, 20, 26, 24
    span = w - left - right
    h = top + row_h * len(entries) + 26

    def x(v):
        return left + v * span
    parts = [f'<svg class="forest" viewBox="0 0 {w} {h}" role="img" aria-label="Win share against external AIs">']
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        cls = 'fair' if tick == 0.25 else ('bar' if tick == 0.5 else 'grid')
        parts.append(f'<line class="{cls}" x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top - 8}" y2="{h - 22}"/>')
        parts.append(f'<text class="tick" x="{x(tick):.1f}" y="{h - 8}" text-anchor="middle">{tick:.2f}</text>')
    parts.append(f'<text class="axis" x="{x(0.25) + 4:.1f}" y="{top - 12}">fair share</text>')
    parts.append(f'<text class="axis" x="{x(0.5) + 4:.1f}" y="{top - 12}">retire at .50</text>')
    for i, e in enumerate(entries):
        y = top + i * row_h + row_h / 2
        retired = e['win'] >= 0.5 and e['ci'][0] > 0.4
        kind = 'pass' if retired else ('neg' if e['ci'][1] < 0.25 else 'flat')
        parts.append(f'<text class="label" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{esc(e["opponent"])} '
                     f'<tspan class="dim">· {esc(e["bot"].replace("planner_", ""))}</tspan></text>')
        parts.append(f'<line class="ci {kind}" x1="{x(e["ci"][0]):.1f}" x2="{x(e["ci"][1]):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<circle class="pt {kind}" cx="{x(e["win"]):.1f}" cy="{y:.1f}" r="4.5"/>')
    parts.append('</svg>')
    return f'<figure class="plot">{"".join(parts)}</figure>'


def screen_table(reg: dict) -> str:
    head = '<tr><th>Screen</th><th>Bot</th><th class="n">Seeds</th>' + ''.join(
        f'<th class="n">{t}</th>' for t in TABLES) + '<th>Verdict</th></tr>'
    body = []
    for s in reg['screens']:
        cells = []
        for t in TABLES:
            v = s['tables'].get(t)
            if v is None:
                cells.append('<td class="n dim">—</td>')
                continue
            p, lo, hi = v
            ci = f'<span class="ci-t">[{fmt(lo)}, {fmt(hi)}]</span>' if lo is not None else ''
            cells.append(f'<td class="n">{fmt(p)}{ci}</td>')
        verdict = s['verdict']
        body.append(f'<tr><td><code>{esc(s["id"])}</code><div class="note">{esc(s.get("note", ""))}</div></td>'
                    f'<td><code>{esc(s["bot"])}</code></td><td class="n">{s["seeds"]}</td>{"".join(cells)}'
                    f'<td><span class="chip {verdict.lower()}">{esc(verdict)}</span></td></tr>')
    return f'<div class="scroll"><table>{head}{"".join(body)}</table></div>'


def model_table(reg: dict) -> str:
    rows = ''.join(
        f'<tr><td>{esc(m["name"])}</td><td class="n">{fmt(m["auc"], signed=False) if m["auc"] else "—"}</td>'
        f'<td class="n">{fmt(m["xeval"]) if m["xeval"] is not None else "—"}</td>'
        f'<td>{"<code>" + esc(m["played"]) + "</code>" if m["played"] not in ("no", "pending") else esc(m["played"])}</td>'
        f'<td class="note">{esc(m["note"])}</td></tr>' for m in reg['models'])
    return ('<div class="scroll"><table><tr><th>Value model</th><th class="n">Test AUC</th>'
            '<th class="n">In-turn pick gain</th><th>Played as</th><th>Note</th></tr>' + rows + '</table></div>')


def render(reg: dict) -> str:
    prod, ev = reg['production'], reg['evaluation']
    css = (HERE / 'style.css').read_text()
    ais = ''.join(f'<article class="ai"><h4>{esc(a["name"])}</h4><p>{esc(a["decides"])}</p>'
                  f'<p class="dim">{esc(a["connection"])}</p><span class="chip flat">{esc(a["status"])}</span></article>'
                  for a in reg['external_ais'])
    flight = ''.join(f'<li><span class="chip run">{esc(f["status"])}</span> {esc(f["name"])} '
                     f'<span class="dim">· {esc(f["where"])}</span></li>' for f in reg['in_flight'])
    return f"""<title>Catan Bot Atlas</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}</style>
<div class="wrap">
<header class="mast">
  <div class="eyebrow">catan-ai · reconciled {esc(reg['reconciled'])}</div>
  <h1>Catan Bot Atlas</h1>
  <div class="prod">
    <div class="slot"><span class="k">Served</span><code>{esc(prod['served'])}</code><span class="dim">since {esc(prod['served_since'])}</span></div>
    <div class="slot cand"><span class="k">Candidate</span><code>{esc(prod['candidate'])}</code><span class="dim">{esc(prod['candidate_status'])}</span></div>
  </div>
</header>

<section>
  <h2>Evaluation strategy <span class="chip run">{esc(ev['status'])}</span></h2>
  <p class="lead">{esc(ev['owner_direction'])}</p>
  <div class="cols">
    <div><h3>Current rules</h3><ul>{''.join(f'<li>{esc(x)}</li>' for x in ev['current_rules'])}</ul></div>
    <div><h3>Open questions</h3><ul>{''.join(f'<li>{esc(x)}</li>' for x in ev['open_questions'])}</ul></div>
  </div>
</section>

<section>
  <h2>Screens against the served bot</h2>
  <p class="lead">Paired lift in win rate (percentage points) vs <code>{esc(prod['served'])}</code>, 95% interval where recorded. Green: interval above zero. The dashed line is the +5 pp DEV bar.</p>
  <div class="plots">{forest(reg)}</div>
  {screen_table(reg)}
</section>

<section>
  <h2>External AIs</h2>
  <p class="lead">One bot alone against three copies of an outside AI, rotated through every seat. Fair share is .25; an AI is retired as a benchmark for a bot once that bot wins at least .50 with the lower bound above .40.</p>
  {ladder(reg)}
  <div class="ais">{ais}</div>
</section>

<section>
  <h2>Value models</h2>
  <p class="lead">Same 100k self-play games and 5,000 held-out games. In-turn pick gain is the chosen option's playout win rate minus the served option's, per decision.</p>
  {model_table(reg)}
</section>

<section class="cols">
  <div><h2>What we have learned</h2><ul>{''.join(f'<li>{esc(x)}</li>' for x in reg['findings'])}</ul></div>
  <div><h2>In flight</h2><ul class="flight">{flight}</ul></div>
</section>
<footer class="dim">Built from <code>docs/atlas/registry.json</code> by <code>docs/atlas/build.py</code>. Served trace <code>{esc(prod['served_trace_sha256'][:12])}</code> · candidate trace <code>{esc(prod['candidate_trace_sha256'][:12])}</code>.</footer>
</div>
"""


def main(argv) -> int:
    reg = json.loads(REGISTRY.read_text())
    problems = check_registry(reg)
    page = render(reg)
    if '--check' in argv:
        if OUT.exists() and OUT.read_text() != page:
            problems.append('atlas.html is stale; run python3 build.py')
        print('\n'.join(problems) if problems else 'CONSISTENT')
        return 1 if problems else 0
    if problems:
        print('\n'.join(problems))
        return 1
    OUT.write_text(page)
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
