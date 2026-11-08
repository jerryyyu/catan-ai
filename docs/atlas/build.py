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
    for pol in reg.get('policies', []):
        vp = pol.get('vs_prod') or {}
        for t in TABLES:
            v = vp.get(t)
            if v is None:
                continue
            point, lo, hi = v
            if (lo is None) != (hi is None) or (lo is not None and not (lo <= point <= hi)):
                problems.append(f"policy {pol['name']}/{t}: bad interval {v}")
        for k, v in (pol.get('external') or {}).items():
            if k not in {c['key'] for c in reg.get('external_columns', [])}:
                problems.append(f"policy {pol['name']}: unknown external column {k}")
            w, lo, hi, n = v
            if not (0 <= lo <= w <= hi <= 1) or n <= 0:
                problems.append(f"policy {pol['name']}/{k}: bad win share {v}")
        if not pol.get('search') or not pol.get('model'):
            problems.append(f"policy {pol['name']}: search and model must be stated")
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


def _lift_cell(v) -> str:
    if v is None:
        return '<td class="n cell none">—</td>'
    p, lo, hi = v
    kind = 'good' if (lo is not None and lo > 0) else ('bad' if (hi is not None and hi < 0) or (lo is None and p < 0) else 'mid')
    ci = f'<span class="ci-t">{lo * 100:+.1f} to {hi * 100:+.1f}</span>' if lo is not None else '<span class="ci-t">no interval</span>'
    return f'<td class="n cell {kind}"><b>{p * 100:+.1f}</b>{ci}</td>'


def _share_cell(v) -> str:
    if v is None:
        return '<td class="n cell none">—</td>'
    w, lo, hi, n = v
    retired = w >= 0.5 and lo > 0.4
    kind = 'good' if retired else ('ok' if lo > 0.25 else ('bad' if hi < 0.25 else 'mid'))
    tag = ' · ret.' if retired else ''
    return (f'<td class="n cell {kind}"><b>{w * 100:.1f}%</b>'
            f'<span class="ci-t">{lo * 100:.0f}–{hi * 100:.0f} · n{n}{tag}</span></td>')


def policy_matrix(reg: dict) -> str:
    ext = reg.get('external_columns', [])
    head1 = ('<tr class="grp"><th colspan="3"></th><th colspan="4" class="g">vs production '
             f'(<code>{esc(reg["production"]["served"])}</code>): lift, pp</th>'
             f'<th colspan="{len(ext)}" class="g">vs external AIs: win share</th></tr>')
    head2 = ('<tr><th>Policy</th><th>Search</th><th>Value model</th>'
             '<th class="n">Tactical</th><th class="n">Mixed</th><th class="n">Prod copies</th><th class="n">Seeds</th>'
             + ''.join(f'<th class="n">{esc(c["label"])}</th>' for c in ext) + '</tr>')
    body, group = [], None
    for pol in reg['policies']:
        if pol['group'] != group:
            group = pol['group']
            body.append(f'<tr class="section"><td colspan="{7 + len(ext)}">{esc(group)}</td></tr>')
        vp = pol.get('vs_prod')
        if vp is None:
            lift = ('<td class="n cell none" colspan="4">baseline</td>' if pol['status'] == 'served'
                    else '<td class="n cell none" colspan="4">not yet measured</td>')
        else:
            lift = ''.join(_lift_cell(vp.get(t)) for t in TABLES) + f'<td class="n dim">{vp["seeds"]}</td>'
        shares = ''.join(_share_cell((pol.get('external') or {}).get(c['key'])) for c in ext)
        status = pol['status']
        chip = 'pass' if status == 'served' else ('fail' if 'FAIL' in status else 'run')
        body.append(f'<tr><td class="pol"><code>{esc(pol["name"])}</code><span class="chip {chip}">{esc(status)}</span></td>'
                    f'<td class="txt">{esc(pol["search"])}</td><td class="txt">{esc(pol["model"])}</td>{lift}{shares}</tr>')
    return f'<div class="scroll matrix"><table>{head1}{head2}{"".join(body)}</table></div>'


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
        '</tr>' for m in reg['models'])
    return ('<div class="scroll"><table><tr><th>Value model</th><th class="n">Test AUC</th>'
            '<th class="n">Pick gain</th><th>Played as</th></tr>' + rows + '</table></div>')


def render(reg: dict) -> str:
    prod, ev = reg['production'], reg['evaluation']
    css = (HERE / 'style.css').read_text()
    ais = ''.join(f'<tr><td>{esc(a["name"])}</td><td>{esc(a["search"])}</td><td>{esc(a["hidden"])}</td>'
                  f'<td class="n">{esc(a["per_decision"])}</td><td><span class="chip flat">{esc(a["status"])}</span></td></tr>'
                  for a in reg['external_ais'])
    rules = ''.join('<tr>' + ''.join(f'<td>{esc(c)}</td>' for c in row) + '</tr>' for row in ev['current_rules'])
    flight = ''.join(f'<tr><td>{esc(f["name"])}</td><td class="dim">{esc(f["where"])}</td>'
                     f'<td><span class="chip run">{esc(f["status"])}</span></td></tr>' for f in reg['in_flight'])
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
  <h2>Every policy at a glance</h2>
  <div class="legend"><span class="cell good"><b>green</b></span> CI above 0 / AI retired <span class="cell bad"><b>red</b></span> CI below 0 / below 25%</div>
  {policy_matrix(reg)}
</section>

<section>
  <h2>Evaluation strategy <span class="chip run">{esc(ev['status'])}</span></h2>
  <div class="scroll"><table><tr><th>Step</th><th>Opponents</th><th>Bar</th></tr>{rules}</table></div>
</section>

<section>
  <h2>External AIs</h2>
  <div class="scroll"><table><tr><th>AI</th><th>Search</th><th>Hidden cards</th><th class="n">Time / move</th><th>Status</th></tr>{ais}</table></div>
</section>

<section>
  <h2>Value models</h2>
  {model_table(reg)}
</section>

<section>
  <h2>In flight</h2>
  <div class="scroll"><table>{flight}</table></div>
</section>
<footer class="dim">Source <code>docs/atlas/registry.json</code> · served trace <code>{esc(prod['served_trace_sha256'][:12])}</code> · candidate trace <code>{esc(prod['candidate_trace_sha256'][:12])}</code></footer>
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
