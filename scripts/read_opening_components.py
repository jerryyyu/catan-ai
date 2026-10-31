"""Read the complete preregistered opening component diagnostic."""
import json
from pathlib import Path


def main():
    records = [json.loads(line) for line in Path('runs/opening-components-v1.jsonl').read_text().splitlines()]
    metadata, rows = records[0], records[1:]
    expected = {(panel, seed, seat) for panel, start in [('tactical', 1429000), ('mixed', 1429100)]
                for seed in range(start, start + 4) for seat in range(4)}
    assert len(rows) == len(expected)
    assert {(r['panel'], r['seed'], r['seat']) for r in rows} == expected
    output = {}
    for panel in ('tactical', 'mixed'):
        selected = [r for r in rows if r['panel'] == panel]
        output[panel] = {}
        for arm in metadata['arms']:
            results = [r[arm] for r in selected]
            wins = sum(r[arm]['winner'] == r['seat'] for r in selected)
            lift = sum(int(r[arm]['winner'] == r['seat']) - int(r['baseline']['winner'] == r['seat'])
                       for r in selected) / len(selected)
            truncated = sum(r['truncated'] for r in results)
            maximum = max(r['maximum_step'] for r in results)
            output[panel][arm] = {
                'games': len(selected), 'wins': wins, 'paired_lift': lift,
                'truncated': truncated, 'maximum_step': maximum,
                'pass': lift >= .05 and truncated == 0 and maximum <= 2,
                'mean_setup_value': sum(r['setup']['value'] for r in results) / len(results),
                'settlement_pair_changed': sum(r[arm]['setup']['settlements'] != r['baseline']['setup']['settlements'] for r in selected),
                **{key + '_by_turn40': sum(r[key] is not None and r[key] <= 40 for r in results)
                   for key in ('first_city', 'first_expansion')},
                **{key + '_not_reached': sum(r[key] is None for r in results)
                   for key in ('first_city', 'first_expansion')},
            }
    print(json.dumps(output, indent=2))
    Path('runs/opening-components-v1-summary.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
