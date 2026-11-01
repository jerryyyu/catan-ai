"""Read-only move advisor for a Colonist game you are playing yourself.

It takes a screenshot (or reads a saved image or position file), asks Claude to
transcribe the visible board into ``schema.Position``, rebuilds that position in
the engine, and prints the served planner's suggested move. It never sends
mouse or keyboard input. Use it only in private games whose players agree to
it; Colonist's terms may forbid assistance in public or ranked games.

    uv run --with anthropic python -m tools.colonist_advisor.advisor --image shot.png
    uv run --with anthropic python -m tools.colonist_advisor.advisor --watch 5
    uv run python -m tools.colonist_advisor.advisor --state position.json
"""
import argparse
import base64
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from catan_ai.bots import make_policy
from catan_ai.observation import Observation

from .reconstruct import check, describe, to_game
from .schema import Position

MODEL = 'claude-opus-5-5'
INSTRUCTIONS = """Transcribe this Catan-style board game screenshot into the given schema. Read only what is visible; do not guess hidden cards.

Hex positions: the board has five rows of pointy-top hexes with 3, 4, 5, 4, 3 hexes, top to bottom. Row 0 is the top row; columns count from 0 at the left of each row.
Corners of a hex: top, top_right, bottom_right, bottom, bottom_left, top_left. Sides: top_right, right, bottom_right, bottom_left, left, top_left. Name each settlement/city by one corner of any hex it touches, and each road by one side of any hex it borders. For pieces on the coast, use the land hex.
Terrain: brick (hills), lumber (forest), wool (pasture), grain (fields), ore (mountains), desert.
Ports: list each harbor with the coastal side of the land hex it serves and its resource (null for 3:1).
Players: use each player's piece color as the color key everywhere. Mark the player whose hand is shown at the bottom as is_me. Visible points are the points shown publicly.
Phase: 'roll' if the current player has not rolled; 'actions' after rolling; 'move_robber' or 'steal' or 'discard' when the game is waiting for that; 'trade_response' if someone is offering me a trade; 'other_player' when it is not my decision.
List anything you could not read with confidence in 'uncertain'."""


def capture(region=None):
    path = Path(tempfile.mkstemp(suffix='.png')[1])
    command = ['screencapture', '-x', '-t', 'png']
    if region:
        command += ['-R', region]
    subprocess.run(command + [str(path)], check=True)
    return path


def read_position(image: Path) -> Position:
    import anthropic
    client = anthropic.Anthropic()
    data = base64.standard_b64encode(image.read_bytes()).decode()
    response = client.messages.parse(
        model=MODEL, max_tokens=16000, output_config={'effort': 'high'},
        messages=[{'role': 'user', 'content': [
            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': data}},
            {'type': 'text', 'text': INSTRUCTIONS}]}],
        output_format=Position)
    if response.stop_reason == 'refusal':
        raise RuntimeError(f'screen read declined: {response.stop_details}')
    return response.parsed_output


def advise(position: Position, policy_name: str) -> str:
    lines = []
    problems = check(position)
    if position.uncertain:
        lines.append('Unsure about: ' + '; '.join(position.uncertain))
    if problems:
        lines.append('Cannot advise, the read looks wrong: ' + '; '.join(problems))
        return '\n'.join(lines)
    me = next(p for p in position.players if p.is_me)
    if position.phase == 'other_player' or position.current_color != me.color and position.phase not in ('discard', 'trade_response'):
        lines.append(f'Waiting: it is {position.current_color}\'s turn.')
        return '\n'.join(lines)
    game, seat = to_game(position)
    view = Observation.from_game(game, seat)
    if not view.actions:
        lines.append(f'No legal move found for phase {position.phase}; check the read.')
        return '\n'.join(lines)
    action = make_policy(policy_name, 0).choose(view, seat)
    colors = [p.color for p in position.players]
    lines.append(f'Suggested ({policy_name}): {describe(action, game, colors)}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--image', type=Path, help='advise on a saved screenshot')
    source.add_argument('--state', type=Path, help='advise on a saved position JSON (no API call)')
    parser.add_argument('--region', help='screen region x,y,w,h to capture (default: whole screen)')
    parser.add_argument('--watch', type=float, help='re-capture every N seconds and advise when the screen changes')
    parser.add_argument('--save', type=Path, help='directory to keep each screenshot and its read position')
    parser.add_argument('--policy', default='planner_unified')
    args = parser.parse_args()
    print('Read-only advisor: it never clicks or types. Private, consenting games only.', file=sys.stderr)

    if args.state:
        print(advise(Position.model_validate_json(args.state.read_text()), args.policy))
        return
    last = None
    while True:
        image = args.image or capture(args.region)
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest != last:
            last = digest
            position = read_position(image)
            if args.save:
                args.save.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime('%Y%m%dT%H%M%S')
                (args.save / f'{stamp}.png').write_bytes(image.read_bytes())
                (args.save / f'{stamp}.json').write_text(position.model_dump_json(indent=2))
            print(time.strftime('%H:%M:%S'), advise(position, args.policy), flush=True)
        if args.image or not args.watch:
            return
        time.sleep(args.watch)


if __name__ == '__main__':
    main()
