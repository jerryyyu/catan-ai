import { useEffect, useState } from 'react';
import Board from './Board';
import Trade from './Trade';
import type { Action, State } from './types';

type RoomState = {seat: number; revision: number; ready: boolean; joined_seats: number[]; bot_seat: number; state: State};

export default function Room() {
  const params = new URLSearchParams(location.hash.slice(1));
  const id = params.get('room') ?? '';
  const token = params.get('token') ?? '';
  const [room, setRoom] = useState<RoomState | null>(null);
  const [invitations, setInvitations] = useState<Record<string, string>>({});
  const [botSeat, setBotSeat] = useState(0);
  const [bot, setBot] = useState('tactical');
  const [name, setName] = useState('');
  const [matches, setMatches] = useState(0);
  const [platform, setPlatform] = useState('');
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [replay, setReplay] = useState<{state: State; step: number; total_steps: number} | null>(null);
  const headers = {'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json'};

  async function request(path: string, body?: unknown) {
    const response = await fetch(path, {method: body === undefined ? 'GET' : 'POST', headers,
      body: body === undefined ? undefined : JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed');
    return data;
  }

  function update(next: RoomState) {
    setRoom(previous => previous && previous.revision > next.revision ? previous : next);
  }

  useEffect(() => {
    if (!id || !token) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try { const data = await request(`/api/rooms/${encodeURIComponent(id)}`); if (active) update(data); }
      catch (e) { if (active) setError(String(e)); }
      finally { if (active) timer = setTimeout(poll, 1000); }
    }
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [id, token]);

  async function perform(task: () => Promise<void>) {
    setBusy(true); setError('');
    try { await task(); } catch (e) { setError(String(e)); }
    finally { setBusy(false); }
  }

  async function create() {
    await perform(async () => {
      const data = await request('/api/rooms', {bot_seat: botSeat, bot});
      setInvitations(Object.fromEntries(Object.entries(data.invitations).map(([seat, value]) =>
        [seat, `${location.origin}/#room=${encodeURIComponent(data.room_id)}&token=${encodeURIComponent(String(value))}`])));
    });
  }

  async function act(action: Action) {
    if (!room || replay) return;
    await perform(async () => update(await request(`/api/rooms/${id}/actions`, {revision: room.revision, action})));
  }

  if (!id) return <main className="landing"><h1>Private match</h1><p>Three people and one AI bot. Share one seat invitation with each player, including yourself.</p>
    <label>Bot <select aria-label="Room bot" value={bot} onChange={e => setBot(e.target.value)}><option value="tactical">Tactical (default)</option><option value="planner_parameter">Trained challenger (pilot)</option><option value="heuristic">Heuristic</option></select></label>
    <label>Bot seat <select aria-label="Bot seat" value={botSeat} onChange={e => setBotSeat(Number(e.target.value))}>{[0, 1, 2, 3].map(s => <option key={s} value={s}>Seat {s + 1}</option>)}</select></label>
    <button disabled={busy} onClick={create}>Create private room</button>
    {Object.entries(invitations).map(([seat, link]) => <p key={seat}><a href={link}>Join seat {Number(seat) + 1}</a><input aria-label={`Seat ${Number(seat) + 1} invitation`} readOnly value={link} onFocus={e => e.target.select()}/></p>)}
    <p>Invitations grant access to that seat. Share them privately.</p><a href="#">Back to solo play</a>{error && <p role="alert">{error}</p>}</main>;

  if (!room) return <main className="landing"><p>Opening room…</p>{error && <p role="alert">{error}</p>}</main>;
  if (!room.joined_seats.includes(room.seat)) return <main className="landing"><h1>Join seat {room.seat + 1}</h1>
    <p>This private research pilot contains one AI bot. It records game actions and your experience answers for evaluation.</p>
    <label>Display name <input aria-label="Display name" maxLength={40} value={name} onChange={e => setName(e.target.value)}/></label>
    <label>Completed Catan matches <input aria-label="Completed Catan matches" type="number" min={0} value={matches} onChange={e => setMatches(Math.max(0, Math.floor(Number(e.target.value))))}/></label>
    <label>Usual platform (optional) <input value={platform} maxLength={80} onChange={e => setPlatform(e.target.value)}/></label>
    <label><input type="checkbox" checked={consent} onChange={e => setConsent(e.target.checked)}/> I agree to participate and have this match recorded.</label>
    <button disabled={busy || !consent || !name.trim()} onClick={() => perform(async () => update(await request(`/api/rooms/${id}/join`, {name: name.trim(), completed_matches: matches, platform, consent})))}>Join match</button>
    {error && <p role="alert">{error}</p>}</main>;

  const state = replay?.state ?? room.state;
  const self = state.players[room.seat];
  const disabled = busy || !room.ready || replay !== null;
  const buttons = state.legal_actions.filter(a => !['build_road', 'build_settlement', 'build_city', 'move_robber', 'accept_trade', 'reject_trade', 'offer_trade'].includes(a.type));
  function label(action: Action) {
    if (action.type === 'bank_trade') return `${self.trade_rates[action.give]} ${action.give} → 1 ${action.receive}`;
    if (action.type === 'play_development') return `Play ${action.card.replaceAll('_', ' ')} ${action.resource ?? action.resources?.join(' + ') ?? ''}`;
    if (action.type === 'discard') return `Discard ${action.resource}`;
    if (action.type === 'steal') return `Steal from ${state.players[action.victim].name}`;
    return action.type.replaceAll('_', ' ');
  }
  return <main className="game-shell"><header><strong>Private match · Seat {room.seat + 1}</strong>
    <button disabled={busy} onClick={() => perform(async () => setReplay(await request(`/api/rooms/${id}/replay`)))}>Review match</button>
    {replay && <button onClick={() => setReplay(null)}>Back to live match</button>}</header>
    {!room.ready && <p role="status">Waiting for players: {room.joined_seats.length}/3 joined.</p>}
    {replay && <label>Replay step {replay.step}/{replay.total_steps}<input aria-label="Replay step" type="range" min={0} max={replay.total_steps} value={replay.step} disabled={busy} onChange={e => { const step = Number(e.target.value); void perform(async () => setReplay(await request(`/api/rooms/${id}/replay?step=${step}`))); }}/></label>}
    <section className="players">{state.players.map(p => <article key={p.seat} className={state.current === p.seat ? 'active' : ''}><strong>{p.name}{p.seat === room.bot_seat ? ' (AI)' : ''}</strong><small>{p.points} points · {p.resource_count} cards</small></article>)}</section>
    <section className="table"><Board state={state} disabled={disabled} onAction={act}/></section>
    <aside className="panel"><p>Turn {state.turn + 1}: {state.last_event}</p>
      <p>{Object.entries(self.resources ?? {}).map(([r, n]) => `${r}: ${n}`).join(' · ')}</p>
      <p>Your development cards: {self.development?.join(', ') || 'None'}</p>
      <Trade state={state} busy={disabled} act={act}/>
      <div className="actions">{buttons.map((a, i) => <button key={i} disabled={disabled} onClick={() => act(a)}>{label(a)}</button>)}</div>
      {state.winner !== null && <p>{state.players[state.winner].name} wins</p>}{error && <p role="alert">{error}</p>}
    </aside></main>;
}
