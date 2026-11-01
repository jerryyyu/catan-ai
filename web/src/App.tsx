import { useEffect, useState } from "react";
import type { Action, State } from "./types";
import Board from "./Board";
import Trade from "./Trade";
import Replay from "./Replay";
import Room from './Room';

const resourceNames = ["brick", "lumber", "wool", "grain", "ore"];

export default function App() {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const changed = () => setHash(location.hash);
    window.addEventListener('hashchange', changed);
    return () => window.removeEventListener('hashchange', changed);
  }, []);
  return hash.startsWith('#room') ? <Room key={hash}/> : <><a href="#room">Private three-human match</a><Pilot/></>;
}

function Pilot() {
  const [gameId, setGameId] = useState<string | null>(null);
  const [state, setState] = useState<State | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [opponent, setOpponent] = useState('planner_unified');
  const [servedAvailable, setServedAvailable] = useState<boolean | null>(null);
  const [humanSeat, setHumanSeat] = useState(0);
  const [showReplay, setShowReplay] = useState(false);

  useEffect(() => {
    let active = true;
    fetch('/api/bots').then(response => response.ok ? response.json() : null).then(data => {
      if (!active) return;
      const available = Boolean(data?.served_available);
      setServedAvailable(available);
      if (!available) setOpponent(current => current === 'planner_unified' ? (data?.default ?? 'tactical') : current);
    }).catch(() => { if (active) { setServedAvailable(false); setOpponent(current => current === 'planner_unified' ? 'tactical' : current); } });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem('catan-pilot-game');
    if (!saved) return;
    let active = true;
    setBusy(true);
    fetch(`/api/games/${encodeURIComponent(saved)}`).then(async response => {
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? 'Could not resume saved game');
      if (active) { setGameId(saved); setState(data); }
    }).catch(e => { if (active) setError(e instanceof Error ? e.message : String(e)); })
      .finally(() => { if (active) setBusy(false); });
    return () => { active = false; };
  }, []);

  async function newGame() {
    setBusy(true); setError(null);
    try {
      const response = await fetch("/api/games", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: "You", opponent, human_seat: humanSeat }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Could not start game");
      setGameId(data.game_id); setState(data.state);
      localStorage.setItem('catan-pilot-game', data.game_id);
      if (data.opponent === 'learned' && data.model_loaded === false) {
        setError('Learned model unavailable. These opponents use Tactical.');
      }
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  async function act(action: Action) {
    if (!gameId) return;
    setBusy(true); setError(null);
    try {
      const response = await fetch(`/api/games/${gameId}/actions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Action failed");
      setState(data);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  if (!state) return <main className="landing"><div className="eyebrow">A reproducible game lab</div><h1>Catan<br/><em>AI</em></h1><p>Play the bots. Measure the next idea. Keep the engine honest.</p><label>Opponents <select aria-label="Opponents" value={opponent} onChange={e => setOpponent(e.target.value)} disabled={busy}><option value="heuristic">Heuristic</option><option value="mc">Monte Carlo LCB (experimental)</option><option value="planner_unified" disabled={servedAvailable === false}>Trained planner (default)</option><option value="planner_devfix" disabled={servedAvailable === false}>Trained planner, build search only</option><option value="planner_parameter" disabled={servedAvailable === false}>Trained planner, before the card fix</option><option value="tactical">Tactical (previous default)</option><option value="mc_mean">Monte Carlo mean (experimental, slower)</option><option value="learned">Learned baseline (experimental)</option></select></label><label>Your seat <select aria-label="Your seat" value={humanSeat} onChange={e => setHumanSeat(Number(e.target.value))} disabled={busy}>{[0, 1, 2, 3].map(seat => <option key={seat} value={seat}>Seat {seat + 1}</option>)}</select></label><button onClick={newGame} disabled={busy || servedAvailable === null}>Start a game</button>{servedAvailable === false && <p className="error">Trained planner unavailable on this server version. Using Tactical.</p>}{error && <p className="error">{error}</p>}</main>;

  if (showReplay && gameId) return <Replay gameId={gameId} close={() => setShowReplay(false)}/>;
  const self = state.players.find(p => p.resources !== null)!;
  const direct = state.legal_actions.filter(a => a.type === "roll" || a.type === "end_turn");
  const trades = state.legal_actions.filter((a): a is Extract<Action, {type: "bank_trade"}> => a.type === "bank_trade");
  return <main className="game-shell">
    <header><div><span className="mark">CA</span><strong>Catan AI Lab</strong></div><button disabled={busy} onClick={() => setShowReplay(true)}>Review match</button><a href={`/api/games/${gameId}/replay`} download={`catan-pilot-${gameId}.json`}>Download replay</a><button className="quiet" disabled={busy} onClick={newGame}>New game</button></header>
    <section className="players">{state.players.map(p => <article key={p.seat} className={state.current === p.seat ? "active" : ""}><div className={`avatar p${p.seat}`}>{p.name[0]}</div><div><strong>{p.name}</strong><small>{p.points} points · {p.resource_count} cards</small></div></article>)}</section>
    <section className="table"><Board state={state} onAction={act} disabled={busy}/></section>
    <aside className="panel"><div className="turn"><span>Turn {state.turn + 1}</span><strong>{state.last_event}</strong></div>
      <Trade key={gameId} state={state} busy={busy} act={act}/>
      <div aria-label="Longest Road">{state.players.filter(p => p.longest_road).map(p => <p key={p.seat}>{p.name}: Longest Road ({p.road_length}) · 2 points</p>)}</div>
      <div aria-label="Development cards"><p>Your development cards: {self.development?.map(c => c.replaceAll('_', ' ')).join(', ') || 'None'}</p>{state.players.map(p => <p key={p.seat}>{p.name}: {p.development_count} cards, {p.knights} knights{p.largest_army ? ' · Largest Army (2 points)' : ''}</p>)}</div>
      <details className="trades"><summary>Development card actions</summary>{state.legal_actions.map((a, i) => a.type === "buy_development" ? <button key={i} disabled={busy} onClick={() => act(a)}>Buy development card</button> : a.type === "play_development" ? <button key={i} disabled={busy} onClick={() => act(a)}>Play {a.card.replaceAll('_', ' ')} {a.resource ?? a.resources?.join(' + ')}</button> : null)}</details>
      {state.rolled && <div className="dice">{state.rolled}</div>}
      {self.resources && <div className="hand">{resourceNames.map(r => <div key={r}><i className={r}/><span>{r}</span><strong>{self.resources![r]}</strong></div>)}</div>}
      <div className="actions">{direct.map((a, i) => <button key={i} onClick={() => act(a)} disabled={busy}>{a.type === "roll" ? "Roll dice" : "End turn"}</button>)}</div>
      {state.phase === "discard" && <p>Discard {state.discard_remaining} more cards.</p>}
      <div className="actions">{state.legal_actions.map((a, i) => a.type === "discard" ? <button key={i} disabled={busy} onClick={() => act(a)}>Discard {a.resource}</button> : a.type === "steal" ? <button key={i} disabled={busy} onClick={() => act(a)}>Steal from {state.players[a.victim].name}</button> : null)}</div>
      {trades.length > 0 && <details className="trades"><summary>Maritime trade</summary>{trades.map((a,i)=><button key={i} onClick={()=>act(a)} disabled={busy}>{self.trade_rates[a.give]} {a.give} → 1 {a.receive}</button>)}</details>}
      <p className="hint">{state.phase === "setup_settlement" ? "Choose a glowing intersection." : state.phase === "setup_road" ? "Choose a glowing road." : state.phase === "robber" ? "Move the robber to a highlighted tile." : "Affordable builds glow on the board."}</p>
      {state.winner !== null && <div className="winner">{state.players[state.winner].name} wins</div>}{error && <p className="error">{error}</p>}
    </aside>
  </main>;
}
