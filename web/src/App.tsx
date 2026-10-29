import { useState } from "react";
import type { Action, State } from "./types";
import Board from "./Board";

const resourceNames = ["brick", "lumber", "wool", "grain", "ore"];

export default function App() {
  const [gameId, setGameId] = useState<string | null>(null);
  const [state, setState] = useState<State | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function newGame() {
    setBusy(true); setError(null);
    try {
      const response = await fetch("/api/games", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: "You" }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Could not start game");
      setGameId(data.game_id); setState(data.state);
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

  if (!state) return <main className="landing"><div className="eyebrow">A reproducible game lab</div><h1>Catan<br/><em>AI</em></h1><p>Play the baseline. Measure the next idea. Keep the engine honest.</p><button onClick={newGame} disabled={busy}>Start a game</button>{error && <p className="error">{error}</p>}</main>;

  const direct = state.legal_actions.filter(a => a.type === "roll" || a.type === "end_turn");
  const trades = state.legal_actions.filter((a): a is Extract<Action, {type: "bank_trade"}> => a.type === "bank_trade");
  return <main className="game-shell">
    <header><div><span className="mark">CA</span><strong>Catan AI Lab</strong></div><button className="quiet" onClick={newGame}>New game</button></header>
    <section className="players">{state.players.map(p => <article key={p.seat} className={state.current === p.seat ? "active" : ""}><div className={`avatar p${p.seat}`}>{p.name[0]}</div><div><strong>{p.name}</strong><small>{p.points} points · {p.resource_count} cards</small></div></article>)}</section>
    <section className="table"><Board state={state} onAction={act} disabled={busy}/></section>
    <aside className="panel"><div className="turn"><span>Turn {state.turn + 1}</span><strong>{state.last_event}</strong></div>
      {state.rolled && <div className="dice">{state.rolled}</div>}
      {state.players[0].resources && <div className="hand">{resourceNames.map(r => <div key={r}><i className={r}/><span>{r}</span><strong>{state.players[0].resources![r]}</strong></div>)}</div>}
      <div className="actions">{direct.map((a, i) => <button key={i} onClick={() => act(a)} disabled={busy}>{a.type === "roll" ? "Roll dice" : "End turn"}</button>)}</div>
      {trades.length > 0 && <details className="trades"><summary>Trade 4:1 with bank</summary>{trades.map((a,i)=><button key={i} onClick={()=>act(a)} disabled={busy}>{a.give} → {a.receive}</button>)}</details>}
      <p className="hint">{state.phase === "setup_settlement" ? "Choose a glowing intersection." : state.phase === "setup_road" ? "Choose a glowing road." : state.phase === "robber" ? "Move the robber to a highlighted tile." : "Affordable builds glow on the board."}</p>
      {state.winner !== null && <div className="winner">{state.players[state.winner].name} wins</div>}{error && <p className="error">{error}</p>}
    </aside>
  </main>;
}
