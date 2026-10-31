import { useEffect, useState } from 'react';
import Board from './Board';
import type { State } from './types';

type Snapshot = { step: number; total_steps: number; state: State; human_seat: number };

export default function Replay({gameId, close}: {gameId: string; close: () => void}) {
  const [step, setStep] = useState(0);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);
  useEffect(() => {
    const controller = new AbortController();
    setBusy(true); setError(null);
    fetch(`/api/games/${encodeURIComponent(gameId)}/replay?step=${step}`, {signal: controller.signal})
      .then(async response => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail ?? 'Could not load replay');
        if (!controller.signal.aborted) setSnapshot(data);
      })
      .catch(e => { if (!controller.signal.aborted) setError(String(e)); })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [gameId, step]);
  return <main className="game-shell" aria-label="Match replay">
    <header><strong>Match replay</strong><button onClick={close}>Back to play</button></header>
    <section>
      <label>Decision <input aria-label="Replay decision" type="range" min={0}
        max={snapshot?.total_steps ?? 0} value={step} disabled={!snapshot}
        onChange={e => setStep(Number(e.target.value))}/></label>
      <button disabled={busy || step === 0} onClick={() => setStep(step - 1)}>Previous decision</button>
      <button disabled={busy || !snapshot || step >= snapshot.total_steps}
        onClick={() => setStep(step + 1)}>Next decision</button>
      <p aria-live="polite">{busy ? 'Loading replay…' : `Decision ${snapshot?.step ?? 0} of ${snapshot?.total_steps ?? 0}`}</p>
      {error && <p className="error">{error}</p>}
    </section>
    {snapshot && <>
      <section className="table"><Board state={snapshot.state} disabled onAction={() => {}}/></section>
      <aside className="panel">
        <p>{snapshot.state.last_event}</p>
        <p>Your recorded hand: {Object.entries(snapshot.state.players[snapshot.human_seat].resources ?? {})
          .map(([resource, count]) => `${count} ${resource}`).join(', ')}</p>
        <p>Replay shows only information available to your seat. Playing resumes from the live position.</p>
      </aside>
    </>}
  </main>;
}
