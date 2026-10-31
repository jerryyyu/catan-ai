import { useState } from 'react';
import type { Action, State } from './types';

const resources = ['brick', 'lumber', 'wool', 'grain', 'ore'];

export default function Trade({state, busy, act}: {state: State; busy: boolean; act: (a: Action) => void}) {
  const self = state.players.find(p => p.resources !== null)!;
  const [recipient, setRecipient] = useState(() => state.players.find(p => p.seat !== self.seat)!.seat);
  const [give, setGive] = useState<Record<string, number>>({});
  const [receive, setReceive] = useState<Record<string, number>>({});
  const bundle = (values: Record<string, number>) => Object.fromEntries(Object.entries(values).filter(([, n]) => n > 0));
  const valid = Object.values(give).some(n => n > 0) && Object.values(receive).some(n => n > 0) && resources.every(r => !(give[r] && receive[r]));
  return <section>
    {state.trade_offer && <p>{state.players[state.trade_offer.owner].name} offers {Object.entries(state.trade_offer.give).map(([r,n]) => `${n} ${r}`).join(', ')} for {Object.entries(state.trade_offer.receive).map(([r,n]) => `${n} ${r}`).join(', ')}.</p>}
    {state.legal_actions.map((a,i) => a.type === 'accept_trade' || a.type === 'reject_trade' ? <button key={i} disabled={busy} onClick={() => act(a)}>{a.type === 'accept_trade' ? 'Accept trade' : 'Decline trade'}</button> : null)}
    {state.phase === 'actions' && state.current === self.seat && <details><summary>Offer player trade</summary>
      <label>Partner <select value={recipient} onChange={e => setRecipient(Number(e.target.value))}>{state.players.filter(p => p.seat !== self.seat).map(p => <option key={p.seat} value={p.seat}>{p.name}</option>)}</select></label>
      {resources.map(r => <div key={r}><span>{r}</span><label> Give <input aria-label={`Give ${r}`} type="number" min={0} max={self.resources?.[r] ?? 0} value={give[r] ?? 0} onChange={e => setGive({...give, [r]: Math.max(0, Math.min(self.resources?.[r] ?? 0, Math.floor(Number(e.target.value))))})}/></label><label> Receive <input aria-label={`Receive ${r}`} type="number" min={0} max={19} value={receive[r] ?? 0} onChange={e => setReceive({...receive, [r]: Math.max(0, Math.min(19, Math.floor(Number(e.target.value))))})}/></label></div>)}
      <button disabled={busy || !valid} onClick={() => act({type:'offer_trade', recipient, give:bundle(give), receive:bundle(receive)})}>Offer trade</button>
    </details>}
  </section>;
}
