import type { Action, State } from "./types";

const colors: Record<string, string> = { brick: "#b95f46", lumber: "#426b45", wool: "#91ad68", grain: "#d9ad4b", ore: "#78858a", desert: "#d8bc7e" };
const playerColors = ["#f4e8c7", "#ef7258", "#4ca8a8", "#d9a441"];

export default function Board({ state, onAction, disabled }: { state: State; onAction: (a: Action) => void; disabled: boolean }) {
  const xs = state.board.vertices.map(v => v[0]), ys = state.board.vertices.map(v => v[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const point = (v: [number, number]) => [70 + (v[0]-minX)/(maxX-minX)*660, 55 + (v[1]-minY)/(maxY-minY)*570] as const;
  const vertexActions = new Map<number, Action>(); const edgeActions = new Map<string, Action>(); const tileActions = new Map<number, Action>();
  state.legal_actions.forEach(a => { if ("vertex" in a) vertexActions.set(a.vertex, a); if ("edge" in a) edgeActions.set([...a.edge].sort((x,y)=>x-y).join("-"), a); if ("tile" in a) tileActions.set(a.tile, a); });
  return <svg viewBox="0 0 800 680" aria-label="Catan board">
    {state.board.tiles.map((tile, i) => { const pts = tile.vertices.map(v => point(state.board.vertices[v]).join(",")).join(" "); const center = tile.vertices.map(v => point(state.board.vertices[v])).reduce((a,p)=>[a[0]+p[0]/6,a[1]+p[1]/6],[0,0]); const action=tileActions.get(i); return <g key={i} className={action ? "clickable" : ""} onClick={()=>!disabled&&action&&onAction(action)}><polygon points={pts} fill={colors[tile.resource]}/><circle cx={center[0]} cy={center[1]} r="23" className="token"/><text x={center[0]} y={center[1]+6} textAnchor="middle" className={(tile.number===6||tile.number===8)?"hot":""}>{tile.number ?? "·"}</text>{state.robber_tile===i&&<text x={center[0]+25} y={center[1]-23} className="robber">●</text>}</g>; })}
    {state.board.edges.map(edge => { const [a,b]=edge.map(v=>point(state.board.vertices[v])); const key=[...edge].sort((x,y)=>x-y).join("-"); const owner=state.players.findIndex(p=>p.roads.some(e=>[...e].sort((x,y)=>x-y).join("-")===key)); const action=edgeActions.get(key); return <line key={key} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className={action?"edge legal":"edge"} stroke={owner>=0?playerColors[owner]:undefined} onClick={()=>!disabled&&action&&onAction(action)}/>; })}
    {state.board.vertices.map((vertex, i) => { const [x,y]=point(vertex); const owner=state.players.findIndex(p=>p.settlements.includes(i)||p.cities.includes(i)); const city=owner>=0&&state.players[owner].cities.includes(i); const action=vertexActions.get(i); return <g key={i} className={action?"clickable":""} onClick={()=>!disabled&&action&&onAction(action)}>{owner>=0?<rect x={x-(city?9:7)} y={y-(city?9:7)} width={city?18:14} height={city?18:14} rx="3" fill={playerColors[owner]} className="building"/>:<circle cx={x} cy={y} r={action?8:2.5} className={action?"vertex legal":"vertex"}/>}</g>; })}
  </svg>;
}
