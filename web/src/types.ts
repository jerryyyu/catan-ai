export type Action =
  | { type: "roll" | "end_turn" }
  | { type: "build_settlement" | "build_city"; vertex: number }
  | { type: "build_road"; edge: [number, number] }
  | { type: "move_robber"; tile: number }
  | { type: "bank_trade"; give: string; receive: string };

export type Tile = { q: number; r: number; resource: string; number: number | null; vertices: number[] };
export type Player = {
  seat: number; name: string; points: number; resource_count: number;
  resources: Record<string, number> | null; roads: [number, number][];
  settlements: number[]; cities: number[];
};
export type State = {
  phase: string; current: number; turn: number; rolled: number | null; robber_tile: number;
  winner: number | null; last_event: string;
  board: { vertices: [number, number][]; edges: [number, number][]; tiles: Tile[] };
  players: Player[]; legal_actions: Action[];
};
