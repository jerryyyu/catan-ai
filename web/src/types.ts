export type Action =
  | { type: "roll" | "end_turn" }
  | { type: "build_settlement" | "build_city"; vertex: number }
  | { type: "build_road"; edge: [number, number] }
  | { type: "move_robber"; tile: number }
  | { type: "discard"; resource: string }
  | { type: "steal"; victim: number }
  | { type: "buy_development" }
  | { type: "offer_trade"; recipient: number; give: Record<string, number>; receive: Record<string, number> }
  | { type: "accept_trade" | "reject_trade" }
  | { type: "play_development"; card: string; resource?: string; resources?: string[] }
  | { type: "bank_trade"; give: string; receive: string };

export type Tile = { q: number; r: number; resource: string; number: number | null; vertices: number[] };
export type Player = {
  seat: number; name: string; points: number; resource_count: number;
  resources: Record<string, number> | null; roads: [number, number][];
  settlements: number[]; cities: number[];
  trade_rates: Record<string, number>;
  road_length: number; longest_road: boolean;
  development: string[] | null; development_count: number; knights: number; largest_army: boolean;
};
export type State = {
  phase: string; current: number; turn: number; rolled: number | null; robber_tile: number;
  winner: number | null; last_event: string;
  discard_remaining: number;
  trade_offer: {owner: number; recipient: number; give: Record<string, number>; receive: Record<string, number>} | null;
  board: { vertices: [number, number][]; edges: [number, number][]; tiles: Tile[]; ports: {vertices: [number, number]; resource: string | null}[] };
  players: Player[]; legal_actions: Action[];
};
