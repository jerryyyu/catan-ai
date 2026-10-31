// Pure helpers behind the trade form. The server stays authoritative on legality;
// these only shape what the form submits and whether the button is enabled.

export const RESOURCES = ['brick', 'lumber', 'wool', 'grain', 'ore'] as const;

export type Bundle = Record<string, number>;

/** Drop zero and negative counts so the offer carries only real quantities. */
export function bundle(values: Bundle): Bundle {
  return Object.fromEntries(Object.entries(values).filter(([, n]) => n > 0));
}

/** An offer needs something on both sides and may not give and receive one resource. */
export function validOffer(give: Bundle, receive: Bundle): boolean {
  const gives = Object.values(give).some(n => n > 0);
  const receives = Object.values(receive).some(n => n > 0);
  const disjoint = RESOURCES.every(r => !(give[r] && receive[r]));
  return gives && receives && disjoint;
}

/** Clamp a typed count to a whole number within [0, max]. */
export function clampCount(raw: string | number, max: number): number {
  const n = Math.floor(Number(raw));
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(max, n));
}
