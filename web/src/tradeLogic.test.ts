import { describe, expect, it } from 'vitest';
import { bundle, clampCount, validOffer } from './tradeLogic';

describe('bundle', () => {
  it('keeps only positive counts', () => {
    expect(bundle({ brick: 2, ore: 0, wool: -1 })).toEqual({ brick: 2 });
  });
});

describe('validOffer', () => {
  it('requires something on both sides', () => {
    expect(validOffer({ brick: 1 }, {})).toBe(false);
    expect(validOffer({}, { ore: 1 })).toBe(false);
    expect(validOffer({ brick: 1 }, { ore: 1 })).toBe(true);
  });
  it('rejects giving and receiving the same resource', () => {
    expect(validOffer({ grain: 1 }, { grain: 2 })).toBe(false);
  });
});

describe('clampCount', () => {
  it('floors, clamps and survives bad input', () => {
    expect(clampCount('2.9', 5)).toBe(2);
    expect(clampCount(9, 5)).toBe(5);
    expect(clampCount(-3, 5)).toBe(0);
    expect(clampCount('x', 5)).toBe(0);
  });
});
