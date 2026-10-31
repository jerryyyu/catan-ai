import { test, expect } from '@playwright/test';

test('human completes setup and plays through real API phases', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  const started = page.waitForResponse(r => r.url().endsWith('/api/games') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Start a game' }).click();
  let state = (await (await started).json()).state;
  for (let step = 0; step < 45 && state.winner === null; step++) {
    await expect(page.locator('.turn strong')).toHaveText(state.last_event);
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const reply = page.waitForResponse(r => r.url().endsWith('/actions'));
    if (state.phase === 'setup_settlement') {
      await page.locator('svg .vertex.legal').first().click();
    } else if (state.phase === 'setup_road' || state.phase === 'free_roads') {
      const point = await page.locator('svg .edge.legal').first().evaluate(el => {
        const box = el.getBoundingClientRect();
        return {x: box.x + box.width / 2, y: box.y + box.height / 2};
      });
      await page.mouse.click(point.x, point.y);
    } else if (state.phase === 'roll') {
      await page.getByRole('button', { name: 'Roll dice', exact: true }).click();
    } else if (state.phase === 'robber') {
      await page.locator('svg g.clickable').first().click();
    } else if (state.phase === 'steal') {
      await page.getByRole('button', { name: /^Steal from/ }).first().click();
    } else if (state.phase === 'discard') {
      await page.getByRole('button', { name: /^Discard / }).first().click();
    } else if (state.phase === 'trade_response') {
      await page.getByRole('button', { name: 'Decline trade' }).click();
    } else {
      await page.getByRole('button', { name: 'End turn', exact: true }).click();
    }
    const response = await reply;
    expect(response.status()).toBe(200);
    state = await response.json();
    expect(state.current).toBe(0);
  }
  expect(state.turn).toBeGreaterThan(4);
  expect(errors).toEqual([]);
});
