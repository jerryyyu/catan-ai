import {test, expect} from '@playwright/test';

test('new games start from the server default and show its fallback reason', async ({page}) => {
  const info = await (await page.request.get('/api/bots')).json();
  await page.goto('/');
  const select = page.getByLabel('Opponents');
  await expect(select).toHaveValue(info.default);
  expect(info.served).toBe('planner_pv2_tiles_va_t9');
  await expect(page.locator('option[value="planner_pv2_tiles_va_t9"]')).toHaveText('Trained planner, retrained value net (served default)');
  await expect(page.locator('option[value="planner_pv2_tiles"]')).toHaveText('Trained planner with value search (previous default)');
  const tiles = page.locator('option[value="planner_pv2_tiles"]');
  if (info.tiles_available) await expect(tiles).toBeEnabled(); else await expect(tiles).toBeDisabled();
  if (info.fallback_reason) await expect(page.getByText(info.fallback_reason, {exact: false})).toBeVisible();
  else await expect(page.locator('.landing .error')).toHaveCount(0);
  const created = page.waitForResponse(response => response.url().endsWith('/api/games') && response.request().method() === 'POST');
  await page.getByRole('button', {name: 'Start a game'}).click();
  const body = await (await created).json();
  expect(body.opponent).toBe(info.default);
});
