import {test, expect} from '@playwright/test';

test('new games start from the server default and show its fallback reason', async ({page}) => {
  const info = await (await page.request.get('/api/bots')).json();
  await page.goto('/');
  const select = page.getByLabel('Opponents');
  await expect(select).toHaveValue(info.default);
  await expect(page.locator('option[value="planner_pv2_tiles"]')).toHaveText('Trained planner with value search (served default)');
  if (info.fallback_reason) await expect(page.getByText(info.fallback_reason, {exact: false})).toBeVisible();
  else await expect(page.locator('.landing .error')).toHaveCount(0);
  const created = page.waitForResponse(response => response.url().endsWith('/api/games') && response.request().method() === 'POST');
  await page.getByRole('button', {name: 'Start a game'}).click();
  const body = await (await created).json();
  expect(body.opponent).toBe(info.default);
});
