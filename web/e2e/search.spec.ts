import { test, expect } from '@playwright/test';

test('experimental search opponent is selectable through the real API', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Opponents').selectOption('mc');
  const response = page.waitForResponse(r => r.url().endsWith('/api/games') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Start a game' }).click();
  const created = await response;
  expect(created.status()).toBe(200);
  expect(created.request().postDataJSON().opponent).toBe('mc');
  await expect(page.locator('.turn strong')).toHaveText('Place your first settlement');
  await page.locator('svg .vertex.legal').first().click();
  await expect(page.locator('svg .edge.legal').first()).toBeAttached();
});

test('learned selection reports unavailable-model fallback', async ({ page }) => {
  await page.route('**/api/games', async route => {
    const response = await route.fetch();
    const body = await response.json();
    await route.fulfill({ response, json: { ...body, model_loaded: false } });
  });
  await page.goto('/');
  await page.getByLabel('Opponents').selectOption('learned');
  await page.getByRole('button', { name: 'Start a game' }).click();
  await expect(page.getByText('Learned model unavailable. These opponents use Tactical.')).toBeVisible();
  await page.locator('svg .vertex.legal').first().click();
  await expect(page.locator('svg .edge.legal').first()).toBeAttached();
});
