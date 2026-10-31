import { test, expect } from '@playwright/test';

test('three private browser seats join, play, resume and review their own views', async ({ page, browser }) => {
  await page.goto('/#room');
  await page.getByRole('button', {name: 'Create private room'}).click();
  const links = [];
  for (const seat of [2, 3, 4]) {
    const input = page.getByLabel(`Seat ${seat} invitation`);
    await expect(input).toBeVisible();
    links.push(await input.inputValue());
  }
  const contexts = await Promise.all(links.map(() => browser.newContext()));
  const players = await Promise.all(contexts.map(context => context.newPage()));
  try {
    for (let i = 0; i < players.length; i++) {
      await players[i].goto(links[i]);
      await players[i].getByLabel('Display name').fill(`Fixture ${i}`);
      await players[i].getByLabel('Completed Catan matches').fill('50');
      await players[i].getByRole('checkbox').check();
      await players[i].getByRole('button', {name: 'Join match'}).click();
      await expect(players[i].locator('.panel')).toBeVisible();
    }
    const first = players[0];
    await expect(first.getByRole('status')).toHaveCount(0);
    const reply = first.waitForResponse(r => r.url().endsWith('/actions'));
    await first.locator('svg .vertex.legal').first().click();
    const response = await reply;
    expect(response.status()).toBe(200);
    const result = await response.json();
    expect(result.seat).toBe(1);
    expect(result.state.players[1].resources).not.toBeNull();
    expect(result.state.players[0].resources).toBeNull();
    await first.reload();
    await expect(first.locator('svg .edge.legal').first()).toBeAttached();
    await first.getByRole('button', {name: 'Review match'}).click();
    const slider = first.getByLabel('Replay step');
    await expect(slider).toBeVisible();
    await slider.fill('0');
    await expect(first.getByText(/^Replay step 0\//)).toBeVisible();
    await first.getByRole('button', {name: 'Back to live match'}).click();
    await expect(first.locator('svg .edge.legal').first()).toBeAttached();
    await expect(players[1].locator('svg .vertex.legal')).toHaveCount(0);
  } finally {
    await Promise.all(contexts.map(context => context.close()));
  }
});
