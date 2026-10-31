import {test, expect} from '@playwright/test';

for (const humanSeat of [0, 1, 2, 3]) {
test(`seat ${humanSeat + 1} resumes the same match and exports its private view`, async ({page}) => {
  await page.goto('/');
  await page.getByLabel('Your seat', {exact: true}).selectOption(String(humanSeat));
  await page.getByRole('button', {name: 'Start a game'}).click();
  await expect(page.locator('.panel')).toBeVisible();
  const id = await page.evaluate(() => localStorage.getItem('catan-pilot-game'));
  expect(id).toBeTruthy();
  let before = await (await page.request.get(`/api/games/${id}`)).json();
  for (let step = 0; step < 2; step++) {
    const response = await page.request.post(`/api/games/${id}/actions`, {data: {action: before.legal_actions[0]}});
    expect(response.status()).toBe(200);
    before = await response.json();
  }
  await page.reload();
  await expect(page.locator('.panel')).toBeVisible();
  await expect(page.locator('.turn strong')).toHaveText(before.last_event);
  expect(await page.evaluate(() => localStorage.getItem('catan-pilot-game'))).toBe(id);
  await expect(page.getByRole('link', {name: 'Download replay'})).toHaveAttribute('href', `/api/games/${id}/replay`);
  const replay = await (await page.request.get(`/api/games/${id}/replay`)).json();
  expect(replay.state).toEqual(before);
  expect(replay.seed).toBeUndefined();
  expect(replay.human_seat).toBe(humanSeat);
  for (const player of replay.state.players) {
    if (player.seat === humanSeat) expect(player.resources).not.toBeNull();
    else expect(player.resources).toBeNull();
  }
  await expect(page.locator('.hand')).toBeVisible();
  await page.getByRole('button', {name: 'Review match', exact: true}).click();
  await expect(page.getByLabel('Match replay', {exact: true})).toBeVisible();
  await expect(page.getByRole('button', {name: 'Next decision'})).toBeEnabled();
  await page.getByRole('button', {name: 'Next decision'}).click();
  await expect(page.getByText(`Decision 1 of ${replay.total_steps}`, {exact: true})).toBeVisible();
  await page.getByRole('button', {name: 'Back to play'}).click();
  await expect(page.locator('.turn strong')).toHaveText(before.last_event);
  expect(await (await page.request.get(`/api/games/${id}`)).json()).toEqual(before);
});
}
