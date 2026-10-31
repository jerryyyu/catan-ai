import {test, expect, type Page} from '@playwright/test';

async function load(page: Page, scenario: string) {
  await page.addInitScript(() => localStorage.removeItem('catan-pilot-game'));
  const fixture = await page.request.post(`/fixtures/${scenario}`);
  const data = await fixture.json();
  await page.route('**/api/games', route => route.fulfill({json: data}));
  await page.goto('/');
  await page.getByRole('button', {name: 'Start a game'}).click();
  await expect(page.locator('.panel')).toBeVisible();
  return data;
}

async function clickAction(page: Page, label: string) {
  const pending = page.waitForResponse(r => r.url().endsWith('/actions'));
  await page.getByRole('button', {name: label, exact: true}).click();
  const response = await pending;
  expect(response.status()).toBe(200);
  const state = await response.json();
  await expect(page.locator('.turn strong')).toHaveText(state.last_event);
  return state;
}

for (const [card, label] of [['knight','Play knight'], ['monopoly','Play monopoly ore'], ['year_of_plenty','Play year of plenty ore + ore'], ['road_building','Play road building']]) {
  test(`play ${card} through engine`, async ({page}) => {
    await load(page, card);
    await page.getByText('Development card actions', {exact: true}).click();
    const state = await clickAction(page, label);
    expect(state.players[0].development_count).toBe(0);
    if (card === 'knight') expect(state.phase).toBe('robber');
    if (card === 'road_building') expect(state.phase).toBe('free_roads');
    if (card === 'monopoly') expect(state.players[0].resources.ore).toBe(5);
    if (card === 'year_of_plenty') expect(state.players[0].resources.ore).toBe(5);
  });
}

test('purchase development card', async ({page}) => {
  await load(page, 'actions');
  await page.getByText('Development card actions', {exact:true}).click();
  const purchased = await clickAction(page, 'Buy development card');
  expect(purchased.players[0].development_count).toBe(1);
  expect(purchased.players[1].development).toBeNull();
});

test('use resource-specific port', async ({page}) => {
  await load(page, 'port');
  await page.getByText('Maritime trade', {exact:true}).click();
  const state = await clickAction(page, '2 brick → 1 ore');
  expect(state.players[0].resources.brick).toBe(1);
  expect(state.players[0].resources.ore).toBe(4);
});

test('discard and select victim', async ({page}) => {
  await load(page, 'discard');
  expect((await clickAction(page, 'Discard brick')).discard_remaining).toBe(1);
  expect((await clickAction(page, 'Discard brick')).phase).toBe('robber');
  await page.unroute('**/api/games');
  await load(page, 'victim');
  const state = await clickAction(page, 'Steal from Ada');
  expect(state.phase).toBe('actions');
  expect(state.players[1].resource_count).toBe(9);
});

test('offer a resource bundle to a bot', async ({page}) => {
  await load(page, 'actions');
  await page.getByText('Offer player trade', {exact:true}).click();
  await page.getByLabel('Give brick', {exact:true}).fill('2');
  await page.getByLabel('Receive ore', {exact:true}).fill('1');
  const state = await clickAction(page, 'Offer trade');
  expect(state.players[0].resources.brick).toBe(1);
  expect(state.players[0].resources.ore).toBe(4);
  expect(state.trade_offer).toBeNull();
});

test('resolve both free roads', async ({page}) => {
  await load(page, 'road_building');
  await page.getByText('Development card actions', {exact:true}).click();
  await clickAction(page, 'Play road building');
  for (let i = 0; i < 2; i++) {
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const edge = page.locator('svg .edge.legal').first();
    const point = await edge.evaluate(el => {const b = el.getBoundingClientRect(); return {x:b.x+b.width/2,y:b.y+b.height/2};});
    const pending = page.waitForResponse(r => r.url().endsWith('/actions'));
    await page.mouse.click(point.x, point.y);
    const response = await pending;
    expect(response.status()).toBe(200);
    const state = await response.json();
    expect(state.players[0].roads.length).toBe(i+1);
    if (i === 1) expect(state.phase).toBe('actions');
  }
});

test('accept an incoming trade and return to play', async ({page}) => {
  await load(page, 'trade_response');
  const state = await clickAction(page, 'Accept trade');
  expect(state.current).toBe(0);
  expect(state.players[1].resources).toBeNull();
});

test('city upgrade reaches victory and disables further actions', async ({page}) => {
  await load(page, 'win');
  const pending = page.waitForResponse(r => r.url().endsWith('/actions'));
  await page.locator('svg g.clickable rect').first().click();
  const response = await pending;
  expect(response.status()).toBe(200);
  const state = await response.json();
  expect(state.winner).toBe(0);
  expect(state.legal_actions).toEqual([]);
  await expect(page.locator('.winner')).toHaveText('You wins');
});
