import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:8013', channel: 'chrome', headless: true },
  webServer: {
    command: 'cd .. && uv run uvicorn browser_server:app --app-dir tests --host 127.0.0.1 --port 8013',
    url: 'http://127.0.0.1:8013/api/healthz',
    reuseExistingServer: false,
  },
});
