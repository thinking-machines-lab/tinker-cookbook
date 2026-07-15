import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: '../',
  testMatch: 'e2e_test.py',
  timeout: 30000,
  use: {
    baseURL: 'http://127.0.0.1:8199',
    headless: true,
  },
});
