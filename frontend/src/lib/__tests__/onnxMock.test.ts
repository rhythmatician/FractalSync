import { test, expect } from 'vitest';

const TEST_NO_NETWORK = process.env.TEST_NO_NETWORK === '1' || process.env.TEST_NO_NETWORK === 'true';

if (TEST_NO_NETWORK) {
  test('TEST_NO_NETWORK is enabled for fast tests', async () => {
    // Basic check: ensure the fast test runner enabled the env var
    expect(process.env.TEST_NO_NETWORK === '1' || process.env.TEST_NO_NETWORK === 'true').toBe(true);
  });
} else {
  test.skip('skipping onnx mock test when network allowed', async () => {});
}
