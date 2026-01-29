/**
 * This TypeScript file intentionally delegates to the canonical Playwright global
 * setup implementation in `playwright-global-setup.cjs` to avoid maintaining
 * multiple copies of the same logic. The Playwright configuration references the
 * `.cjs` file directly; keep behavior changes in that file.
 */

// eslint-disable-next-line @typescript-eslint/no-var-requires
const globalSetup = require('./playwright-global-setup.cjs');

export default globalSetup;
