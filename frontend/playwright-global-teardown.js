// This file is kept as a thin wrapper to avoid duplication with
// `playwright-global-teardown.cjs`. All teardown logic lives in the
// `.cjs` file, which is the one referenced by `playwright.config.ts`.

module.exports = require('./playwright-global-teardown.cjs');
