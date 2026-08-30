
winget install --id Rustlang.Rust  # runtime-core
cargo --version  # verify runtime-core
cargo install wasm-pack --locked  # wasm
wasm-pack --version  # verify wasm
pip install -r backend/requirements.txt # backend
Push-Location frontend;  try { npm install } finally { Pop-Location }  # frontend


## Building
Push-Location runtime-core; try { maturin develop --release } finally { Pop-Location }  # PowerShell-safe pattern for runtime-core
Push-Location wasm-orbit; try { wasm-pack build --target web } finally { Pop-Location }  # PowerShell-safe pattern for wasm bindings (wasm-orbit)
npm --prefix frontend run build --silent  # frontend
# backend does not need to be built

## Training
Push-Location backend; try { ..\.venv\Scripts\python.exe train.py } finally { Pop-Location }

## Testing
npm test --prefix frontend  # Unit tests (frontend): this runs Vitest **excluding** Playwright E2E tests.
pytest backend # backend
cargo test -q # runtime-core

## e2e Testing
npm --prefix frontend run test:e2e # Playwright E2E (frontend): runs Playwright test runner and browsers (uses `playwright.config.ts`).