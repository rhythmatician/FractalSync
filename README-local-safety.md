Local Safety: Prevent accidental frontend/frontend

This repository provides local safeguards to avoid accidentally creating a nested `frontend/frontend` directory (this has caused repeated accidental mistakes).

Two safe, opt-in tools are provided:

1) Git hook (recommended)
- The hook ` .githooks/prevent-frontend-subdir` detects if the working tree or staged changes include `frontend/frontend` and fails the commit.
- To enable it for this repo run (from repo root):

  git config core.hooksPath .githooks

  or run the helper script:

  powershell -File scripts/setup-local-safety.ps1

2) PowerShell protective wrapper (optional)
- If you use PowerShell, the setup script offers to append a small `mkdir` wrapper to your PowerShell profile that prevents creating a directory whose name equals the parent folder's name.
- This is opt-in and reversible by removing the appended block from your PowerShell profile.

If you prefer a different shell (bash/zsh), you can implement a shell alias or function with similar logic that checks the target leaf name against the current directory's base name.
