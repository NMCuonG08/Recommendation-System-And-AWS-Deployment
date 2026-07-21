# Git Hooks (secret guard)

Shared git hooks that block secrets from entering the repo or being pushed.

## Hooks

- `pre-commit` — blocks `.env` files and staged files containing real-looking secrets (AWS keys, RDS/Kaggle creds, private keys, GitHub/Slack tokens).
- `pre-push` — re-scans the pushed commit range as a last line of defense.

Both are self-contained bash using only `git` + `grep` (no pre-commit CLI, no ripgrep dependency) so they work in non-interactive shells.

## Enable (once per clone)

```bash
git config core.hooksPath .githooks
```

Already set in this repo's local config. New clones need the command above.

## Allowlist

Local-only / placeholder credentials are allowed so dev defaults don't trip the guard:

- MinIO local: `admin`, `Password1234`
- Local MLflow: `postgres:postgres@localhost`
- Placeholders: `CHANGE_ME`, `REPLACE_ME`, `your-`, `example`, `changeme`, `XXXX`
- `${ENV_VAR}` references

Real secrets belong in `.env` (gitignored). Tracked files use placeholders only.

## Bypass (not recommended)

```bash
git commit --no-verify
git push --no-verify
```

Use only for verified-false positives. Investigate and tighten the pattern after.