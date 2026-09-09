# Project instructions for Claude Code

## Git workflow — non-negotiable

`main` is protected on GitHub: no direct pushes, no force-pushes, no branch deletion,
linear history, PR required. This applies to admins too. Never try to work around it.

**Never commit or push on `main`.** Every change starts on a branch:

```bash
git switch -c fix/short-description    # feature/ fix/ chore/ docs/ refactor/
```

If uncommitted work is already sitting on `main`, move it to a branch before
committing anything — `git switch -c <branch>` carries the changes along.

**One branch = one concern.** Two unrelated ideas are two branches and two PRs,
even when they touch the same file. Never bundle them.

**Atomic commits.** One logical change per commit, imperative mood
(`Add rolling-window retraining to ARIMA`). Put formatting, lockfile, and generated-file
churn in their own commit so it never hides a real change in the diff.

**Never push unless Bram says to push.** Commit locally, then show
`git log --oneline` and the diff and stop. Push and open a PR only on explicit request:

```bash
git push -u origin <branch>
gh pr create --fill --base main
```

**Bram merges. Claude never merges.** Claude opens the PR; Bram reviews and merges.
Prefer squash-merge.

**Sync with rebase:** `git pull --rebase origin main`.

**Ask first, every time**, before: force-push, `reset --hard`, `rebase -i` or any history
rewrite on a pushed branch, `git clean`, deleting a branch, or changing branch protection.

## Data and outputs

`data/`, `output/`, and `plots/` hold research artifacts and are gitignored. Keep it that
way: never force-add files from them, and never delete or overwrite anything in them
without asking first.
