---
name: gh-work-on-issue
description: Pick an issue from the "Structure From Harness" GitHub project, mark it In progress, sync main and branch off it, propose a plan, then open a PR rebased on the latest main. Use when the user says /gh-work-on-issue, "what should I work on", or names an issue number to start.
argument-hint: "[issue-number]"
allowed-tools: Bash(gh project:*), Bash(gh issue:*), Bash(gh pr:*), Bash(gh api:*), Bash(git:*), Read
---

# Work on a GitHub project issue

Project: `HantigC` user project **#1** ("Structure From Harness"), repo `HantigC/sfh`.
Status options: `TODO`, `In progress`, `In review`, `Done`. Priority: `P0`–`P2`. Size: `XS`–`XL`.

Argument: `$ARGUMENTS` (an issue number, optional).

## 1. Choose the issue

If an issue number was given, skip to step 2.

Otherwise list the open candidates (anything not `Done`):

```bash
gh project item-list 1 --owner HantigC --format json --limit 200 \
  --jq '.items[] | select(.status != "Done" and .content.type == "Issue")
        | [.content.number, .status, (.priority // "-"), (.size // "-"), .title] | @tsv'
```

Show them as a short table sorted by: `In progress` first, then priority (P0 → P2, unset last),
then size (smaller first). Mark one as your recommendation with a one-line reason (e.g. already in
progress, highest priority, unblocks others). Ask the user in plain text which number to take —
no multiple-choice popup. Wait for the answer.

## 2. Load the issue

```bash
gh issue view <N> --repo HantigC/sfh --json number,title,body,labels,assignees,comments,state
```

If the issue is closed or not in the project, say so and stop.

## 3. Mark it In progress

Only if its status isn't already `In progress`:

```bash
PROJECT_ID=$(gh project view 1 --owner HantigC --format json --jq .id)
ITEM_ID=$(gh project item-list 1 --owner HantigC --format json --limit 200 \
  --jq ".items[] | select(.content.number == <N>) | .id")
read FIELD_ID OPTION_ID < <(gh project field-list 1 --owner HantigC --format json \
  --jq '.fields[] | select(.name == "Status") | [.id, (.options[] | select(.name == "In progress") | .id)] | @tsv')
gh project item-edit --project-id "$PROJECT_ID" --id "$ITEM_ID" \
  --field-id "$FIELD_ID" --single-select-option-id "$OPTION_ID"
```

If it has no assignee, assign it: `gh issue edit <N> --repo HantigC/sfh --add-assignee @me`.

## 4. Sync main and branch off it

Check `git status --short` first. If tracked files have uncommitted changes, show them and ask the
user what to do before switching branches.

If a branch for this issue already exists (`git branch --list '<N>-*'`), switch to it and skip the
rest of this step. Otherwise:

```bash
git switch main
git pull --ff-only origin main
git switch -c <N>-<short-kebab-slug-of-title>
```

If the fast-forward pull fails (local `main` has diverged from `origin/main`), stop and tell the
user. Never reset or force anything on `main`.

From here on, **all work for the issue happens on this branch**. Never commit to `main`.

## 5. Get oriented

- Read `docs/architecture.md` and any modules the issue touches.
- Check `TODO.md` for a matching entry.

## 6. Report back

Give the user:
- The issue title, number and link, and the branch you created.
- What it asks for, in two or three sentences. If the body is empty or vague, say so and ask the
  open questions you need answered before starting.
- A short proposed plan (files/layers to touch, tests to add) that respects the hexagonal rules in
  `CLAUDE.md`.

Don't start writing code until the user agrees with the plan.

## 7. Work on it

Implement the agreed plan on the issue branch. Before every commit, confirm
`git branch --show-current` is the issue branch; if it is `main`, switch back to the issue branch
(`git switch <N>-<slug>`, carrying the uncommitted changes) before committing. Commit as you go
with the repo's message style
(`feat: ...`, `docs: ...`, `setup: ...`). Before opening the PR, run `uv run pytest`,
`uv run mypy`, `uv run lint-imports` and `uv run ruff check`.

## 8. Open the PR

When the user asks for the PR, first bring the branch up to date with the remote `main`:

```bash
git fetch origin main
git rev-list --count HEAD..origin/main   # > 0 means main moved since you branched
```

If `main` moved:
1. `git rebase origin/main`. On conflicts, resolve them in a way that keeps both sides' intent,
   `git add` the files and `git rebase --continue`. If a conflict is ambiguous, stop and ask the
   user rather than guessing.
2. Run the checks from step 7 again and fix anything the new `main` broke.
3. Also fast-forward local `main`: `git fetch origin main:main`.

Then push and open the PR, following `.github/pull_request_template.md` if it exists:

```bash
git push --force-with-lease -u origin HEAD   # force only needed after a rebase
gh pr create --repo HantigC/sfh --base main --title "<type>: <summary>" --body "<filled template, with Closes #<N>>"
```

Finally move the issue to `In review` (same commands as step 3, with option `In review`) and give
the user the PR link.
