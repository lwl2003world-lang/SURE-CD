# Publishing SURE-CD to GitHub

## Current state

At the initial documentation review, this folder had no Git history. Git was installed; GitHub CLI (`gh`) was not found on the command path. The source repository is being prepared for `https://github.com/lwl2003world-lang/SURE-CD`. The instructions below also describe setup for another checkout.

## Information to provide

1. Existing repository URL, or the desired account/organization and repository name.
2. Private or public visibility. A private initial repository allows release details to be completed first.
3. Name and email to attach to commits. A GitHub no-reply email is suitable if you prefer not to publish a personal address.
4. Any double-blind anonymity requirements and finalized paper metadata when available.
5. Approved task-weight hosting location and original backbone/dataset provenance needed by the reproduction guide.

Do not send passwords, access tokens, SSH private keys, or recovery codes in chat or project files.

## What you do locally

1. Sign in to GitHub in your browser.
2. Create an **empty** repository with the desired name/visibility. Do not pre-create a README, license or `.gitignore`, because these already exist locally. Share the HTTPS repository URL with the assistant.
3. Install [GitHub CLI](https://cli.github.com/) if using the recommended command-line workflow. Open a new PowerShell window after installing.
4. Run the browser-based local authentication flow:

```powershell
gh auth login --hostname github.com --git-protocol https --web
gh auth status
```

Complete the browser authorization yourself. Tell the assistant only whether login succeeded, without sharing credentials. See [official GitHub CLI authentication instructions](https://cli.github.com/manual/gh_auth_login). An already working Git HTTPS/SSH login also suffices; GitHub CLI is convenient rather than mandatory.

Once the URL, visibility, commit identity, and authentication are ready, ask the assistant to prepare the selected source files and push. The [GitHub local-code publication guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github) explains this workflow.

## What the assistant prepares before committing

Use an explicit source list: `README.md`, `LICENSE`, `.gitignore`, `requirements.txt`, `docs/`, `configs/`, `data/`, `models/`, `train.py`, and `test_eval.py`. Inspect the staged content before committing. Do not blindly use `git add .` in this working folder.

This folder also contains manuscripts, internal assets, scratch directories and editor files, including `XIDAN_write/`, `paper_data/`, `tmp/`, and `.vscode/`. Decide separately which assets belong in a public supplement. The existing `.gitignore` does not guarantee every internal file is excluded.

Existing ignore rules exclude backbone weights, checkpoints, datasets, generated results, and `temp/`. Keep those rules for the source repository. Publish approved task weights separately with stable links/checksums instead of adding large binaries as ordinary source files.

Before public release, inspect selected content for secrets/internal information and anonymity issues, complete the unresolved reproduction evidence, and add finalized citation metadata. Preserve upstream copyright notices, including the Apache-2.0 header in the adapted UPerNet module; check accompanying license/notice files before distribution. Backbone, dataset, and borrowed-asset terms are separate from the root MIT code license.
