# Uploading this to GitHub

This folder **is** the repository root. Its contents go directly into
`Modeling-Recurrent-Events-Using-Recurrent-Neural-Networks-in-AGTs/`, not into
a subfolder.

## Two hidden files that drag-and-drop will miss

| File | Why it matters |
|---|---|
| `.gitignore` | without it, generated CSVs, `results/`, `__pycache__` and figures all get committed |
| `results/.gitkeep` | git does not track empty directories; without this, `results/` vanishes |

macOS Finder and the GitHub web uploader both skip dotfiles by default. Either
press `Cmd+Shift+.` in Finder to reveal them before dragging, or use the
command line (below), which handles them automatically.

## Recommended: command line

Preserves file history through the renames, which the web uploader cannot do.

```bash
cd /path/to/your/local/clone
git switch -c restructure

# 1. rename the existing files so git records them as renames, not deletes
bash /path/to/this/folder/migrate.sh

# 2. copy the new contents over the renamed tree
cp -R /path/to/this/folder/. .
rm -f UPLOAD_NOTES.md migrate.sh        # not part of the repository

# 3. verify before committing
python experiments/run_diagnostics.py    # five checks, all should PASS
git status
git log --follow -- simulation/model_demo.ipynb   # history survived the rename

# 4. commit and push
git add -A
git commit -m "Restructure into a package; correct four implementation defects"
git push -u origin restructure
```

Then open a pull request, or merge to `main` directly if you prefer.

## If you use the web uploader instead

Renames will show as delete-plus-add, so `git log --follow` will no longer trace
a file past this commit. That is a real loss but not a fatal one.

1. Delete the old `1.Literature Review/`, `2.Simulation/`, `3.Application/`
   directories through the web UI first, so the old notebooks do not linger
   alongside the new ones.
2. Upload the contents of this folder.
3. Add `.gitignore` separately via **Add file → Create new file**, pasting its
   contents, since the uploader will not pick it up.
4. Same for `results/.gitkeep` (create the file at path `results/.gitkeep`,
   leave it empty).

## After pushing

1. Check the rendered README on the repository landing page: the table-to-script
   mapping and the link to `CORRECTIONS.md` are what a reviewer following the
   citation in Section 5.3 will read first.
2. Create a release (`v2.0.0`), then connect the repository to Zenodo to mint
   the archived DOI. That fills the last remaining placeholder in Section 5.3.
   A bare GitHub link can be force-pushed or renamed; reviewers increasingly ask
   for the immutable version.
3. Move your existing literature-review PDFs into `literature_review/`. The
   folder currently holds only a README.

## What is deliberately not here

- **Generated data.** `data/crc.csv` and `data/cgd.csv` come from the R
  preprocessing scripts and are git-ignored. Both source datasets ship with
  public R packages.
- **Results.** Everything under `results/` is regenerable.
- **The manuscript.** The `.tex`, `.bib` and PDFs live in the `revised-tex`
  folder, separate from this one. They are not part of the code repository.
