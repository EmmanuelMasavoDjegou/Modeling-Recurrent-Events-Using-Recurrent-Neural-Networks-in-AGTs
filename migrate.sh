#!/usr/bin/env bash
###############################################################################
# migrate.sh - restructure the existing repository in place, preserving history
#
# Run from the repository root, on a branch, with a clean working tree:
#
#   git switch -c restructure
#   bash migrate.sh
#   git status          # review before committing
#
# `git mv` is used throughout so that `git log --follow` still traces each file
# back through the rename. Copying the new files over the old ones instead would
# show up as a mass delete-and-add and lose that history.
#
# This script only MOVES and REMOVES. Afterwards, copy the new `rnn_agt/`,
# `experiments/`, notebooks and R scripts into place, then commit.
###############################################################################
set -euo pipefail

if [ ! -d .git ]; then
  echo "error: run this from the repository root (no .git found)." >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree is not clean. Commit or stash first." >&2
  exit 1
fi

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
  echo "error: you are on '$branch'. Create a branch first:" >&2
  echo "  git switch -c restructure" >&2
  exit 1
fi

echo "==> creating new directories"
mkdir -p rnn_agt experiments data results

echo "==> renaming top-level directories"
[ -d "1.Literature Review" ] && git mv "1.Literature Review" literature_review
[ -d "2.Simulation" ]        && git mv "2.Simulation"        simulation
[ -d "3.Application" ]       && git mv "3.Application"       application

echo "==> renaming notebooks to describe what they do"
# The old v1/v2 suffixes do not tell a reader which notebook is current, and
# this repository is now cited in a published paper. Names describe purpose.
cd simulation
[ -f "RNN_AGT_v2.ipynb" ]                          && git mv "RNN_AGT_v2.ipynb"                          model_demo.ipynb
[ -f "RNN-AGT_v1.ipynb" ]                          && git mv "RNN-AGT_v1.ipynb"                          defect_impact.ipynb
[ -f "RNN-AGT.ipynb" ]                             && git mv "RNN-AGT.ipynb"                             defect_impact.ipynb
[ -f "Mod_Perform_Sub_sampling_Config.ipynb" ]     && git mv "Mod_Perform_Sub_sampling_Config.ipynb"     subsampling_sensitivity.ipynb
[ -f "Mod_Perform_Sub_sampling_Config_v2_ipynb.ipynb" ] && git mv "Mod_Perform_Sub_sampling_Config_v2_ipynb.ipynb" subsampling_sensitivity.ipynb
[ -f "High_Dim_Cov_ipynb.ipynb" ]                  && git mv "High_Dim_Cov_ipynb.ipynb"                  high_dimensional.ipynb
[ -f "SimulationPlots.ipynb" ]                     && git mv "SimulationPlots.ipynb"                     figures.ipynb
cd ..

cd application
[ -f "Application_v2.ipynb" ] && git mv "Application_v2.ipynb" real_data_analysis.ipynb
[ -f "Application_v1.ipynb" ] && git mv "Application_v1.ipynb" split_sensitivity.ipynb
# R filenames on disk had spaces where the old README used underscores
[ -f "Dataset1_Data Preprocessing.R" ] && git mv "Dataset1_Data Preprocessing.R" Dataset1_Data_Preprocessing.R
[ -f "Dataset2_Data Preprocessing.R" ] && git mv "Dataset2_Data Preprocessing.R" Dataset2_Data_Preprocessing.R
cd ..

echo
echo "==> done. Next steps:"
cat <<'EOF'

  1. Copy the new files over the renamed ones:

       cp -r <new>/rnn_agt/.       rnn_agt/
       cp -r <new>/experiments/.   experiments/
       cp    <new>/simulation/*.ipynb   simulation/
       cp    <new>/application/*.ipynb  application/
       cp    <new>/application/*.R      application/
       cp    <new>/README.md <new>/CORRECTIONS.md <new>/LICENSE \
             <new>/CITATION.cff <new>/requirements.txt <new>/.gitignore .
       cp    <new>/data/README.md    data/
       cp    <new>/results/README.md results/

  2. Verify before committing:

       python experiments/run_diagnostics.py      # five checks, all PASS
       git status
       git log --follow -- simulation/model_demo.ipynb   # history preserved

  3. Commit:

       git add -A
       git commit -m "Restructure: extract rnn_agt package, correct four defects"

  4. Old generated CSVs are now git-ignored. If data_cp.csv or
     cgd_preprocessed.csv are tracked, untrack them:

       git rm --cached data_cp.csv cgd_preprocessed.csv 2>/dev/null || true

EOF
