#!/usr/bin/env bash
# Deploy the page from this machine with the live model, not whatever frontend/model
# holds: after a retrain the Modal volume has a newer one. Use this, never the bare
# `wrangler pages deploy`, once the retrain runs on Modal.
#
#   QZ_LIVE_MODEL=1 bash scripts/deploy_page.sh   # takes the best model from Modal
#   bash scripts/deploy_page.sh                   # before retraining exists: as it is
#
# Needs the deploy extra (pip install -e ".[deploy]") and `modal setup` once.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${QZ_RUN:-pilot-final}
VOLUME=quorizero-runs

if [ -n "${QZ_LIVE_MODEL:-}" ]; then
  files=$(modal volume ls "$VOLUME" "$RUN")  # fails loudly if Modal can't be reached
  if grep -q "best.txt" <<<"$files"; then  # a promoted retrain: that is the live model
    run=$(mktemp -d)
    modal volume get "$VOLUME" "$RUN/best.txt" "$run/best.txt"
    best=$(cat "$run/best.txt")
    modal volume get "$VOLUME" "$RUN/config.json" "$run/config.json"
    modal volume get "$VOLUME" "$RUN/$best" "$run/"
    python -m model.export "$run/$best" frontend/model --version "$RUN/$best"
    rm -rf "$run"
  else
    echo "no retrain promoted yet, keeping frontend/model"
  fi
fi

# the retrain deploys the page code on GitHub main, so local-only changes get undone
git fetch -q origin main || true
if ! git diff --quiet origin/main -- frontend; then
  echo "warning: frontend/ differs from GitHub main, push it or the next retrain undoes it"
fi

echo "deploying model $(cat frontend/model/version.json)"
npx wrangler pages deploy frontend --project-name "${QZ_PAGES_PROJECT:-quorizero}" --branch main
