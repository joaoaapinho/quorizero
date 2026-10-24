#!/usr/bin/env bash
# Cloud Run job (see Dockerfile, docs/deploy.md): retrain when the website's
# weighted loss count is due, then redeploy the page if the new model passed the gate.
#
#   QZ_API, QZ_EXPORT_TOKEN                      games API and owner token
#   QZ_BUCKET                                    gs://<bucket>/runs, where runs live between jobs
#   QZ_RUN                                       run to improve (default pilot-final)
#   CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID  redeploy the page (Pages: Edit only)
#   WANDB_API_KEY                                optional: log the run to W&B
#   QZ_FORCE=1                                   optional: run even if not due (tests)
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${QZ_RUN:-pilot-final}
extra=()
[ -n "${WANDB_API_KEY:-}" ] && extra+=(--wandb)
[ -n "${QZ_FORCE:-}" ] && extra+=(--force)

if [ -z "${QZ_FORCE:-}" ]; then
  set +e
  python -m ingestion.retrain --api "$QZ_API" --check
  due=$?
  set -e
  if [ "$due" = 10 ]; then echo "not due, nothing to do"; exit 0; fi
  [ "$due" = 0 ] || exit "$due"
fi

python scripts/sync_run.py down "$QZ_BUCKET/$RUN" "runs/$RUN"
rm -rf frontend/model  # exported again only if the new model is promoted
python -m ingestion.retrain --run "$RUN" --api "$QZ_API" "${extra[@]}"
python scripts/sync_run.py up "runs/$RUN" "$QZ_BUCKET/$RUN"  # checkpoint and logs

if [ -f frontend/model/model.onnx ]; then
  wrangler pages deploy frontend --project-name "${QZ_PAGES_PROJECT:-quorizero}" --branch main
else
  echo "kept the current model, the site is unchanged"
fi
