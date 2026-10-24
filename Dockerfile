# Retraining job for Cloud Run with a GPU (scripts/retrain_job.sh, docs/deploy.md)
# gcloud builds submit --tag <region>-docker.pkg.dev/<project>/qz/retrain
FROM node:22-bookworm-slim AS node

FROM python:3.11-slim-bookworm
# Node, only for wrangler: redeploys the page after a promotion
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && npm install -g wrangler@4.151.3

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
# the CUDA libraries come as pip wheels, the GPU driver from Cloud Run
RUN pip install --no-cache-dir -e ".[ml,cloud]" "jax[cuda12]==0.10.2"
COPY . .
ENTRYPOINT ["bash", "scripts/retrain_job.sh"]
