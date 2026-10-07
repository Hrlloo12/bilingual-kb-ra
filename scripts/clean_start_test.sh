#!/usr/bin/env bash
set -euo pipefail

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="results/deployment/clean_start_${STAMP}"
READY_TIMEOUT_S=${READY_TIMEOUT_S:-2700}
COMPOSE=(docker compose --profile gpu)
APP_UID=1000
mkdir -p "$OUT"
chown -R "$APP_UID:$APP_UID" results 2>/dev/null || sudo chown -R "$APP_UID:$APP_UID" results

echo "== removing containers, volumes and images"
"${COMPOSE[@]}" down --volumes --remove-orphans
if [ "${PURGE_IMAGES:-1}" = "1" ]; then
  docker image rm -f kb-rag-api:local vllm/vllm-openai:v0.10.1.1 opensearchproject/opensearch:2.19.1 qdrant/qdrant:v1.14.1 valkey/valkey:8.1 2>/dev/null || true
  docker builder prune --all --force >/dev/null
fi

started=$(date +%s)
echo "== building"
"${COMPOSE[@]}" build --no-cache --pull
built=$(date +%s)

echo "== starting"
"${COMPOSE[@]}" up -d
api_id=$("${COMPOSE[@]}" ps -q api)
until [ "$(docker inspect -f '{{.State.Health.Status}}' "$api_id")" = "healthy" ]; do
  if [ $(( $(date +%s) - built )) -gt "$READY_TIMEOUT_S" ]; then
    echo "api not healthy after ${READY_TIMEOUT_S}s" >&2
    "${COMPOSE[@]}" ps -a > "$OUT/compose_ps.txt"
    "${COMPOSE[@]}" logs --no-color --tail 200 > "$OUT/compose_logs_tail.txt"
    exit 1
  fi
  sleep 10
done
ready=$(date +%s)

echo "== smoke test"
smoke_status=0
"${COMPOSE[@]}" exec -T api python scripts/smoke_api.py --base-url http://localhost:8080 --output "/app/$OUT/smoke.json" || smoke_status=$?

"${COMPOSE[@]}" ps -a > "$OUT/compose_ps.txt"
"${COMPOSE[@]}" logs --no-color indexer > "$OUT/indexer.log"
"${COMPOSE[@]}" logs --no-color vllm > "$OUT/vllm.log"
"${COMPOSE[@]}" logs --no-color api > "$OUT/api.log"
nvidia-smi > "$OUT/nvidia_smi.txt"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv > "$OUT/gpu.csv"
docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' > "$OUT/docker_stats.csv"
docker image ls --format '{{.Repository}}:{{.Tag}},{{.Size}}' > "$OUT/images.csv"
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true

cat > "$OUT/timing.json" <<EOF
{
  "build_seconds": $((built - started)),
  "start_to_healthy_seconds": $((ready - built)),
  "total_seconds": $((ready - started)),
  "smoke_exit_code": $smoke_status
}
EOF
cat "$OUT/timing.json"
exit "$smoke_status"
