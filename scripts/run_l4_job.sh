#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_REPO:?}" "${CODE_ARCHIVE:?}" "${RUN_NAME:?}"
TASK=${TASK:-smart_search}
OPENSEARCH_VERSION=2.19.1
QDRANT_VERSION=v1.14.1
VALKEY_VERSIONS="8.1.1 8.1.0"
GENERATOR_MODEL=${GENERATOR_MODEL:-Qwen/Qwen3-4B-Instruct-2507-FP8}
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3-8B-FP8}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.55}
EVAL_SPLITS=${EVAL_SPLITS:-validation test}
export API_PORT=${API_PORT:-8080}
PRICING_QUERY="أبغى أعرف الـ pricing حق الباقة المؤسسية"
WORK=/work
mkdir -p "$WORK/out" && cd "$WORK"

fetch() {
  python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$1" "$2"
}

wait_for() {
  for _ in $(seq 1 300); do
    if python -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)" "$1" 2>/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "timed out waiting for $1" >&2
  return 1
}

upload_out() {
  python - "$1" <<'EOF'
import os, sys
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="/work/out",
    path_in_repo=f"runs/{os.environ['RUN_NAME']}",
    repo_id=os.environ["DATASET_REPO"],
    repo_type="dataset",
    commit_message=f"L4 run {os.environ['RUN_NAME']}: {sys.argv[1]}",
)
EOF
}

uv venv "$WORK/venv" --python 3.12 -q
. "$WORK/venv/bin/activate"
uv pip install -q vllm==0.10.1.1 transformers==4.55.4 sentence-transformers==4.1.0 opensearch-py==2.8.0 qdrant-client==1.14.2 \
  pydantic==2.11.7 PyYAML==6.0.2 Jinja2==3.1.6 python-docx==1.1.2 PyMuPDF==1.25.5 beautifulsoup4==4.13.4 lxml==5.4.0 \
  httpx==0.28.1 valkey==6.1.0 fastapi==0.115.12 uvicorn==0.34.2 psutil numpy sacrebleu==2.5.1

nvidia-smi
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 2 > "$WORK/gpu.csv" &
python - > "$WORK/host.csv" <<'EOF' &
import time
import psutil
groups = {"vllm": ("vllm", "VLLM"), "opensearch": ("java",), "qdrant": ("qdrant",), "valkey": ("valkey",), "api": ("rag.api",)}
print("timestamp,cpu_percent,ram_used_mib," + ",".join(f"{name}_rss_mib" for name in groups), flush=True)
psutil.cpu_percent(None)
while True:
    time.sleep(5)
    rss = dict.fromkeys(groups, 0.0)
    for process in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            label = " ".join(process.info["cmdline"] or [process.info["name"] or ""])
            for name, needles in groups.items():
                if any(needle in label for needle in needles):
                    rss[name] += process.info["memory_info"].rss / 2**20
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    memory = psutil.virtual_memory()
    print(f"{time.time():.0f},{psutil.cpu_percent(None)},{(memory.total - memory.available) / 2**20:.0f}," + ",".join(f"{rss[name]:.0f}" for name in groups), flush=True)
EOF

python - <<'EOF'
import os, tarfile
from huggingface_hub import hf_hub_download
source = os.environ["CODE_ARCHIVE"]
archive = source if os.path.isfile(source) else hf_hub_download(os.environ["DATASET_REPO"], source, repo_type="dataset")
tarfile.open(archive).extractall("/work/repo")
EOF

fetch "https://artifacts.opensearch.org/releases/core/opensearch/${OPENSEARCH_VERSION}/opensearch-min-${OPENSEARCH_VERSION}-linux-x64.tar.gz" opensearch.tar.gz
fetch "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-musl.tar.gz" qdrant.tar.gz
tar -xzf opensearch.tar.gz && mkdir -p qdrant && tar -xzf qdrant.tar.gz -C qdrant
./qdrant/qdrant --version

uv pip install -q --no-deps -e "$WORK/repo"

id search >/dev/null 2>&1 || useradd -m search
touch "$WORK/opensearch.log"
chown -R search "$WORK/opensearch-${OPENSEARCH_VERSION}" "$WORK/opensearch.log"
su search -c "OPENSEARCH_JAVA_OPTS='-Xms1g -Xmx1g' $WORK/opensearch-${OPENSEARCH_VERSION}/bin/opensearch -E discovery.type=single-node -E network.host=127.0.0.1 > $WORK/opensearch.log 2>&1 &"
(cd qdrant && ./qdrant > "$WORK/qdrant.log" 2>&1 &)

if [ "$TASK" = "interactive" ] || [ "$TASK" = "rewrite_candidates" ] || [ "$TASK" = "benchmark" ]; then
  for version in $VALKEY_VERSIONS; do
    if fetch "https://download.valkey.io/releases/valkey-${version}-jammy-x86_64.tar.gz" valkey.tar.gz 2>/dev/null; then
      echo "valkey ${version}"
      break
    fi
  done
  mkdir -p valkey && tar -xzf valkey.tar.gz -C valkey --strip-components=1
  ./valkey/bin/valkey-server --version
  ./valkey/bin/valkey-server --port 6379 --save "" --appendonly no --maxmemory 256mb --maxmemory-policy allkeys-lru > "$WORK/valkey.log" 2>&1 &
fi

vllm serve "$GENERATOR_MODEL" --gpu-memory-utilization "$VLLM_GPU_UTIL" --max-model-len 8192 --port 8000 > "$WORK/vllm.log" 2>&1 &
VLLM_PID=$!
wait_for http://127.0.0.1:9200 || { tail -50 "$WORK/opensearch.log"; exit 1; }
wait_for http://127.0.0.1:6333/readyz || { tail -50 "$WORK/qdrant.log"; exit 1; }

cd "$WORK/repo"
python scripts/build_index.py | tee "$WORK/build_index.json"
wait_for http://127.0.0.1:8000/health || { tail -80 "$WORK/vllm.log"; exit 1; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > "$WORK/gpu_after_load.csv"

if [ "$TASK" = "smart_search" ]; then
  if [ -n "${FIXED_THRESHOLD:-}" ]; then
    THRESHOLD=$FIXED_THRESHOLD
    echo "fixed abstention threshold: $THRESHOLD"
  else
    python scripts/evaluate_smart_search.py --splits validation --abstain-threshold 0 --label "${RUN_NAME}_ungated" --output-dir "$WORK/out"
    THRESHOLD=$(python scripts/calibrate_abstention.py --records "$WORK/out/${RUN_NAME}_ungated_validation.jsonl" --output "$WORK/out/abstention_end_to_end.json")
    echo "validation-calibrated abstention threshold: $THRESHOLD"
  fi
  python scripts/evaluate_smart_search.py --splits $EVAL_SPLITS --abstain-threshold "$THRESHOLD" --label "$RUN_NAME" --output-dir "$WORK/out"
elif [ "$TASK" = "interactive" ]; then
  python -m rag.api > "$WORK/api.log" 2>&1 &
  wait_for http://127.0.0.1:${API_PORT}/health || { tail -80 "$WORK/api.log"; exit 1; }
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv > "$WORK/gpu_after_api.csv"
  python scripts/smoke_api.py --base-url http://127.0.0.1:${API_PORT} --output "$WORK/out/smoke.json" || true
  python - > "$WORK/out/pricing_query.json" <<EOF
import json, httpx
body = httpx.post("http://127.0.0.1:${API_PORT}/v1/search", json={"mode": "smart_search", "query": "$PRICING_QUERY"}, timeout=120).json()
print(json.dumps({key: body[key] for key in ("query", "status", "answer", "abstain_reason", "retrieved", "latency_ms")}, ensure_ascii=False, indent=2))
EOF
  python scripts/evaluate_interactive.py --base-url http://127.0.0.1:${API_PORT} --splits $EVAL_SPLITS --label "$RUN_NAME" --output-dir "$WORK/out"
  cp "$WORK"/api.log "$WORK"/gpu_after_api.csv "$WORK"/host.csv "$WORK/out/"
elif [ "$TASK" = "rewrite_candidates" ]; then
  for candidate in ${CANDIDATES:-v1 v2 v3}; do
    REWRITE_PROMPT="$candidate" python -m rag.api > "$WORK/api_${candidate}.log" 2>&1 &
    api_pid=$!
    wait_for http://127.0.0.1:${API_PORT}/health || { tail -80 "$WORK/api_${candidate}.log"; exit 1; }
    python scripts/evaluate_interactive.py --base-url http://127.0.0.1:${API_PORT} --splits validation --systems interactive --label "rewrite_${candidate}" --output-dir "$WORK/out"
    kill "$api_pid" && wait "$api_pid" || true
    cp "$WORK"/host.csv "$WORK/out/"
    upload_out "candidate ${candidate} done"
  done
  cp "$WORK"/host.csv "$WORK/out/"
elif [ "$TASK" = "benchmark" ]; then
  python -m rag.api > "$WORK/api.log" 2>&1 &
  api_pid=$!
  wait_for http://127.0.0.1:${API_PORT}/health || { tail -80 "$WORK/api.log"; exit 1; }
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv > "$WORK/out/gpu_after_api.csv"
  python scripts/smoke_api.py --base-url http://127.0.0.1:${API_PORT} --output "$WORK/out/smoke.json" || true
  python scripts/evaluate_smart_search.py --splits validation test --label final --output-dir "$WORK/out/smart_search"
  upload_out "smart search evaluation"
  python scripts/benchmark_concurrency.py --base-url http://127.0.0.1:${API_PORT} --mode smart_search --output "$WORK/out/concurrency/smart_search.json"
  python scripts/benchmark_concurrency.py --base-url http://127.0.0.1:${API_PORT} --mode quick_search --users 1 4 16 --output "$WORK/out/concurrency/quick_search.json"
  cp "$WORK"/host.csv "$WORK"/api.log "$WORK/out/"
  upload_out "concurrency"
  kill "$api_pid" && wait "$api_pid" || true
  python scripts/benchmark_scaling.py --output "$WORK/out/scaling/scaling.json"
  cp "$WORK"/host.csv "$WORK/out/"
  upload_out "scaling"
  kill "$VLLM_PID" && wait "$VLLM_PID" || true
  vllm serve "$JUDGE_MODEL" --gpu-memory-utilization 0.85 --max-model-len 8192 --port 8001 > "$WORK/judge.log" 2>&1 &
  wait_for http://127.0.0.1:8001/health || { tail -80 "$WORK/judge.log"; exit 1; }
  python scripts/evaluate_generation.py --records "$WORK/out/smart_search/final_test.jsonl" --split test --judge-url http://127.0.0.1:8001/v1 --judge-model "$JUDGE_MODEL" --output "$WORK/out/generation/generation_test.json"
  upload_out "generation metrics"
else
  echo "unknown TASK $TASK" >&2
  exit 1
fi

uv pip freeze > "$WORK/out/pip_freeze.txt"
python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)" > "$WORK/out/versions.txt"
cp "$WORK"/gpu.csv "$WORK"/gpu_after_load.csv "$WORK"/build_index.json "$WORK"/vllm.log "$WORK/out/"
upload_out "complete"
echo "run complete: runs/$RUN_NAME"
