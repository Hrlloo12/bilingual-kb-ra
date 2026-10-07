#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_REPO:?}" "${CODE_ARCHIVE:?}" "${RUN_NAME:?}"
OPENSEARCH_VERSION=2.19.1
QDRANT_VERSION=v1.14.1
GENERATOR_MODEL=${GENERATOR_MODEL:-Qwen/Qwen3-4B-Instruct-2507-FP8}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.55}
EVAL_SPLITS=${EVAL_SPLITS:-validation test}
WORK=/work
mkdir -p "$WORK" && cd "$WORK"

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

uv venv "$WORK/venv" --python 3.12 -q
. "$WORK/venv/bin/activate"
uv pip install -q vllm==0.10.1.1 transformers==4.55.4 sentence-transformers==4.1.0 opensearch-py==2.8.0 qdrant-client==1.14.2   pydantic==2.11.7 PyYAML==6.0.2 Jinja2==3.1.6 python-docx==1.1.2 PyMuPDF==1.25.5 beautifulsoup4==4.13.4 lxml==5.4.0   httpx==0.28.1 valkey==6.1.0 numpy

nvidia-smi
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 2 > "$WORK/gpu.csv" &

python - <<'EOF'
import os, tarfile
from huggingface_hub import hf_hub_download
archive = hf_hub_download(os.environ["DATASET_REPO"], os.environ["CODE_ARCHIVE"], repo_type="dataset")
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

vllm serve "$GENERATOR_MODEL" --gpu-memory-utilization "$VLLM_GPU_UTIL" --max-model-len 8192 --port 8000 > "$WORK/vllm.log" 2>&1 &
wait_for http://127.0.0.1:9200 || { tail -50 "$WORK/opensearch.log"; exit 1; }
wait_for http://127.0.0.1:6333/readyz || { tail -50 "$WORK/qdrant.log"; exit 1; }

cd "$WORK/repo"
python scripts/build_index.py | tee "$WORK/build_index.json"
wait_for http://127.0.0.1:8000/health || { tail -80 "$WORK/vllm.log"; exit 1; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > "$WORK/gpu_after_load.csv"

python scripts/evaluate_smart_search.py --splits validation --abstain-threshold 0 --label "${RUN_NAME}_ungated" --output-dir "$WORK/out"
THRESHOLD=$(python scripts/calibrate_abstention.py --records "$WORK/out/${RUN_NAME}_ungated_validation.jsonl" --output "$WORK/out/abstention_end_to_end.json")
echo "validation-calibrated abstention threshold: $THRESHOLD"
python scripts/evaluate_smart_search.py --splits $EVAL_SPLITS --abstain-threshold "$THRESHOLD" --label "$RUN_NAME" --output-dir "$WORK/out"

uv pip freeze > "$WORK/out/pip_freeze.txt"
python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)" > "$WORK/out/versions.txt"
cp "$WORK"/gpu.csv "$WORK"/gpu_after_load.csv "$WORK"/build_index.json "$WORK"/vllm.log "$WORK/out/"
python - <<'EOF'
import os
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="/work/out",
    path_in_repo=f"runs/{os.environ['RUN_NAME']}",
    repo_id=os.environ["DATASET_REPO"],
    repo_type="dataset",
    commit_message=f"Smart AI Search run {os.environ['RUN_NAME']}",
)
EOF
echo "run complete: runs/$RUN_NAME"
