#!/usr/bin/env bash
# ===========================================================================
# run_app.sh — Launch the Streamlit app inside a Run:ai pod.
#
# The app binds to 0.0.0.0:8501 so it is reachable from outside the pod
# (via Run:ai nodeport, port-forward, or ingress).
#
# Usage:
#   bash scripts/run_app.sh           # default port 8501
#   PORT=8080 bash scripts/run_app.sh # custom port
# ===========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8501}"

# Quick sanity checks
if ! python -c "import streamlit" 2>/dev/null; then
    echo "ERROR: streamlit not installed. Run setup first:"
    echo "  bash scripts/setup_poweredge_pod.sh"
    exit 1
fi

DB_PATH="$REPO_ROOT/data/embeddings/wattbot_jinav4.db"
if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Vector database not found at $DB_PATH"
    echo "Run setup first:  bash scripts/setup_poweredge_pod.sh"
    exit 1
fi

CLUSTER_HOST="${CLUSTER_HOST:-deepthought.doit.wisc.edu}"

echo ""
echo "============================================================"
if [[ -n "${STREAMLIT_BASE_PATH:-}" ]]; then
    echo "  App URL: https://${CLUSTER_HOST}${STREAMLIT_BASE_PATH}/"
else
    echo "  App URL: http://localhost:$PORT"
    echo "  (Set STREAMLIT_BASE_PATH for the full proxy URL)"
fi
echo "============================================================"
echo ""

exec streamlit run app.py \
    --server.port="$PORT" \
    --server.address="0.0.0.0" \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
