#!/usr/bin/env bash
set -euo pipefail

# Convert knee ONNX/PT model to TRT and run smoke tests.
#
# Usage:
#   bash knee-exo-ctrl/convert_and_smoketest_trt.sh
#   bash knee-exo-ctrl/convert_and_smoketest_trt.sh --env jinwoo-addbiomech
#   bash knee-exo-ctrl/convert_and_smoketest_trt.sh --precision fp32 --seq-len 100
#
# Notes:
# - Run on Jetson / CUDA host with TensorRT installed.
# - Defaults are set for runs/0423_ik_id_knee_huber_noise.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KNEE_DIR="${ROOT_DIR}/knee-exo-ctrl"

ENV_NAME="jinwoo-addbiomech"
CHECKPOINT="${ROOT_DIR}/runs/0423_ik_id_knee_huber_noise/best_model.pt"
ONNX_PATH="${KNEE_DIR}/best_model_0423_knee.onnx"
TRT_PATH="${KNEE_DIR}/best_model_0423_knee.trt"
SEQ_LEN=""
PRECISION="fp16"
WORKSPACE_GB="2.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"; shift 2 ;;
    --checkpoint)
      CHECKPOINT="$2"; shift 2 ;;
    --onnx)
      ONNX_PATH="$2"; shift 2 ;;
    --trt)
      TRT_PATH="$2"; shift 2 ;;
    --seq-len)
      SEQ_LEN="$2"; shift 2 ;;
    --precision)
      PRECISION="$2"; shift 2 ;;
    --workspace-gb)
      WORKSPACE_GB="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0 ;;
    *)
      echo "Unknown arg: $1"
      exit 1 ;;
  esac
done

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

echo "=== TRT Conversion + Smoke Test ==="
echo "Env        : ${ENV_NAME}"
echo "Checkpoint : ${CHECKPOINT}"
echo "ONNX       : ${ONNX_PATH}"
echo "TRT        : ${TRT_PATH}"
echo "Precision  : ${PRECISION}"
echo "Workspace  : ${WORKSPACE_GB} GB"
if [[ -n "${SEQ_LEN}" ]]; then
  echo "Seq len    : ${SEQ_LEN}"
fi
echo

cd "${ROOT_DIR}"

CONVERT_CMD=(conda run -n "${ENV_NAME}" python "${KNEE_DIR}/convert_to_trt.py"
  --checkpoint "${CHECKPOINT}"
  --onnx "${ONNX_PATH}"
  --output "${TRT_PATH}"
  --precision "${PRECISION}"
  --workspace-gb "${WORKSPACE_GB}"
  --keep-onnx
  --verify
)
if [[ -n "${SEQ_LEN}" ]]; then
  CONVERT_CMD+=(--seq-len "${SEQ_LEN}")
fi

echo "[1/3] Converting PT/ONNX -> TRT ..."
"${CONVERT_CMD[@]}"

echo
echo "[2/3] Smoke test: ONNX numerical verify ..."
VERIFY_CMD=(conda run -n "${ENV_NAME}" python "${KNEE_DIR}/run_onnx.py" verify
  --onnx "${ONNX_PATH}"
  --checkpoint "${CHECKPOINT}"
)
if [[ -n "${SEQ_LEN}" ]]; then
  VERIFY_CMD+=(--seq-len "${SEQ_LEN}")
fi
"${VERIFY_CMD[@]}"

echo
echo "[3/3] Smoke test: ONNX latency bench ..."
BENCH_CMD=(conda run -n "${ENV_NAME}" python "${KNEE_DIR}/run_onnx.py" bench
  --onnx "${ONNX_PATH}"
  --checkpoint "${CHECKPOINT}"
  --warmup 50
  --n-reps 200
)
if [[ -n "${SEQ_LEN}" ]]; then
  BENCH_CMD+=(--seq-len "${SEQ_LEN}")
fi
"${BENCH_CMD[@]}"

echo
echo "Done."
echo "TRT engine: ${TRT_PATH}"
