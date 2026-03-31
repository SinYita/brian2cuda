#!/usr/bin/env bash
#
# Remote verification script for Issue #269 work (heterogeneous delay multi-block effects).
#
# What it does (on a GPU/CUDA machine):
# - Clones/updates the repo
# - Checks out a branch/commit
# - Initializes frozen_repos/brian2 submodule and applies brian2.diff (idempotent)
# - Creates a Python venv and installs dependencies
# - Installs brian2cuda + frozen Brian2 (as required by this repo's README)
# - Runs pytest for brian2cuda/tests
# - Optionally runs ./bench.sh to produce a Phase 4 report
#
# Usage examples:
#   bash remote_verify_issue269.sh
#   bash remote_verify_issue269.sh --branch heterog-delays-parallel-effects
#   bash remote_verify_issue269.sh --workdir ~/work --run-bench
#   bash remote_verify_issue269.sh --repo https://github.com/SinYita/brian2cuda.git --commit <sha>
#
set -euo pipefail

REPO_URL="https://github.com/SinYita/brian2cuda.git"
BRANCH="heterog-delays-parallel-effects"
COMMIT=""
WORKDIR="${HOME}/work"
REPODIR_NAME="brian2cuda"
PYTHON_BIN="python3"
VENV_DIR_NAME=".venv-brian2cuda-issue269"
RUN_BENCH="0"
PYTEST_ARGS="-q"
TEST_TARGET="brian2cuda/tests"

usage() {
  cat <<'EOF'
remote_verify_issue269.sh [options]

Options:
  --repo <url>          Repo URL (default: https://github.com/SinYita/brian2cuda.git)
  --branch <name>       Branch to checkout (default: heterog-delays-parallel-effects)
  --commit <sha>        Checkout a specific commit (overrides --branch)
  --workdir <path>      Working directory on server (default: ~/work)
  --python <bin>        Python executable (default: python3)
  --venv <dir-name>     Venv directory name (default: .venv-brian2cuda-issue269)
  --pytest-args "<arg>" Extra pytest args (default: -q)
  --tests <path>        Test path to run (default: brian2cuda/tests)
  --run-bench           Also run ./bench.sh after tests
  -h, --help            Show this help

Notes:
  - Requires: git, a working CUDA toolchain (nvcc), and NVIDIA driver (nvidia-smi).
  - This repo expects using frozen_repos/brian2 + applying brian2.diff (per README.md).
EOF
}

log() { printf "[%s] %s\n" "$(date +'%H:%M:%S')" "$*"; }
die() { printf "[%s] ERROR: %s\n" "$(date +'%H:%M:%S')" "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_URL="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --commit) COMMIT="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --venv) VENV_DIR_NAME="$2"; shift 2 ;;
    --pytest-args) PYTEST_ARGS="$2"; shift 2 ;;
    --tests) TEST_TARGET="$2"; shift 2 ;;
    --run-bench) RUN_BENCH="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

command -v git >/dev/null 2>&1 || die "git not found"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python not found: ${PYTHON_BIN}"

log "Workdir: ${WORKDIR}"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

REPODIR="${WORKDIR}/${REPODIR_NAME}"

if [[ -d "${REPODIR}/.git" ]]; then
  log "Repo exists. Updating."
  git -C "${REPODIR}" fetch --all --tags
else
  log "Cloning ${REPO_URL}"
  git clone "${REPO_URL}" "${REPODIR}"
fi

cd "${REPODIR}"

if [[ -n "${COMMIT}" ]]; then
  log "Checking out commit ${COMMIT}"
  git checkout --detach "${COMMIT}"
else
  log "Checking out branch ${BRANCH}"
  git checkout "${BRANCH}"
  git pull --ff-only || true
fi

log "Git HEAD: $(git rev-parse HEAD)"

log "Checking GPU/CUDA toolchain presence"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || die "nvidia-smi failed (driver/GPU not available)"
else
  die "nvidia-smi not found (need NVIDIA driver tools)"
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || die "nvcc exists but failed"
else
  die "nvcc not found (need CUDA toolkit on server)"
fi

log "Detecting GPU compute capability via nvidia-smi"
GPU_COMPUTE_CAPABILITY="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
if [[ -z "${GPU_COMPUTE_CAPABILITY}" ]]; then
  die "Could not detect GPU compute capability via nvidia-smi"
fi
log "Compute capability: ${GPU_COMPUTE_CAPABILITY}"

log "Initializing submodule frozen_repos/brian2"
git submodule update --init frozen_repos/brian2

log "Applying frozen Brian2 patch brian2.diff (idempotent)"
if [[ -f "frozen_repos/brian2.diff" ]]; then
  if git -C frozen_repos/brian2 apply --check ../brian2.diff >/dev/null 2>&1; then
    git -C frozen_repos/brian2 apply ../brian2.diff
    log "Applied brian2.diff"
  else
    log "brian2.diff already applied (or not applicable); continuing"
  fi
else
  die "Missing frozen_repos/brian2.diff"
fi

VENV_DIR="${REPODIR}/${VENV_DIR_NAME}"
if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

log "Upgrading packaging tools"
python -m pip install -U pip setuptools wheel

log "Ensuring pytest available"
python -m pip install -U pytest

log "Pinning NumPy for frozen Brian2 compatibility (np.bool removal in NumPy>=1.24)"
python -m pip install -U "numpy<1.24"

log "Uninstalling any existing brian2 to avoid mixing versions"
python -m pip uninstall -y brian2 >/dev/null 2>&1 || true

log "Installing brian2cuda (editable) + frozen Brian2 (editable)"
python -m pip install -e .
python -m pip install -e ./frozen_repos/brian2

log "Sanity check import paths"
python - <<'PY'
import brian2, brian2cuda
print("brian2:", brian2.__file__)
print("brian2cuda:", brian2cuda.__file__)
PY

log "Running test suite via brian2 PreferencePlugin (sets config.brian_prefs)"
# The frozen Brian2 version used by this repo relies on config.brian_prefs/config.device
# being injected via brian2.tests.PreferencePlugin. Running pytest directly will
# lead to many errors like:
#   AttributeError: 'Config' object has no attribute 'brian_prefs'
python - <<PY
import pytest
import brian2cuda  # registers brian2cuda preferences (devices.cuda_standalone)
from brian2.tests import PreferencePlugin
from brian2.core.preferences import prefs

#
# Avoid dependency on CUDA Samples' deviceQuery binary.
# Use nvidia-smi's compute_cap result and disable GPU auto-detection.
#
prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
prefs.devices.cuda_standalone.cuda_backend.gpu_id = 0
prefs.devices.cuda_standalone.cuda_backend.compute_capability = float("${GPU_COMPUTE_CAPABILITY}")

pref = PreferencePlugin(prefs, fail_for_not_implemented=False)
pref.device = "cuda_standalone"
pref.device_options = {"directory": None, "with_output": False, "build_on_run": False}

args = "${PYTEST_ARGS}".split() + ["${TEST_TARGET}"]
raise SystemExit(pytest.main(args, plugins=[pref]))
PY

if [[ "${RUN_BENCH}" == "1" ]]; then
  if [[ -f "./bench.sh" ]]; then
    log "Running bench.sh for Phase 4 verification"
    chmod +x ./bench.sh
    ./bench.sh
    log "Latest Phase 4 report:"
    ls -1dt benchmark_results_* 2>/dev/null | head -n 1 | while read -r d; do
      if [[ -f "${d}/PHASE4_REPORT.md" ]]; then
        echo "---- ${d}/PHASE4_REPORT.md (head) ----"
        head -n 80 "${d}/PHASE4_REPORT.md"
      else
        echo "No PHASE4_REPORT.md found under ${d}"
      fi
    done
  else
    die "Requested --run-bench but bench.sh not found in repo root"
  fi
fi

log "DONE. If pytest passed (and optionally Phase4 report looks good), the issue fix is very likely validated."

