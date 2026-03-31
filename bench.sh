#!/bin/bash

################################################################################
# Phase 4: Testing & Performance Verification Script for Issue #269
# 
# This script runs comprehensive tests for the heterogeneous delay multi-block
# parallelism implementation (Phases 1-3). It:
# 1. Checks environment setup
# 2. Installs dependencies
# 3. Verifies code changes (Phase 1-3)
# 4. Runs regression tests
# 5. Executes Brunel-Hakim benchmark with profiling
# 6. Analyzes and compares results
#
# Usage: ./bench.sh [options]
# Options:
#   --help           Show help message
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/benchmark_results_${TIMESTAMP}"

# Test parameters
BRUNEL_N=5000
BRUNEL_DURATION="0.1*second"
HETEROG_DELAYS=True
NARROW_DELAY_DIST=True
PROFILE_RUN=True
CUDA_COMPUTE_CAPABILITY=""

# Flags
NO_COMPILE=false

################################################################################
# UTILITY FUNCTIONS
################################################################################

log_info() {
    echo "[INFO $(date +'%H:%M:%S')] $*"
}

log_error() {
    echo "[ERROR $(date +'%H:%M:%S')] $*" >&2
}

log_section() {
    echo ""
    echo "================================================================================"
    echo "  $*"
    echo "================================================================================"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command not found: $1"
        return 1
    fi
    log_info "✓ Found $1: $(command -v "$1")"
}

run_cmd() {
    local cmd="$*"
    log_info "Running: $cmd"
    eval "$cmd" || return 1
}

################################################################################
# ENVIRONMENT SETUP
################################################################################

setup_environment() {
    log_section "ENVIRONMENT SETUP"
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    log_info "Script directory: $SCRIPT_DIR"
    log_info "Results directory: $RESULTS_DIR"
    
    # Check required commands
    log_info "Checking required commands..."
    check_command "python" || return 1
    check_command "nvcc" || log_error "CUDA compiler not found. Is CUDA installed?"
    check_command "git" || return 1
    
    # Check Python version
    local python_version
    python_version=$(python --version 2>&1 | awk '{print $2}')
    log_info "Python version: $python_version"
    
    # Check CUDA version
    local cuda_version
    cuda_version=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
    log_info "CUDA version: $cuda_version"

    # Determine compute capability without relying on CUDA Samples deviceQuery.
    if [ -n "${BRIAN2CUDA_COMPUTE_CAPABILITY:-}" ]; then
        CUDA_COMPUTE_CAPABILITY="$BRIAN2CUDA_COMPUTE_CAPABILITY"
        log_info "Using compute capability from env BRIAN2CUDA_COMPUTE_CAPABILITY=${CUDA_COMPUTE_CAPABILITY}"
    else
        if command -v nvidia-smi >/dev/null 2>&1; then
            CUDA_COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')
        fi

        if [[ -z "$CUDA_COMPUTE_CAPABILITY" || ! "$CUDA_COMPUTE_CAPABILITY" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            log_error "Couldn't detect GPU compute capability via nvidia-smi. Set BRIAN2CUDA_COMPUTE_CAPABILITY manually (e.g. 8.0)."
            return 1
        fi

        log_info "Detected GPU compute capability: $CUDA_COMPUTE_CAPABILITY"
    fi
    
    return 0
}

################################################################################
# DEPENDENCY INSTALLATION
################################################################################

install_dependencies() {
    log_section "INSTALLING DEPENDENCIES"
    
    log_info "Pinning pip to a metadata-compatible version (<24.1)..."
    run_cmd "python -m pip install --upgrade 'pip<24.1' setuptools wheel" || return 1

    # Some prebuilt environments contain an invalid bleach metadata record that
    # breaks pip>=24.1 dependency processing. Clean it proactively.
    if python -m pip show bleach >/dev/null 2>&1; then
        log_info "Sanitizing bleach/tinycss2 metadata for compatibility..."
        run_cmd "python -m pip uninstall -y bleach tinycss2" || true
        run_cmd "python -m pip install 'bleach<6' 'tinycss2>=1.1.0,<1.2'" || return 1
    fi
    
    # Brian2CUDA is tightly coupled to a specific Brian2 codegen API.
    # Prefer the repository-pinned Brian2 from frozen_repos to avoid
    # template key mismatches (e.g. missing create_j/setup_iterator).
    if [ -d "${SCRIPT_DIR}/frozen_repos/brian2" ] && [ ! -f "${SCRIPT_DIR}/frozen_repos/brian2/setup.py" ] && command -v git >/dev/null 2>&1; then
        log_info "frozen_repos/brian2 exists but looks empty; initializing submodule..."
        run_cmd "git -C \"${SCRIPT_DIR}\" submodule update --init frozen_repos/brian2" || true
    fi

    if [ -d "${SCRIPT_DIR}/frozen_repos/brian2" ] && [ -f "${SCRIPT_DIR}/frozen_repos/brian2/setup.py" ]; then
        log_info "Installing repository-pinned Brian2 from frozen_repos/brian2..."

        if [ -f "${SCRIPT_DIR}/frozen_repos/brian2.diff" ] && command -v git >/dev/null 2>&1; then
            log_info "Checking whether brian2.diff needs to be applied..."
            if git -C "${SCRIPT_DIR}/frozen_repos/brian2" apply --check "${SCRIPT_DIR}/frozen_repos/brian2.diff" >/dev/null 2>&1; then
                run_cmd "git -C \"${SCRIPT_DIR}/frozen_repos/brian2\" apply \"${SCRIPT_DIR}/frozen_repos/brian2.diff\"" || return 1
            else
                log_info "brian2.diff already applied (or not applicable), continuing..."
            fi
        fi

        run_cmd "python -m pip uninstall -y brian2" || true
        run_cmd "python -m pip install \"${SCRIPT_DIR}/frozen_repos/brian2\"" || return 1
    else
        log_info "frozen_repos/brian2 not available, using conservative Brian2 pin from PyPI..."
        run_cmd "python -m pip uninstall -y brian2" || true
        run_cmd "python -m pip install 'brian2==2.4.2'" || return 1
    fi
    
    log_info "Installing other dependencies..."
    run_cmd "python -m pip install numpy scipy matplotlib pandas" || return 1
    
    # Verify Brian2 installation
    log_info "Verifying Brian2 installation..."
    python -c "import brian2; print(f'Brian2 version: {brian2.__version__}')"
    
    return 0
}

################################################################################
# ENVIRONMENT CONFIGURATION
################################################################################

configure_brian2_preferences() {
    log_section "CONFIGURING BRIAN2 PREFERENCES"
    
    log_info "Creating Brian2 preferences configuration..."
    
    # Create a Python script to configure preferences
    cat > "${RESULTS_DIR}/setup_prefs.py" << 'EOF'
from brian2 import prefs
import os

# CUDA compiler preferences
prefs.codegen.cpp.compiler_flags = ' -fPIC'  # Position-independent code

# Logging preferences
prefs['logging.console_logging_level'] = 'INFO'

# Device preferences
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("Brian2 preferences configured:")
for key, value in prefs.items():
    print(f"  {key}: {value}")
EOF
    
    log_info "Preferences setup script created"
    return 0
}

################################################################################
# CODE VERIFICATION
################################################################################

verify_phase1_changes() {
    log_section "VERIFYING PHASE 1 CHANGES"
    
    local cuda_vector_file="${SCRIPT_DIR}/brian2cuda/brianlib/cudaVector.h"
    local spike_queue_file="${SCRIPT_DIR}/brian2cuda/brianlib/spikequeue.h"
    
    if [ ! -f "$cuda_vector_file" ]; then
        log_error "cudaVector.h not found at $cuda_vector_file"
        return 1
    fi
    
    if [ ! -f "$spike_queue_file" ]; then
        log_error "spikequeue.h not found at $spike_queue_file"
        return 1
    fi
    
    # Check for Phase 1 key changes
    log_info "Checking cudaVector.h for Phase 1 changes..."
    if grep -q "volatile size_type\* m_size" "$cuda_vector_file"; then
        log_info "✓ Found pointer m_size declaration"
    else
        log_error "✗ Phase 1 change not found: m_size pointer"
        return 1
    fi
    
    if grep -q "bool m_size_owned" "$cuda_vector_file"; then
        log_info "✓ Found m_size_owned flag"
    else
        log_error "✗ Phase 1 change not found: m_size_owned"
        return 1
    fi
    
    if grep -q "set_size_address" "$cuda_vector_file"; then
        log_info "✓ Found set_size_address method"
    else
        log_error "✗ Phase 1 change not found: set_size_address"
        return 1
    fi
    
    log_info "Checking spikequeue.h for Phase 1 changes..."
    if grep -q "volatile size_type\* queue_sizes" "$spike_queue_file"; then
        log_info "✓ Found queue_sizes member"
    else
        log_error "✗ Phase 1 change not found: queue_sizes"
        return 1
    fi
    
    return 0
}

verify_phase2_changes() {
    log_section "VERIFYING PHASE 2 CHANGES"
    
    local synapses_file="${SCRIPT_DIR}/brian2cuda/templates/synapses.cu"
    local common_group_file="${SCRIPT_DIR}/brian2cuda/templates/common_group.cu"
    
    if [ ! -f "$synapses_file" ]; then
        log_error "synapses.cu not found at $synapses_file"
        return 1
    fi
    
    log_info "Checking synapses.cu for Phase 2 changes..."
    if grep -q "queue->current_offset" "$synapses_file"; then
        log_info "✓ Found current_offset calculation"
    else
        log_error "✗ Phase 2 change not found: current_offset"
        return 1
    fi
    
    if grep -q "blocks_per_partition = 4" "$synapses_file"; then
        log_info "✓ Found blocks_per_partition assignment"
    else
        log_error "✗ Phase 2 change not found: blocks_per_partition"
        return 1
    fi
    
    if grep -q "hosts_queue_sizes" "$synapses_file" || grep -q "host_queue_sizes" "$synapses_file"; then
        log_info "✓ Found host_queue_sizes buffer"
    else
        log_error "✗ Phase 2 change not found: host_queue_sizes"
        return 1
    fi
    
    return 0
}

verify_phase3_changes() {
    log_section "VERIFYING PHASE 3 CHANGES"
    
    local synapses_file="${SCRIPT_DIR}/brian2cuda/templates/synapses.cu"
    
    log_info "Checking synapses.cu for Phase 3 changes..."
    if grep -q "partition = bid % num_parallel_blocks" "$synapses_file"; then
        log_info "✓ Found bid remapping (partition calculation)"
    else
        log_error "✗ Phase 3 change not found: bid remapping"
        return 1
    fi
    
    if grep -q "worker_id = bid / num_parallel_blocks" "$synapses_file"; then
        log_info "✓ Found worker_id calculation"
    else
        log_error "✗ Phase 3 change not found: worker_id"
        return 1
    fi
    
    if grep -q "worker_id.*num_workers" "$synapses_file"; then
        log_info "✓ Found grid-stride loop with worker_id"
    else
        log_error "✗ Phase 3 change not found: grid-stride loop"
        return 1
    fi
    
    return 0
}

verify_all_changes() {
    verify_phase1_changes || return 1
    verify_phase2_changes || return 1
    verify_phase3_changes || return 1
    log_info "✓ All Phases 1-3 changes verified successfully"
    return 0
}

################################################################################
# COMPILATION
################################################################################

compile_brian2cuda() {
    log_section "COMPILING BRIAN2CUDA"
    
    if [ "$NO_COMPILE" = true ]; then
        log_info "Skipping compilation (--no-compile flag set)"
        return 0
    fi
    
    log_info "Installing brian2cuda in development mode..."
    cd "$SCRIPT_DIR"
    run_cmd "python -m pip install -e ." || return 1
    
    log_info "✓ Brian2CUDA compiled successfully"
    return 0
}

################################################################################
# REGRESSION TESTING
################################################################################

run_regression_tests() {
    log_section "RUNNING REGRESSION TESTS"
    
    local test_dir="${SCRIPT_DIR}/brian2cuda/tests"
    local tools_test_suite_dir="${SCRIPT_DIR}/brian2cuda/tools/test_suite"
    
    if [ ! -d "$test_dir" ]; then
        log_error "Test directory not found: $test_dir"
        return 1
    fi
    
    log_info "Running tests from: $test_dir"
    cd "$SCRIPT_DIR"
    
    # Disable plugin autoload to avoid incompatible external pytest plugins.
    if command -v pytest &> /dev/null; then
        log_info "Running tests with pytest..."
        if [ -d "$tools_test_suite_dir" ]; then
            run_cmd "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v --tb=short -x \"$test_dir\" --ignore=\"$tools_test_suite_dir\"" || return 1
        else
            run_cmd "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v --tb=short -x \"$test_dir\"" || return 1
        fi
    elif python -c "import pytest" &> /dev/null; then
        log_info "Running tests with Python pytest module..."
        if [ -d "$tools_test_suite_dir" ]; then
            run_cmd "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -v --tb=short -x \"$test_dir\" --ignore=\"$tools_test_suite_dir\"" || return 1
        else
            run_cmd "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -v --tb=short -x \"$test_dir\"" || return 1
        fi
    else
        log_error "pytest is required for regression tests but is not installed"
        return 1
    fi

    log_info "✓ All regression tests passed"
    return 0
}

################################################################################
# BENCHMARK EXECUTION
################################################################################

create_benchmark_script() {
    local test_name="$1"
    local num_blocks="$2"
    local results_file="$3"
    
    log_info "Creating benchmark script for: $test_name (num_blocks=$num_blocks)"
    
    cat > "${RESULTS_DIR}/run_${test_name}.py" << EOF
#!/usr/bin/env python
"""
Brunel-Hakim benchmark for Phase 4 testing
Configuration: heterogeneous delays, multiple blocks
"""

import os
import sys
import time
import glob
import traceback

# Add examples to path
sys.path.insert(0, os.path.join('${SCRIPT_DIR}', 'examples'))

# Import utils
from utils import set_prefs, update_from_command_line

params = {
    'devicename': 'cuda_standalone',
    'heterog_delays': True,
    'narrow_delaydistr': True,
    'resultsfolder': 'results_${test_name}',
    'codefolder': 'code_${test_name}',
    'N': ${BRUNEL_N},
    'profiling': ${PROFILE_RUN},
    'monitors': False,  # Disable monitors for faster runs
    'single_precision': False,
    'num_blocks': ${num_blocks},
    'atomics': True,
    'bundle_mode': True
}

# Do imports after configuration
import matplotlib
matplotlib.use('Agg')

from brian2 import *
import brian2cuda

# Set preferences
name = set_prefs(params, prefs)
codefolder = os.path.join(params['codefolder'], name)

# Avoid dependency on CUDA Samples deviceQuery binary.
prefs['devices.cuda_standalone.cuda_backend.detect_gpus'] = False
prefs['devices.cuda_standalone.cuda_backend.gpu_id'] = 0
prefs['devices.cuda_standalone.cuda_backend.compute_capability'] = ${CUDA_COMPUTE_CAPABILITY}
prefs['logging.console_logging_level'] = 'WARNING'

print(f'Benchmark: {name}')
print(f'Code directory: {codefolder}')

# Set device
set_device(params['devicename'], directory=codefolder, compile=True, run=True,
           debug=False)

# Model parameters
Vr = 10*mV
theta = 20*mV
tau = 20*ms
delta = 2*ms
taurefr = 2*ms
duration = eval('${BRUNEL_DURATION}')
C = 1000
sparseness = float(C)/params['N']
J = .1*mV
sigmaext = 0.33*mV
muext = 27*mV

# Neuron equations
eqs = """
dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
"""

# Create neuron group
group = NeuronGroup(params['N'], eqs, threshold='V>theta',
                    reset='V=Vr', refractory=taurefr)
group.V = Vr

# Create synapses with heterogeneous delays
conn = Synapses(group, group, on_pre='V += -J')
conn.connect(p=sparseness)
conn.delay = "delta + 2 * dt * rand() - dt"

print(f'Number of neurons: {params["N"]}')
print('Number of synapses: unavailable before run in standalone mode')
print(f'Sparseness: {sparseness:.4f}')
print(f'Simulation duration: {duration}')

# Run simulation with timing
print('Starting simulation...')
start_time = time.time()
try:
    run(duration, report='text', profile=params['profiling'])
except Exception as exc:
    print('Run failed with exception:')
    traceback.print_exc()

    def _dump_latest_tmp(pattern, label):
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not files:
            print(f'No {label} files matched {pattern}')
            return
        latest = files[-1]
        print(f'Latest {label}: {latest}')
        try:
            with open(latest, 'r') as f:
                lines = f.readlines()
            print(f'--- BEGIN {label} TAIL ---')
            for line in lines[-200:]:
                print(line.rstrip())
            print(f'--- END {label} TAIL ---')
        except Exception as read_exc:
            print(f'Failed to read {latest}: {read_exc}')

    _dump_latest_tmp('/tmp/brian_stderr_*.log', 'brian_stderr')
    _dump_latest_tmp('/tmp/brian_stdout_*.log', 'brian_stdout')
    _dump_latest_tmp('/tmp/brian_debug_*.log', 'brian_debug')
    raise
end_time = time.time()

elapsed_time = end_time - start_time
print(f'Simulation completed in {elapsed_time:.2f} seconds')

# Save profiling information
if params['profiling']:
    print('\\nProfiling Summary:')
    print(profiling_summary())
    
    profilingpath = os.path.join(params['resultsfolder'], f'{name}_profiling.txt')
    os.makedirs(params['resultsfolder'], exist_ok=True)
    with open(profilingpath, 'w') as f:
        f.write(str(profiling_summary()))
    print(f'Profiling saved to: {profilingpath}')
    
    # Save timing info
    timing_file = os.path.join(params['resultsfolder'], f'{name}_timing.txt')
    with open(timing_file, 'w') as f:
        f.write(f'Total elapsed time: {elapsed_time:.2f} seconds\\n')
    print(f'Timing saved to: {timing_file}')

print('Benchmark completed successfully!')
EOF
    
    chmod +x "${RESULTS_DIR}/run_${test_name}.py"
    return 0
}

run_benchmark() {
    local test_name="$1"
    local num_blocks="$2"
    
    log_section "RUNNING BENCHMARK: $test_name (num_blocks=$num_blocks)"
    
    local results_file="${RESULTS_DIR}/${test_name}_results.txt"
    
    # Create benchmark script
    create_benchmark_script "$test_name" "$num_blocks" "$results_file" || return 1
    
    # Run benchmark from RESULTS_DIR so generated result/code folders are grouped
    cd "$RESULTS_DIR"
    log_info "Executing benchmark: run_${test_name}.py"
    
    python "./run_${test_name}.py"
    
    if [ $? -ne 0 ]; then
        log_error "Benchmark failed: $test_name"

        return 1
    fi
    
    log_info "✓ Benchmark completed: $test_name"
    return 0
}

################################################################################
# RESULTS ANALYSIS
################################################################################

analyze_profiling() {
    log_section "ANALYZING PROFILING DATA"
    
    cat > "${RESULTS_DIR}/analyze_results.py" << 'ANALYSIS_EOF'
#!/usr/bin/env python
"""
Analyze profiling results from Phase 4 benchmarks
"""

import os
import re
from pathlib import Path

def parse_profiling_output(filepath):
    """Parse Brian2 profiling output"""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    times = {}
    
    # Parse different operation times
    # Looking for patterns like "operation_name (XX ms)"
    patterns = {
        'effect': r'effect.*?([\d.]+)\s*ms',
        'propagation': r'push|propagat.*?([\d.]+)\s*ms',
        'neurons': r'neuron|state.*?([\d.]+)\s*ms',
        'total': r'Total.*?([\d.]+)\s*ms'
    }
    
    for name, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            times[name] = float(match.group(1))
    
    return times

def main():
    results_dir = Path('.')
    profiling_files = list(results_dir.glob('results_*/*_profiling.txt'))
    
    print("\n" + "="*80)
    print("PROFILING ANALYSIS SUMMARY")
    print("="*80)
    
    all_results = {}
    for pfile in profiling_files:
        test_name = pfile.parent.name
        times = parse_profiling_output(str(pfile))
        if times:
            all_results[test_name] = times
            print(f"\n{test_name}:")
            for op, time_ms in times.items():
                print(f"  {op:20s}: {time_ms:10.2f} ms")
    
    # Compare results if multiple benchmarks
    if len(all_results) >= 2:
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)

        baseline_name = 'results_baseline' if 'results_baseline' in all_results else None
        optimized_name = 'results_optimized' if 'results_optimized' in all_results else None

        if baseline_name and optimized_name:
            baseline = all_results[baseline_name]
            after = all_results[optimized_name]
            print(f"\nComparison: {optimized_name} vs {baseline_name}")
            for key in ['effect', 'propagation', 'neurons', 'total']:
                if key in baseline and key in after:
                    baseline_time = baseline[key]
                    after_time = after[key]
                    improvement = ((baseline_time - after_time) / baseline_time) * 100
                    print(f"  {key:20s}: {baseline_time:10.2f} -> {after_time:10.2f} ms ({improvement:+.1f}%)")
        else:
            test_names = sorted(all_results.keys())
            baseline = all_results[test_names[0]]
            after = all_results[test_names[1]]
            print(f"\nComparison: {test_names[1]} vs {test_names[0]}")
            for key in ['effect', 'propagation', 'neurons', 'total']:
                if key in baseline and key in after:
                    baseline_time = baseline[key]
                    after_time = after[key]
                    improvement = ((baseline_time - after_time) / baseline_time) * 100
                    print(f"  {key:20s}: {baseline_time:10.2f} -> {after_time:10.2f} ms ({improvement:+.1f}%)")

if __name__ == '__main__':
    main()
ANALYSIS_EOF
    
    python "${RESULTS_DIR}/analyze_results.py"
    return 0
}

################################################################################
# MAIN EXECUTION
################################################################################

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-compile)
                NO_COMPILE=true
                shift
                ;;
            --help|-h)
                print_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_help
                exit 1
                ;;
        esac
    done
}

print_help() {
    cat << EOF
Phase 4 Testing & Performance Verification Script for Issue #269

Usage: ./bench.sh [OPTIONS]

OPTIONS:
  --no-compile      Skip compilation, use existing build
  --help, -h        Show this help message

EXAMPLES:
    # Full test run
  ./bench.sh

  # Run tests without recompilation
  ./bench.sh --no-compile

EOF
}

main() {
    local exit_code=0
    
    log_info "============================================"
    log_info "Phase 4: Testing & Verification"
    log_info "Issue #269: Multi-block Heterogeneous Delays"
    log_info "============================================"
    
    # Parse arguments
    parse_arguments "$@"
    
    # Setup environment
    if ! setup_environment; then
        log_error "Environment setup failed"
        exit 1
    fi
    
    # Install dependencies
    if ! install_dependencies; then
        log_error "Dependency installation failed"
        exit 1
    fi
    
    # Configure preferences
    configure_brian2_preferences || return 1
    
    # Verify code changes
    if ! verify_all_changes; then
        log_error "Code verification failed"
        exit 1
    fi
    
    # Compile
    if ! compile_brian2cuda; then
        log_error "Compilation failed"
        exit 1
    fi
    
    # Run regression tests
    if ! run_regression_tests; then
        log_error "Regression tests failed"
        exit_code=1
    fi
    
    # Run benchmarks
    log_section "RUNNING BENCHMARKS"
    
    # Baseline (will be just single block implementation on current system)
    if ! run_benchmark "baseline" "1"; then
        log_error "Baseline benchmark failed"
        exit_code=1
    fi
    
    # Optimized (multi-block with dynamic sizing)
    if ! run_benchmark "optimized" "None"; then
        log_error "Optimized benchmark failed"
        exit_code=1
    fi
    
    # Analyze results
    if ! analyze_profiling; then
        log_error "Result analysis failed"
        exit_code=1
    fi
    
    log_section "PHASE 4 TESTING COMPLETE"
    
    if [ $exit_code -eq 0 ]; then
        log_info "✓ All tests completed successfully!"
        log_info "Results saved to: $RESULTS_DIR"
    else
        log_error "Some tests failed. See terminal output above for details."
    fi
    
    return $exit_code
}

# Run main function
main "$@"
exit_code=$?

echo ""
echo "Results directory: $RESULTS_DIR"
echo ""

exit $exit_code
