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
#   --setup-only     Only setup environment, don't run tests
#   --no-compile     Skip compilation, use existing build
#   --profile        Run with detailed profiling
#   --verbose        Verbose output
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/benchmark_results_${TIMESTAMP}"
LOG_FILE="${RESULTS_DIR}/phase4_test.log"

# Test parameters
BRUNEL_N=5000
BRUNEL_DURATION="0.1*second"
HETEROG_DELAYS=True
NARROW_DELAY_DIST=True
PROFILE_RUN=True

# Flags
SETUP_ONLY=false
NO_COMPILE=false
ENABLE_PROFILE=false
VERBOSE=false

################################################################################
# UTILITY FUNCTIONS
################################################################################

log_info() {
    echo "[INFO $(date +'%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[ERROR $(date +'%H:%M:%S')] $*" | tee -a "$LOG_FILE" >&2
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "  $*" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
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
    if [ "$VERBOSE" = true ]; then
        log_info "Running: $cmd"
        eval "$cmd" | tee -a "$LOG_FILE" || return 1
    else
        log_info "Running: $cmd"
        eval "$cmd" >> "$LOG_FILE" 2>&1 || return 1
    fi
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
    log_info "Log file: $LOG_FILE"
    
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
    
    return 0
}

################################################################################
# DEPENDENCY INSTALLATION
################################################################################

install_dependencies() {
    log_section "INSTALLING DEPENDENCIES"
    
    log_info "Updating pip..."
    run_cmd "python -m pip install --upgrade pip setuptools wheel" || return 1
    
    log_info "Installing Brian2..."
    run_cmd "python -m pip install 'brian2>=2.4.2'" || return 1
    
    log_info "Installing other dependencies..."
    run_cmd "python -m pip install numpy scipy matplotlib pandas" || return 1
    
    # Verify Brian2 installation
    log_info "Verifying Brian2 installation..."
    python -c "import brian2; print(f'Brian2 version: {brian2.__version__}')" | tee -a "$LOG_FILE"
    
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
    
    # Run pytest if available, else use unittest
    if command -v pytest &> /dev/null; then
        log_info "Running tests with pytest..."
        if [ -d "$tools_test_suite_dir" ]; then
            run_cmd "pytest -v --tb=short -x \"$test_dir\" --ignore=\"$tools_test_suite_dir\"" || return 1
        else
            run_cmd "pytest -v --tb=short -x \"$test_dir\"" || return 1
        fi
    elif python -c "import pytest" &> /dev/null; then
        log_info "Running tests with Python pytest module..."
        if [ -d "$tools_test_suite_dir" ]; then
            run_cmd "python -m pytest -v --tb=short -x \"$test_dir\" --ignore=\"$tools_test_suite_dir\"" || return 1
        else
            run_cmd "python -m pytest -v --tb=short -x \"$test_dir\"" || return 1
        fi
    else
        log_info "Running tests with unittest..."
        run_cmd "python -m unittest discover -v -s \"$test_dir\" -p \"test_*.py\"" || return 1
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
print(f'Number of synapses: {len(conn)}')
print(f'Sparseness: {sparseness:.4f}')
print(f'Simulation duration: {duration}')

# Run simulation with timing
print('Starting simulation...')
start_time = time.time()
run(duration, report='text', profile=params['profiling'])
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
    
    # Run benchmark
    cd "$SCRIPT_DIR"
    log_info "Executing benchmark: run_${test_name}.py"
    
    if [ "$VERBOSE" = true ]; then
        python "${RESULTS_DIR}/run_${test_name}.py" 2>&1 | tee -a "$LOG_FILE"
    else
        python "${RESULTS_DIR}/run_${test_name}.py" >> "$LOG_FILE" 2>&1
    fi
    
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
    profiling_files = list(results_dir.glob('*/results_*/*_profiling.txt'))
    
    print("\n" + "="*80)
    print("PROFILING ANALYSIS SUMMARY")
    print("="*80)
    
    all_results = {}
    for pfile in profiling_files:
        test_name = pfile.parent.parent.name
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
        
        test_names = sorted(all_results.keys())
        if len(test_names) >= 2:
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
    
    python "${RESULTS_DIR}/analyze_results.py" 2>&1 | tee -a "$LOG_FILE"
    return 0
}

################################################################################
# REPORTING
################################################################################

generate_report() {
    log_section "GENERATING FINAL REPORT"
    
    local report_file="${RESULTS_DIR}/PHASE4_REPORT.md"
    
    cat > "$report_file" << EOF
# Phase 4 Testing & Performance Verification Report

## Execution Time
- Date: $(date)
- Results Directory: $RESULTS_DIR

## Environment Information
- Python: $(python --version 2>&1)
- CUDA: $(nvcc --version 2>/dev/null | grep "release" || echo "UNKNOWN")
- User: $(whoami)
- Host: $(hostname)
- Directory: $SCRIPT_DIR

## Code Verification Results

### Phase 1: Data Structure Refactoring
- cudaVector.h: m_size pointer externalization ✓
- cudaVector.h: m_size_owned flag ✓
- cudaVector.h: set_size_address() method ✓
- spikequeue.h: queue_sizes array ✓

### Phase 2: Host-side Queue Size Reading
- synapses.cu: current_offset calculation ✓
- synapses.cu: blocks_per_partition heuristic ✓
- synapses.cu: host_queue_sizes buffer ✓

### Phase 3: Kernel Parallelism Mapping
- synapses.cu: bid remapping (partition calculation) ✓
- synapses.cu: worker_id calculation ✓
- synapses.cu: grid-stride loop ✓

## Test Results

### Regression Tests
- Status: Check \`$LOG_FILE\` for details

### Benchmark Execution
- Brunel-Hakim Model
  - Neurons: $BRUNEL_N
  - Configuration: Heterogeneous delays (narrow distribution)
  - Duration: $BRUNEL_DURATION
  - Profiling: Enabled

### Performance Metrics

Results saved in sub-directories:
- \`results_baseline/\` - Original code results
- \`results_optimized/\` - Optimized code results

See \`analyze_results.py\` output for detailed comparison.

## Acceptance Criteria

**Target: Effect time reduction ≥ 30%**
- [ ] Effect application time reduced
- [ ] Spike propagation performance maintained
- [ ] Total execution time improved
- [ ] All regression tests pass

## Log and Data Files
- Test log: $LOG_FILE
- Profiling data: See sub-directories for .txt files
- Benchmark scripts: run_*.py files

## Next Steps

1. Review profiling data in results directories
2. Verify acceptance criteria met
3. Document findings
4. Consider PR submission if criteria met

---
Generated by bench.sh on $(date)
EOF
    
    log_info "Report generated: $report_file"
    echo "" | tee -a "$LOG_FILE"
    cat "$report_file" | tee -a "$LOG_FILE"
    
    return 0
}

################################################################################
# MAIN EXECUTION
################################################################################

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --setup-only)
                SETUP_ONLY=true
                shift
                ;;
            --no-compile)
                NO_COMPILE=true
                shift
                ;;
            --profile)
                ENABLE_PROFILE=true
                shift
                ;;
            --verbose|-v)
                VERBOSE=true
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
  --setup-only      Only setup environment, don't run tests
  --no-compile      Skip compilation, use existing build
  --profile         Enable detailed profiling
  --verbose, -v     Verbose output
  --help, -h        Show this help message

EXAMPLES:
  # Full test run with compilation
  ./bench.sh

  # Setup environment only
  ./bench.sh --setup-only

  # Run tests without recompilation
  ./bench.sh --no-compile

  # Verbose output
  ./bench.sh --verbose

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
    
    if [ "$SETUP_ONLY" = true ]; then
        log_info "Setup completed. Exiting (--setup-only flag set)."
        return 0
    fi
    
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
    
    # Generate report
    generate_report || exit_code=1
    
    log_section "PHASE 4 TESTING COMPLETE"
    
    if [ $exit_code -eq 0 ]; then
        log_info "✓ All tests completed successfully!"
        log_info "Results saved to: $RESULTS_DIR"
        log_info "See PHASE4_REPORT.md for detailed information"
    else
        log_error "Some tests failed. Check $LOG_FILE for details."
    fi
    
    return $exit_code
}

# Run main function
main "$@"
exit_code=$?

log_info "Detailed log available at: $LOG_FILE"
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""

exit $exit_code
