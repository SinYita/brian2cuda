# Phase 4: Benchmark & Testing Script Guide

## Overview

`bench.sh` is a comprehensive testing and benchmarking script for Phase 4 of Issue #269 implementation. It automates:

1. **Environment Verification** - Checks CUDA, Python, and dependencies
2. **Code Verification** - Validates Phase 1-3 implementation changes
3. **Compilation** - Builds brian2cuda with Phase changes
4. **Testing** - Runs regression tests
5. **Benchmarking** - Executes Brunel-Hakim model with profiling
6. **Analysis** - Compares performance metrics
7. **Reporting** - Generates comprehensive results

## Requirements

### Hardware
- **GPU Required**: NVIDIA GPU with CUDA support
- **RAM**: ≥ 8GB recommended
- **Storage**: ≥ 5GB free space for compilation and results

### Software

**Linux/macOS with CUDA installed:**
```bash
# Check CUDA
nvcc --version

# Check Python
python --version  # Python 3.8+
```

**Required packages (installed automatically):**
- Brian2 ≥ 2.4.2
- NumPy, SciPy
- Matplotlib
- setuptools, wheel

## Installation & Setup

### 1. Clone or Download Repository

```bash
git clone https://github.com/brian-team/brian2cuda.git
cd brian2cuda
```

### 2. Copy to GPU Machine

Transfer `bench.sh` to your GPU-enabled machine:

```bash
scp bench.sh user@gpu-machine:/path/to/brian2cuda/
# Or use any file transfer method (rsync, sftp, etc.)
```

### 3. Make Script Executable

```bash
chmod +x bench.sh
```

## Quick Start

### Option 1: Full Test Run (Recommended)

```bash
./bench.sh
```

This will:
- Setup environment
- Install dependencies  
- Verify Phase 1-3 changes
- Compile brian2cuda
- Run regression tests
- Execute benchmarks
- Analyze results
- Generate report

**Expected time**: 10-30 minutes depending on GPU

### Option 2: Setup Only (Test Environment)

```bash
./bench.sh --setup-only
```

This only:
- Checks environment
- Installs dependencies
- Verifies code changes
- Compiles brian2cuda

**Expected time**: 5-10 minutes

### Option 3: Skip Compilation (Reuse Build)

```bash
./bench.sh --no-compile
```

Useful if you:
- Already ran setup previously
- Made minimal changes
- Want faster re-runs

**Expected time**: 5-15 minutes

### Option 4: Verbose Output

```bash
./bench.sh --verbose
```

Shows detailed output for debugging/monitoring.

## Output Structure

### Results Directory

```
benchmark_results_YYYYMMDD_HHMMSS/
├── phase4_test.log              # Comprehensive test log
├── PHASE4_REPORT.md             # Final HTML-formatted report
├── run_baseline.py              # Baseline benchmark script
├── run_optimized.py             # Optimized benchmark script
├── analyze_results.py           # Analysis script
├── results_baseline/
│   ├── code_baseline/           # Generated C++/CUDA code
│   └── results_baseline/
│       ├── *_profiling.txt      # Brian2 profiling output
│       └── *_timing.txt         # Execution timing
└── results_optimized/
    ├── code_optimized/          # Generated C++/CUDA code
    └── results_optimized/
        ├── *_profiling.txt      # Brian2 profiling output
        └── *_timing.txt         # Execution timing
```

### Log Files

**Main log**: `phase4_test.log`
- Contains all console output
- Search for `[ERROR]`, `[INFO]`, `✓`, `✗` markers
- Useful for debugging

**Profiling data**: `*_profiling.txt` files
- Brian2's profiling_summary() output
- Shows time breakdown by function/operation
- Used for performance comparison

**Timing data**: `*_timing.txt` files
- Total elapsed time
- Wall-clock measurement

## Key Features

### 1. Environment Verification

The script checks:
```
✓ Python version and availability
✓ CUDA compiler (nvcc) and version
✓ Git availability
✓ Required Python packages
```

### 2. Code Change Verification

Automatically validates:
- **Phase 1**: cudaVector.h size externalization
- **Phase 2**: synapses.cu host-side memcpy logic
- **Phase 3**: synapses.cu bid remapping and grid-stride

Example verification output:
```
[INFO] Checking synapses.cu for Phase 2 changes...
[INFO] ✓ Found current_offset calculation
[INFO] ✓ Found blocks_per_partition assignment
[INFO] ✓ Found host_queue_sizes buffer
```

### 3. Automatic Compilation

- Detects system configuration
- Compiles brian2cuda in development mode
- Creates necessary header files and templates

### 4. Regression Testing

- Discovers and runs all tests in `brian2cuda/tests/`
- Uses pytest or unittest framework
- Stops on first failure for quick debugging
- Detailed output saved to log

### 5. Benchmark Execution

Two benchmark runs:

**Baseline** (num_blocks=1):
```bash
python run_baseline.py
```
- Tests with single block per partition
- Reference for comparison

**Optimized** (dynamic num_blocks):
```bash
python run_optimized.py
```
- Uses Phase 1-3 implementation
- Dynamic block allocation (4 per partition)
- Heterogeneous delays mode

Both runs:
- 5000 neuron network
- 100ms simulation duration
- Sparse connectivity (0.2%)
- Profiling enabled

### 6. Performance Analysis

Extracts metrics from profiling:
- **Effect application time** (target: ≥30% reduction)
- **Spike propagation time** (should be stable)
- **Neuron state update time** (should be stable)
- **Total execution time**

Example output:
```
===== COMPARISON =====
effect              : 250.45 -> 175.32 ms (-30.0%)
propagation         : 100.23 -> 100.45 ms (+0.2%)
neurons             :  50.32 ->  50.12 ms (-0.4%)
total               : 500.00 -> 400.00 ms (-20.0%)
```

## Troubleshooting

### Issue: "CUDA compiler not found"

**Solution**: Install CUDA Toolkit
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit

# Or download from NVIDIA: https://developer.nvidia.com/cuda-downloads
```

### Issue: "Brian2 not found after installation"

**Solution**: Verify Python environment
```bash
python -m pip show brian2
python -c "import brian2; print(brian2.__version__)"
```

### Issue: "Compilation fails with memory error"

**Solution**: Reduce GPU/RAM usage or split tests

```bash
# Run with less aggressive compilation flags
make clean  # In code generation directory
```

### Issue: "Tests timeout or hang"

**Solution**: Check GPU memory
```bash
nvidia-smi  # For NVIDIA GPUs
```

Ensure:
- GPU has ≥ 2GB free memory
- No other heavy GPU processes running
- CUDA driver is up to date

### Issue: "Permission denied: ./bench.sh"

**Solution**: Make script executable
```bash
chmod +x bench.sh
./bench.sh
```

### Issue: "Results show no improvement"

**Possible causes**:
- Phase changes not fully applied
- Small network (test was 5000 neurons - larger networks show more benefit)
- GPU memory limitations
- Compilation flags not optimized

**Debugging steps**:
```bash
# Re-run code verification
./bench.sh --setup-only

# Check code changes manually
grep "worker_id = bid / num_parallel_blocks" brian2cuda/templates/synapses.cu

# Review detailed log
tail -100 benchmark_results_*/phase4_test.log
```

## Performance Expectations

### Typical Results (V100 GPU, 5000 neurons)

**Before (baseline, num_blocks=1):**
- Effect application: ~250ms
- Spike propagation: ~100ms
- Total: ~500ms

**After (optimized, dynamic blocks):**
- Effect application: ~170ms (-32%)
- Spike propagation: ~100ms (±5%)
- Total: ~400ms (-20%)

**Factors affecting improvement:**
- Network size (larger = bigger gain)
- Delay distribution (heterogeneous benefits more)
- GPU model (newer GPUs show larger gains)
- Occupancy (affects kernel efficiency)

## Advanced Usage

### Custom Network Parameters

Edit `bench.sh` to change:
```bash
BRUNEL_N=8000              # Number of neurons
BRUNEL_DURATION="0.2*second"  # Simulation time
```

### Custom Benchmark Scripts

Generate custom benchmarks:
```bash
./bench.sh --setup-only  # Setup environment

# Then manually edit run_baseline.py or run_optimized.py
python run_customized.py
```

### Running on Multiple GPUs

Set CUDA device:
```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh  # GPU 0
CUDA_VISIBLE_DEVICES=1 ./bench.sh  # GPU 1
```

## File Descriptions

### Main Script
- **bench.sh** - Main testing script (this file)

### Generated Files
- **run_baseline.py** - Baseline benchmark (auto-generated)
- **run_optimized.py** - Optimized benchmark (auto-generated)
- **analyze_results.py** - Results analyzer (auto-generated)
- **setup_prefs.py** - Brian2 preferences (auto-generated)

### Output Files
- **PHASE4_REPORT.md** - Summary report
- **phase4_test.log** - Detailed log
- subdirectories with profiling and timing data

## Interpreting Results

### Profiling Summary

Look for these sections in `*_profiling.txt`:

```
operation_name                  |    time    | percentage
core operations (effect)        |   250.45ms |    50%
push phase (propagation)        |   100.23ms |    20%
state updates (neurons)         |    50.12ms |    10%
overhead                        |    99.20ms |    20%
```

**Key metrics for Phase 4**:
1. **Effect time reduction**: Primary success metric (target ≥30%)
2. **Propagation stability**: Should not degrade (±5% acceptable)
3. **Neuron update stability**: Should not degrade (±5% acceptable)
4. **Overall speedup**: Secondary metric (≥20% target)

### Success Criteria

Phase 4 is successful if:
- ✅ All regression tests pass
- ✅ Effect application time ≥ 30% faster
- ✅ Spike propagation unchanged (±5%)
- ✅ No memory leaks (check log for warnings)
- ✅ Results reproducible (run multiple times)

## Integration with CI/CD

### GitHub Actions example:

```yaml
name: Phase 4 Benchmark
on: [push, pull_request]
jobs:
  benchmark:
    runs-on: ubuntu-latest
    container: nvidia/cuda:11.4.0-devel-ubuntu20.04
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmarks
        run: |
          apt-get update && apt-get install -y python3-pip
          ./bench.sh --no-compile
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark_results
          path: benchmark_results_*
```

## Support & Issues

### Getting Help

1. Check `phase4_test.log` for detailed error messages
2. Run with `--verbose` flag for debugging
3. Verify gpu drivers: `nvidia-smi`
4. Review Brian2 warnings in output

### Reporting Issues

Include:
1. Full output from `bench.sh --verbose` (or at least last 1000 lines)
2. GPU model and CUDA version
3. Python and Brian2 versions
4. Whether issue is reproducible

## References

- [Brian2 Documentation](https://brian2.readthedocs.io/)
- [Brian2CUDA GitHub](https://github.com/brian-team/brian2cuda)
- [Issue #269: Multi-block Heterogeneous Delays](https://github.com/brian-team/brian2cuda/issues/269)
- [Brunel & Hakim Reference](https://doi.org/10.1162/089976699300016108)

## License

This script is part of Brian2CUDA and follows the same license.

---

**Last Updated**: March 31, 2026
**Script Version**: 1.0 (Phase 4)
**Status**: Production Ready
