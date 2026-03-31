# brian2cuda Repository Overview

## 1. Project Positioning

brian2cuda is a CUDA standalone backend for Brian2. It lets users keep the Brian2 modeling API in Python, then generates C++/CUDA source code and runs simulations on NVIDIA GPUs.

Typical usage in user scripts:

```python
from brian2 import *
import brian2cuda
set_device("cuda_standalone")
```

At a high level, this repository does three things:

- Extends Brian2 code generation with CUDA-specific code objects and generators.
- Implements a standalone device that writes, compiles, and executes generated CUDA/C++ code.
- Provides templates, utilities, and tests to keep correctness and performance under control.

## 2. Top-Level Repository Structure

- `brian2cuda/`
  - Main Python package with code generation, device implementation, templates, utilities, tests, and developer tools.
- `docs_sphinx/`
  - Sphinx docs source.
- `examples/`
  - Runnable Brian2 examples adapted for CUDA standalone.
- `frozen_repos/`
  - Frozen/upstream references and diffs (Brian2, brian2genn, GeNN integration helpers).
- `dev/`
  - Historical and developer material (benchmarks, issue investigations, profiling, docs drafts).
- `README.md`
  - Installation and usage notes, including required Brian2 version alignment.
- `setup.py`
  - Packaging metadata and template/header inclusion.

## 3. Core Package Layout (`brian2cuda/`)

### 3.1 Entry and Registration

- `__init__.py`
  - Imports preferences and registers CUDA standalone components.
  - Activates debug logging by default (currently marked TODO for release cleanup).

### 3.2 Device Layer (Build/Run Orchestration)

- `device.py`
  - Defines `CUDAStandaloneDevice`, derived from Brian2 `CPPStandaloneDevice`.
  - Central orchestrator that:
    - Maps Brian2 arrays to host/device names.
    - Chooses code object class (standard or atomics path).
    - Builds code object source and main program source.
    - Manages RNG preparation paths.
    - Selects GPU and compute capability.
    - Emits `main.cu`, objects files, and compile configuration.
  - Includes `CUDAWriter` for writing generated `.cu/.h/.cpp` files and tracking source/header outputs.

### 3.3 Code Object Layer

- `codeobject.py`
  - Defines two standalone code objects:
    - `CUDAStandaloneCodeObject`: default CUDA code object.
    - `CUDAStandaloneAtomicsCodeObject`: atomics-enabled variant for race-safe parallel effect application.
  - Binds templating (`Templater`) and generator classes.

### 3.4 CUDA Code Generator

- `cuda_generator.py`
  - Implements CUDA-specific statement/kernel code generation.
  - Handles CUDA atomics support code generation based on runtime/GPU capability assumptions.
  - Integrates with Brian2 function and variable translation pipeline.

### 3.5 Preferences and Runtime Tuning

- `cuda_prefs.py`
  - Registers user preferences under `devices.cuda_standalone` and `devices.cuda_standalone.cuda_backend`.
  - Important preference groups:
    - Kernel launch behavior (`SM_multiplier`, `launch_bounds`, occupancy).
    - Synapse propagation modes (bundle pushing, threads per bundle expression).
    - RNG backend behavior (curand generator type/ordering).
    - CUDA backend discovery (`detect_cuda`, `cuda_path`, runtime version).
    - GPU selection (`detect_gpus`, `gpu_id`, compute capability override).
    - Compilation flags (`extra_compile_args_nvcc`).

### 3.6 Brian2 Function Implementations for CUDA

- `binomial.py`
  - Adds CUDA implementation for Brian2 `BinomialFunction`.
  - Supports both normal approximation and inversion sampling paths.
  - Uses host/device random APIs depending on execution context.

- `timedarray.py`
  - Adds CUDA implementation for `TimedArray` (1D and 2D variants).
  - Generates inline host/device functions for indexed time lookup.

### 3.7 Utility Layer

- `utils/gputools.py`
  - CUDA install detection (`CUDA_PATH`, `nvcc`, defaults).
  - Runtime version detection via `nvcc --version`.
  - GPU discovery/selection and compute capability handling.
  - Caches detection results globally to avoid repeated external command calls.

- `utils/stringtools.py`
  - Regex-based literal rewriting, especially float literal suffix handling for generated code.

- `utils/logger.py`
  - Logging hierarchy suppression helpers and standard issue-report message.

### 3.8 Code Templates and C++/CUDA Support Library

- `templates/`
  - Jinja-style CUDA/C++ templates for generated components.
  - Includes main entry (`main.cu`), object/network/run files, and kernel templates such as:
    - `stateupdate.cu`
    - `synapses.cu`
    - monitor and threshold-related kernels.
  - Also includes platform-specific makefiles.

- `brianlib/`
  - CUDA/C++ support headers for runtime internals (vector helpers, queue logic, math helpers, clock and utility support).

### 3.9 Sphinx Extension Utilities

- `sphinxext/`
  - Custom doc tooling for API scraping/reference generation integrated into docs build flow.

## 4. End-to-End Implementation Flow

From model script to GPU execution, the pipeline is:

1. User defines Brian2 network in Python and sets device to CUDA standalone.
2. Brian2/brian2cuda translates abstract equations/statements into code objects.
3. `CUDAStandaloneDevice` prepares template keyword context and policy decisions, for example:
   - whether synaptic effects can use atomics,
   - whether pre/post indices can be dropped from device memory,
   - RNG strategy (device API vs pre-generated host API buffers).
4. `cuda_generator.py` renders CUDA-compatible scalar/vector code and helper support code.
5. Templates in `templates/` are rendered into concrete `.cu/.h` sources.
6. Main source initializes CUDA runtime:
   - selects GPU,
   - sets heap limits,
   - initializes buffers and code object run procedures.
7. Generated project is compiled with NVCC/toolchain and executed as standalone binary.

## 5. Notable Design Choices

- Atomics vs non-atomics synaptic updates
  - Repository provides two code object paths to balance correctness and parallelism.

- Hybrid RNG strategy
  - Device-side curand state usage for per-tick generation.
  - Host API pre-generation for selected one-shot execution contexts.

- Memory pressure controls
  - Optional dropping of synaptic pre/post reference arrays if analysis proves unnecessary.

- Preference-heavy configuration
  - Many low-level CUDA behaviors are exposed as Brian preferences to tune per model/hardware.

## 6. Testing and Validation

Test suite is package-local under `brian2cuda/tests/`.

- `conftest.py`
  - Reuses Brian2 test config and adds CUDA marker registration.

- Representative test files:
  - `test_cuda_standalone.py`: baseline end-to-end standalone simulation behavior.
  - `test_cuda_generator.py`: generation-level correctness.
  - `test_gpu_detection.py`: CUDA/GPU detection logic.
  - `test_random_number_generation.py`: RNG behavior consistency.
  - `test_synaptic_propagations.py`: synapse propagation correctness.
  - plus monitor/network/stateupdater/string utility tests.

- `tests/features/`
  - Feature-focused scripts (`cuda_configuration.py`, `speed.py`).

## 7. Documentation and Tooling

- `docs_sphinx/`
  - Sphinx config auto-generates API reference from package code.
  - Current docs indicate active construction state.

- `brian2cuda/tools/`
  - Scripts for local and cluster test-suite/benchmark orchestration.
  - Includes workflows for running tests on current state or synced copies.

## 8. Dependency and Version Coupling

- `setup.py` depends on `brian2>=2.4.2`.
- `README.md` explains that a specific Brian2 state is expected and provides `frozen_repos/` instructions and patching notes.
- This indicates tight coupling to Brian2 internal interfaces (device/codegen APIs), so upgrading Brian2 often requires compatibility checks.

## 9. Suggested Reading Order for New Contributors

1. `README.md` (install + usage contract)
2. `brian2cuda/__init__.py` and `cuda_prefs.py` (registration and knobs)
3. `brian2cuda/codeobject.py` and `cuda_generator.py` (generation core)
4. `brian2cuda/device.py` (pipeline orchestration)
5. `brian2cuda/templates/main.cu` and `templates/synapses.cu` (runtime entry + hot path)
6. `brian2cuda/tests/test_cuda_standalone.py` and related tests (behavior expectations)

This order gives a fast path from conceptual architecture to implementation details and then to executable behavior guarantees.
