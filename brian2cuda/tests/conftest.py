'''
Module containing fixtures and hooks used by the pytest test suite.
'''
# Use brian2's pytest configuration for the brian2cuda tests (see PR #232 for details)
import brian2.conftest as brian2_conftest
from brian2.conftest import *

# Add a cuda implementation for the fake_randn_randn_fixture,
# used in test_stateupdaters.py
fake_randn.implementations.add_implementation(
    'cuda',
    '''
    __host__ __device__ double randn(int vectorisation_idx)
    {
        return 0.5;
    }
    '''
)


# Register `cuda_standalone` marker
def pytest_configure(config):
    if hasattr(brian2_conftest, 'pytest_configure'):
        brian2_conftest.pytest_configure(config)
    if not hasattr(config, 'fail_for_not_implemented'):
        config.fail_for_not_implemented = False
    config.addinivalue_line(
        "markers", "cuda_standalone: to be used with standalone_only marker"
    )


def pytest_collection_modifyitems(config, items):
    """
    Avoid accidentally running Brian2's full test runner.

    Many brian2cuda tests use `from brian2 import *`. In older Brian2 versions,
    this can import a callable named `test` (aliasing brian2.tests.run). Pytest
    then collects it as a test function and starts running Brian2's entire test
    suite (including doctests requiring sphinx/docutils), which is not intended
    here and breaks the brian2cuda test run.
    """
    kept = []
    for item in items:
        func = getattr(item, "function", None)
        if func is not None:
            mod = getattr(func, "__module__", "")
            name = getattr(func, "__name__", "")
            if mod.startswith("brian2.tests") and name == "run":
                continue
        kept.append(item)
    items[:] = kept
