Architecture
============

Package Layout
--------------

``src/oncoscape/`` is organized into focused subpackages:

- ``config``: YAML config loading
- ``core``: shared filesystem, serialization, tabular I/O, provenance
- ``data``: manifest parsing, registry building, split generation, public-data fetch
- ``reference``: scRNA reference atlas construction
- ``labels``: teacher-label construction
- ``preprocessing``: patch extraction and graph creation
- ``training``: baseline model fitting
- ``evaluation``: prediction and rendering
- ``reporting``: executive / wet-lab / developer reports
- ``pipeline``: orchestration across all stages
- ``validation``: preflight checks

Execution Model
---------------

The documentation and code now align around two execution patterns:

- stage-by-stage scripts for debugging
- a single orchestrated pipeline entry point for cluster use

Current Modeling Strategy
-------------------------

The repository currently uses an enhanced classical baseline rather than a heavy deep-learning stack as the default execution path.

This baseline includes:

- richer handcrafted tile features
- validation-driven model selection
- spatial smoothing during prediction
- deterministic end-to-end execution on CPU-friendly environments

The public interfaces are intentionally stable so stronger backends can replace the baseline without changing the outer pipeline contract.
