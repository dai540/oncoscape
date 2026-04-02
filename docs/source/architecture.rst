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
- ``training``: deep multitask model fitting and classical fallbacks
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

The repository now uses a deep spatial multitask model as the default execution path for HPC builds.

The default path includes:

- image encoding with a configurable encoder
- spatial message passing across a slide graph
- multitask prediction for compartment, composition, and programs
- validation-based checkpoint selection
- spatial smoothing during evaluation and rendering

An enhanced classical fallback still exists for CPU-only debugging and minimal environments, but the public interfaces are stable so the deep path and fallback share the same outer pipeline contract.
