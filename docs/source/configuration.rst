Configuration
=============

Primary Files
-------------

``configs/breast_hpc.yaml``
   Main HPC pipeline configuration.

``configs/breast_sources.template.yaml``
   Source manifest template.

``configs/breast_downloads.template.yaml``
   Public download catalog template.

Important Sections in ``breast_hpc.yaml``
-----------------------------------------

``paths``
   Defines ``project_root``, ``code_root``, ``data_root``, and ``run_root``.

``fetch``
   Controls the optional public-data download catalog.

``registration``
   Controls source manifest ingestion and ``slides.csv`` generation.

``reference``
   Controls reference atlas input, latent dimensionality, and marker export.

``teachers``
   Controls compartment, composition, and program teacher generation behavior.

``patch_extraction``
   Defines 112 um tile extraction and patch QC thresholds.

``training``
   Defines the default ``deep_spatial_multitask`` model, encoder, spatial stack, and smoothing radius.

``evaluation``
   Controls the target split and report metrics.

``render``
   Controls prediction output paths and render tile size.

``reporting``
   Controls the executive, wet-lab, and developer report paths.

Reproducibility
---------------

Each full run writes:

- ``resolved_config.yaml``
- ``provenance.json`` with git commit and config hash

These files should be archived with every model build.

Recommended Validation
----------------------

Before launching a large run, execute the regression tests:

.. code-block:: bash

   python -m unittest discover -s tests -v

The deep pipeline test exercises the full register-to-evaluate path on a synthetic dataset and enforces held-out accuracy thresholds.
