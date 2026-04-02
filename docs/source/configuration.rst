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
   Defines the current enhanced classical baseline settings.

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
