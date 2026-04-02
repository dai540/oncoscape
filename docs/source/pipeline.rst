Pipeline Execution
==================

Fast Path
---------

For the shortest path from clone to a deep end-to-end run:

.. code-block:: bash

   python scripts/00_preflight.py --config configs/breast_hpc.yaml
   python scripts/08_run_pipeline.py --config configs/breast_hpc.yaml

Stage-by-Stage
--------------

0. Preflight

.. code-block:: bash

   python scripts/00_preflight.py --config configs/breast_hpc.yaml

1. Register

.. code-block:: bash

   python scripts/01_download_and_register.py --config configs/breast_hpc.yaml

2. Build reference atlas

.. code-block:: bash

   python scripts/02_build_reference_atlas.py --config configs/breast_hpc.yaml

3. Build teachers

.. code-block:: bash

   python scripts/03_make_teachers.py --config configs/breast_hpc.yaml

4. Extract patches and graphs

.. code-block:: bash

   python scripts/04_extract_patches_and_graphs.py --config configs/breast_hpc.yaml

5. Train

.. code-block:: bash

   python scripts/05_train_breast_model.py --config configs/breast_hpc.yaml

6. Evaluate and render

.. code-block:: bash

   python scripts/06_eval_and_render.py --config configs/breast_hpc.yaml

7. Generate reports

.. code-block:: bash

   python scripts/07_generate_reports.py --config configs/breast_hpc.yaml

Validation
----------

Run the regression suite before or after a pipeline change:

.. code-block:: bash

   python -m unittest discover -s tests -v

This includes a synthetic deep pipeline test that asserts strong held-out accuracy on the end-to-end path.

Primary Outputs
---------------

Expected output families:

- ``run/data_interim/slides.csv``
- ``run/data_interim/adata/*.h5ad``
- ``run/outputs/reference_atlas.h5ad``
- ``run/data_interim/labels/teacher_labels.h5ad``
- ``run/data_interim/tile_dataset.parquet`` or ``.parquet.csv``
- ``run/outputs/checkpoints/best.pt``
- ``run/outputs/reports/test_metrics.json``
- ``run/outputs/predictions/<slide_id>/...``
- ``run/outputs/reports/provenance.json``
- ``run/outputs/reports/resolved_config.yaml``

Cluster Launch
--------------

The repository includes a SLURM template:

.. code-block:: bash

   sbatch scripts/cluster/run_breast_pipeline.slurm

The template runs:

1. preflight
2. full pipeline execution
