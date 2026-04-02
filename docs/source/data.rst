Data Preparation
================

Public Data Strategy
--------------------

The pipeline assumes that public breast cancer resources are downloaded directly to cluster storage.

Required datasets:

- Wu 2021 breast spatial atlas
- 10x Visium FFPE human breast cancer
- 10x Visium fresh frozen human breast cancer
- 10x Xenium FFPE human breast
- GSE176078
- GSE161529

Recommended dataset:

- GSE235326

Download Catalog
----------------

Use ``configs/breast_downloads.template.yaml`` as the starting point for the HPC download catalog.

Copy:

.. code-block:: bash

   cp configs/breast_downloads.template.yaml /project/run/config/breast_downloads.yaml

Dry-run the fetch stage:

.. code-block:: bash

   python scripts/00_fetch_public_data.py --config configs/breast_hpc.yaml --dry-run

Execute the fetch stage:

.. code-block:: bash

   python scripts/00_fetch_public_data.py --config configs/breast_hpc.yaml

Source Manifest
---------------

After the public files are present, copy the source manifest template:

.. code-block:: bash

   cp configs/breast_sources.template.yaml /project/run/config/breast_sources.yaml

Minimum required fields per source:

- ``name``
- ``source_type``
- ``platform``
- ``patient_id``
- ``slide_id``
- ``image_path``
- ``counts_path``
- ``annotation_path``
- ``coord_path``
- ``mpp_x``
- ``mpp_y``
- ``coord_unit``
- ``split``

Deterministic Splits
--------------------

When explicit split labels are missing, ``oncoscape`` generates deterministic patient-level and source-level split tables during registration:

- ``run/data_interim/splits/patient_splits.csv``
- ``run/data_interim/splits/source_splits.csv``
