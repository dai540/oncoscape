Overview
========

Project Goal
------------

``oncoscape`` builds a reproducible HPC pipeline that:

1. fetches or registers public breast cancer datasets on cluster storage
2. builds a breast scRNA reference atlas
3. constructs weak teachers from Visium, Xenium, and pathology metadata
4. extracts 112 um H&E tiles and spatial graphs
5. trains a multi-task model
6. evaluates the model and renders slide-level outputs
7. generates executive, wet-lab, and developer summaries

Core Outputs
------------

The pipeline targets three slide-level prediction families:

- ``compartment``: invasive tumor, in situ tumor, stroma, immune-rich, adipose-normal, necrosis/background
- ``composition``: malignant, CAF, endothelial, myeloid, T/NK, B/plasma
- ``programs``: proliferation, EMT, hypoxia, IFN-gamma, inflammatory, angiogenesis, TGFb-like, antigen presentation

Operational Philosophy
----------------------

The repository is built for:

- HPC or GPU-server execution
- public-data-only research
- reproducible, config-driven runs
- strict preflight validation before long jobs

The repository is not intended for:

- diagnostic deployment
- private-cohort-first training
- full-scale local-PC model building
