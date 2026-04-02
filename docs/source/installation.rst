Installation
============

Environment
-----------

Recommended runtime:

- Python 3.11
- CUDA-capable PyTorch for production runs
- 24 GB or larger GPU for heavier training setups
- 16+ CPU cores
- 128+ GB RAM
- fast scratch storage

Create the environment:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate oncoscape
   pip install -e .

Install Sphinx for documentation builds:

.. code-block:: bash

   pip install sphinx

Build the documentation:

.. code-block:: bash

   sphinx-build -b html docs/source docs/_build/html

On Windows you can also run:

.. code-block:: bat

   docs\\make.bat html
