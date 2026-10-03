Introduction
============

The Full-sun Ultraviolet Rocket Spectrometer (FURST) is a NASA sounding rocket
mission designed to measure the full-disk spectrum of the Sun in the far
ultraviolet spectral range (120 - 180 nm) with high spectral resolution, :math:`R > 20,000`.
FURST is a Rowland circle spectrograph :cite:p:`Rowland1882` which is designed
to minimize the number of optical surfaces, and provides the maximum possible
throughput in the ultraviolet.

This package provides an idealized model of the FURST optical system which is
designed to be used by the FURST data reduction pipeline.
It uses the :mod:`optika` package to model and simulate the optical system.

Installation
============

This package is published to PyPI and can be installed using pip:

.. code-block::

    pip install furst-optics

Reports
=======

.. toctree::
    :maxdepth: 2

    reports/design

Citation
========

If you use :mod:`furst` in your research, please cite it.
The citation metadata is kept in
`CITATION.cff <https://github.com/Kankelborg-Group/furst-optics/blob/main/CITATION.cff>`_,
which the "Cite this repository" button on the
`GitHub page <https://github.com/Kankelborg-Group/furst-optics>`_
can export as BibTeX or APA.
Please include the version of :mod:`furst` that you used,
which is given by ``importlib.metadata.version("furst-optics")``.

.. code-block:: bibtex

    @software{furst-optics,
      author = {Smart, Roy T. and Kankelborg, Charles C.},
      title = {furst-optics},
      version = {X.Y.Z},
      url = {https://github.com/Kankelborg-Group/furst-optics},
    }

API Reference
=============

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    furst


Bibliography
============

.. bibliography::

|


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
