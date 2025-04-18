.. _api_documentation:

=================
API Documentation
=================

This is the API reference for the neurodsp module.

Table of Contents
=================

.. contents::
    :local:
    :depth: 1


Create NWB Files
---------
Coming soon!


FileIO
---------

Functions and utilities for reading and writing data

.. currentmodule:: ieeg2nwb.fileio
.. autosummary::
    :toctree: generated/

    read_ielvis
    _get_tdt_store
    get_tdt_data
    read_tdt_ttls
    read_xfm


Proximal Tissue Density
---------

Function for working with PTD

.. currentmodule:: ieeg2nwb.ptd
.. autosummary::
    :toctree: generated/

    get_ptd_index


Surfaces
---------

Functions and utilities for working with surfaces

.. currentmodule:: ieeg2nwb.surfs
.. autosummary::
    :toctree: generated/

    pial_to_inflated
    find_nearest_vertex
    elec_to_parc
    sub_to_fsaverage
    create_indiv_mapping


Utils
---------

Extra functions that are useful

.. currentmodule:: ieeg2nwb.utils
.. autosummary::
    :toctree: generated/

    compress_data
    read_aseg_csv
    copy_fsaverage_data
    load_nwb_settings