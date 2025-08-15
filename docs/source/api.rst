.. _api_documentation:

=================
API Documentation
=================

This is the API reference for the IEEG2NWB package.

Table of Contents
=================

.. contents::
    :local:
    :depth: 2

Channels
--------

.. currentmodule:: ieeg2nwb.channels

.. autosummary::
    :toctree: generated/

    elec_to_parc
    sub_to_fsaverage
    pial_to_inflated
    get_ptd_index

FileIO
------

.. currentmodule:: ieeg2nwb.fileio

.. autosummary::
    :toctree: generated/

    read_ielvis
    _get_tdt_store
    get_tdt_data
    read_tdt_ttls
    read_xfm

Surfaces
--------

.. currentmodule:: ieeg2nwb.surfs

.. autosummary::
    :toctree: generated/

    find_nearest_vertex
    create_indiv_mapping

Utils
-----

.. currentmodule:: ieeg2nwb.utils

.. autosummary::
    :toctree: generated/

    load_nwb_settings
    inspectNwb
    read_aseg_csv
    copy_fsaverage_data
    get_atlases

Volumes
-------

.. currentmodule:: ieeg2nwb.volumes

.. autosummary::
    :toctree: generated/

    annot_to_volume

Converter
---------

.. currentmodule:: ieeg2nwb.converter

.. autosummary::
    :toctree: generated/

    IEEG2NWB
