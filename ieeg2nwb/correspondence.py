# Imports
import numpy as np
import pandas as pd
import os
import os.path as op
import re
import tdt
import uuid
from mne.io import read_raw_edf
from datetime import datetime
from dateutil.tz import tzlocal
from datetime import timedelta
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common.table import DynamicTable
#from hdmf.data_utils import DataChunkIterator
# from pynwb import NWBFile, NWBHDF5IO
# from pynwb.base import TimeSeries
# from pynwb.ecephys import ElectricalSeries, ElectrodeGroup
# from pynwb.file import Subject, ElectrodeTable
# from pynwb.epoch import TimeIntervals
# from pynwb.device import Device
# from pynwb.core import ScratchData
# from ndx_events import LabeledEvents, TTLs
import json
import yaml
import argparse
import sys
from openpyxl import load_workbook
from colorama import Back, Style
from .utils import load_nwb_settings
from .messages import example_usage, additional_notes
from .tdt import getTDTStore, read_tdt_ttls

try:
    from mne.externals.pymatreader import read_mat
except:
    from pymatreader import read_mat




"""
1. Leave the first 11 columns. They will be:
* Label
* XLTEK_chan
* TDT_chan
* cable #
* TDT Bank
* GOOD
* Spec
* SOZ
* Spikey
* Out
* Bad


2. For areas
For grid+strip use parcellations
For depth use segmentation (aparc+aseg.mgz)

3. Find more info
* WMvsGM
* LvsR
* sEEG_ECoG

4. Get PTDIndex!!!

5. DK atlas
* full name!!!
* index
* lobe

6. Destrieux atlas info
* shorthand name (G_temporal_inf)!!!
* d index
* full name (Inferior temporal gyrus (T3))

7. YEO atlas info
* Yeo7 label!!!
* Yeo7 index
* Yeo17 label!!!
* Yeo17 index

8. Coordinates
* LEPTO!!!
* PIALVOX
* fsaverage
"""

settings = load_nwb_settings()

freesufer_subject_directory = "/Applications/freesurfer/7.2.0/subjects"
subject_id = "NS162_02"
elec_recon_dir = op.join(freesufer_subject_directory, subject_id, "elec_recon")

# Read correspondence sheet
fname = ""

corr_sheet = pd.read_excel(fname, sheet_name=0,engine='openpyxl')

# Get PTD
if not op.isfile(op.join(elec_recon_dir, "GreyWhite_classifications.mat")):
    print("Running PTDIndex")
    from .ptd import get_ptd_index
    ptd_all = get_ptd_index(subject_id, subject_dir=freesufer_subject_directory)
else:
    ptd_all = read_mat(op.join(elec_recon_dir, "GreyWhite_classifications.mat"))["PTD_idx"]

df = pd.DataFrame(ptd_all)
df = df.drop(labels=["nb_Gpix", "nb_Wpix","offset"], axis=1)
df.rename(columns={"elec": "label", "PTD": "PTD_index", "location": "FS_Vol"}, inplace=True)
df["PTD_index"].fillna(999.0)

# Get DK atlas volumetric and
