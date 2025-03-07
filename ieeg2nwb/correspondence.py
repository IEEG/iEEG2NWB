#import sys
#sys.path.append("/Users/noahmarkowitz/Documents/GitHub/iEEG2NWB")

# # Use absolute imports instead
# from ieeg2nwb.utils import load_nwb_settings
# from ieeg2nwb.messages import example_usage, additional_notes
# from ieeg2nwb.io import read_ielvis

import numpy as np
import pandas as pd
import os
import os.path as op
import re
from datetime import timedelta
import json
import yaml
import argparse
import sys
import glob
from openpyxl import load_workbook
from colorama import Back, Style
from ieeg2nwb.utils import load_nwb_settings
from ieeg2nwb.messages import example_usage, additional_notes
from fileio.ielvis import read_ielvis
try:
    from mne.externals.pymatreader import read_mat
except:
    from pymatreader import read_mat



# Search for excel sheet if none or many
def _select_excel_file():
    from PyQt5.QtWidgets import QApplication, QFileDialog
    import sys

    # Open file dialog to select .xlsx file
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        
    file_dialog = QFileDialog()
    file_dialog.setDirectory(elec_recon_dir)
    file_dialog.setNameFilter("Excel files (*.xlsx)")
    file_dialog.setFileMode(QFileDialog.ExistingFile)

    fname = None
    if file_dialog.exec_():
        fname = file_dialog.selectedFiles()[0]
    else:
        raise Exception("No file selected")
    
    return fname


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

# TODO: Create function for creating full correspondence sheet

#%% Setup and configuration
settings = load_nwb_settings()

freesufer_subject_directory = "/Applications/freesurfer/7.2.0/subjects"
subject_id = "NS162_02"
elec_recon_dir = op.join(freesufer_subject_directory, subject_id, "elec_recon")

# Find correspondence file
fname = glob.glob(op.join(elec_recon_dir, "*correspondence*.xlsx"))
if len(fname) != 1:
    print("Found {} correspondence files, select a correspondence file".format(len(fname)))
    fname = _select_excel_file()
else:
    fname = fname[0]

corr_sheet = pd.read_excel(fname, sheet_name=0,engine='openpyxl')

# Remove PTD file if it exists to emulate empty
if op.isfile(op.join(elec_recon_dir, "GreyWhite_classifications.mat")):
    os.remove(op.join(elec_recon_dir, "GreyWhite_classifications.mat"))

# Add breakpoint to inspect data
# Read iELVis
ielvis_df = read_ielvis(subject_id, subjects_dir=freesufer_subject_directory)


# Get DK atlas volumetric and





