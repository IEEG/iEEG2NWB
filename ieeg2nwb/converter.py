#!/usr/bin/env python

# IEEG2NWB.py
# A python-based script meant to convert edf, TDT and ecog.mat formatted data into an
# NWB file
#
# Noah Markowitz
# Human Brain Mapping Lab
# North Shore University Hospital
# April 2025

# Imports
import numpy as np
import pandas as pd
import os
import os.path as op
import glob
import re
import gzip
from datetime import datetime
from dateutil.tz import tzlocal
from datetime import timedelta
import soundfile as sf
import tdt
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb import NWBHDF5IO
from pynwb.base import TimeSeries
from pynwb.ecephys import ElectricalSeries
from pynwb.file import ElectrodeTable, Subject
from pynwb.epoch import TimeIntervals
from ndx_events import TTLs
import json
import yaml
import argparse
import warnings
import sys
#from colorama import Back, Style
from pymatreader import read_mat
import h5py
from ieeg2nwb.utils import load_nwb_settings
from ieeg2nwb.fileio.tdt import _get_tdt_store, get_tdt_data, read_tdt_ttls
from ieeg2nwb.fileio import read_ielvis

# TODO
#   - "make BIDS" option to create json sidecars for BIDS
#   - "create_path" to create output directory if it does not exist
#   - "externalize" option
#   - All above should work in
#       * Command-line
#       * params file
#       * gui
#       * batch
#   - If correspondence sheet as blank cells that contain only whitespace, remove the whitespaces

class IEEG2NWB:


    def __init__(self, description=None):

        self._params = load_nwb_settings()

        # NWBFile object
        self.nwbfile = None
        # A Subject object for NWB
        self.subject = None
        # Start date and time
        self.start_time = None
        # Session description
        self.description = description if description is not None else self._params["session_description"]
        # Device object for NWB (the amplifier used)
        self.amplifier = None
        # Events table
        self.events = None
        # Electrode groups (LDa, LFp, etc.)
        self.electrode_groups = []
        # Output NWB file
        self.output_file = None
        # File containing raw data
        self.raw_data_file = None
        # Type of raw data
        self.raw_data_type = None
        # Labels that correspond to each channgel
        self.channel_labels = {"label": [], "channel": []}
        # Number of channels available to use in the data
        self.n_chans = 0
        # For TDT data, the stores containing the primary eeg data
        self.eeg_stores = []
        # Whether to create the path if it does not exist
        self.create_path = False
        # Annotations to add as a list of tuples (onset, description)
        self.annotations = []
        # Correspondence file
        self.correspondence_table = None
        # Table of elctrodes
        self.electable = None
        # Column definitions for electrode table
        self.electable_columns = None
        # Table regions for electrode table
        self.electable_regions = None
        # Channels for electable
        self.electable_channels = None
        # HDF5 files used to save to NWB
        self.hdf5_files = []

        # Additional notes
        self.annotations = {'timestamps': [], 'notes': []}

        # Freesurfer directory
        self.freesurfer_subject_dir = None
        self.freesurfer_subject_id = None


    def init_nwbfile(self, description=None, start_time=None):
        from pynwb import NWBFile
        from uuid import uuid4

        if description is None:
            description = self.description

        if start_time is None and self.start_time is None:
            start_time = datetime.strptime(self._params["start_time"], "%Y-%m-%d %H:%M:%S").astimezone()
        elif start_time is None:
            from dateutil.relativedelta import relativedelta
            start_time = self.start_time - relativedelta(months=1000)
        else:
            raise ValueError("Must provide a start time")

        self.nwbfile = NWBFile(description, str(uuid4()), start_time)



    def create_subject(self,subject_id=None,sex=None,species=None,age=None,subject_description=None):
        """Create subject object."""
        subject = Subject(
            age=self._params['subject_age'] if age is None else 'P' + str(age) + 'Y',
            sex=self._params['subject_sex'] if sex is None else sex,
            species=self._params['subject_species'] if species is None else species,
            subject_id=self._params['subject_id'] if subject_id is None else subject_id,
            description=self._params['subject_description'] if subject_description is None else subject_description
        )
        self.nwbfile.subject = subject

    def create_device(self, device_name, description=None, manufacturer=None):
        """Create device object (ex: amplifier)."""
        is_amplifier = False
        if device_name.lower() == "xltek" or device_name.lower() == "natus":
            device_info = self._params["devices"]["natus"]
            is_amplifier = True
        elif device_name.lower() == "tdt":
            device_info = self._params["devices"]["tdt"]
            is_amplifier = True
        else:
            if description is None or manufacturer is None:
                valid_devices = ", ".join(self._params["devices"].keys())
                raise ValueError(
                    f"device_name must be one of the following: {valid_devices}"
                    "; or provide a description and manufacturer for the device."
                )
            else:
                device_info = {
                    "name": device_name,
                    "description": description,
                    "manufacturer": manufacturer
                }

        if "search" in device_info.keys():
            device_info.pop("search")

        device = self.nwbfile.create_device(**device_info)

        if is_amplifier:
            self.amplifier = device

        return device

    def set_freesurfer(self, subject_id=None, subject_dir=None):
        """Set the freesurfer directory for where to read info."""
        if subject_dir is None:
            from mne import get_config
            self.freesurfer_subject_dir = get_config()['SUBJECTS_DIR']
        else:
            self.freesurfer_subject_dir = subject_dir

        if subject_id is None and self.nwbfile.subject is None:
            raise ValueError("subject_id must be provided if no subject object exists")
        elif subject_id is None:
            self.freesurfer_subject_id = self.nwbfile.subject.subject_id
        else:
            self.freesurfer_subject_id = subject_id

    def read_correspondence_sheet(self, correspondence_sheet=None, raw_data_type=None):
        """Read the correspondence sheet."""
        if correspondence_sheet is not None:
            if not op.exists(correspondence_sheet):
                raise FileNotFoundError(f"Correspondence sheet {correspondence_sheet} not found")
            elif op.exists(correspondence_sheet):
                corr_sheet = pd.read_excel(correspondence_sheet, sheet_name=0, engine='openpyxl')
        elif self.freesurfer_subject_dir is not None and self.freesurfer_subject_id is not None:
            # Find correspondence sheet
            elec_recon_dir = op.join(self.freesurfer_subject_dir, self.freesurfer_subject_id, "elec_recon")
            corr_list = glob.glob(op.join(elec_recon_dir, "*correspondence*"))
            if len(corr_list) == 0:
                raise FileNotFoundError("Correspondence sheet not found")
            elif len(corr_list) > 1:
                raise ValueError("Multiple correspondence sheets found. Remove the ones not needed")
            corr_sheet_fname = corr_list[0]

            # Read correspondence sheet
            corr_sheet = pd.read_excel(corr_sheet_fname, sheet_name=0, engine='openpyxl')

        if raw_data_type is None and self.raw_data_type is None:
            raise ValueError("raw_data_type must be provided if raw data file hasn't been submitted yet")
        elif raw_data_type is None:
            raw_data_type = self.raw_data_type

        # Select channel column
        if raw_data_type in ["edf", "natus", "xltek"]:
            ch_col_name = "xltek*"
        elif raw_data_type == "tdt":
            ch_col_name = "tdt*"
        else:
            raise TypeError("raw_data_type must be one of the following: edf, tdt, xltek")

        corr_columns = list(corr_sheet.columns)

        # Get rid of all channels without labels
        label_col_name = [col for col in corr_columns if 'label' in col.lower()][0]
        corr_sheet.rename(columns={label_col_name: 'label'}, inplace=True)
        corr_sheet["label"] = corr_sheet["label"].str.strip()
        corr_sheet = corr_sheet[~corr_sheet["label"].isin(['[]', ''])]
        corr_sheet = corr_sheet.dropna(subset=["label"])

        # Rename column has relevant channel numbering
        r = re.compile(ch_col_name, re.IGNORECASE)
        chn_col_name = list(filter(r.match, corr_columns))[0]
        corr_sheet.rename(columns={chn_col_name: 'channel'}, inplace=True)

        # Convert the channel column to numeric, drop all non-numeric elements and their rows
        corr_sheet['channel'] = pd.to_numeric(corr_sheet['channel'], errors='coerce')
        corr_sheet = corr_sheet.dropna(subset=['channel'])
        corr_sheet['channel'] = corr_sheet['channel'].astype(int)

        corr_sheet = corr_sheet.reset_index(drop=True)

        self.correspondence_table = corr_sheet

    def create_electrode_groups(self):
        """Create the ElectrodeGroup objects."""
        corr_sheet = self.correspondence_table
        labels = corr_sheet['label'].to_list()
        intracranial_specs = self._params["intracranial_specs"]

        expr = r'[A-Za-z]+'
        elecgroups = {}
        elecs_dict = {"label": [], "group": [], "group_name": []}
        group_col = []
        group_name_col = []
        for elec in labels:

            group_name = re.match(expr, elec).group()

            # If group doesn't exist yet then create it
            if group_name not in elecgroups.keys():

                elec_spec = corr_sheet.loc[corr_sheet['label'] == elec, "spec"].iloc[0]

                if elec_spec not in intracranial_specs.keys():
                    desc = '%s, %s type electrodes. Recorded outside the brain' % (group_name, elec_spec)
                    loc = "outside of brain"
                else:
                    recording_type = intracranial_specs[elec_spec]
                    if elec_spec.startswith("hd"):
                        elec_spec = "high density " + elec_spec.split("hd_")[-1]
                    desc = '%s is a %s type electrodes recording %s data' % (group_name, elec_spec, recording_type)
                    loc = "Brain"

                elecgroups[group_name] = self.nwbfile.create_electrode_group(
                    group_name,
                    description=desc,
                    location=loc,
                    device=self.amplifier
                )

            elecs_dict["label"].append(elec)
            elecs_dict["group"].append(elecgroups[group_name])
            elecs_dict["group_name"].append(group_name)
            #group_col.append(elecgroups[group_name])
            #group_name_col.append(group_name)

        groups_df = pd.DataFrame(elecs_dict)

        if "group" in corr_sheet.columns and "group_name" in corr_sheet.columns:
            corr_sheet = corr_sheet.drop(labels=["group", "group_name"], axis=1)

        corr_sheet = corr_sheet.merge(groups_df, on="label", how="left")

        #corr_sheet.loc[:,'group'] = group_col
        #corr_sheet.loc[:,'group_name'] = group_name_col
        self.correspondence_table = corr_sheet
        self.electrode_groups = elecgroups

        #return elecs_dict

    def process_correspondence_sheet(self, update=True):
        """Add info to correspondence sheet and prepare it to become ElectrodeTable"""

        # Parameters for the columns of the electrode table
        cols_for_table = self._params["electrode_table"]["columns"]

        corr_sheet = self.correspondence_table

        # Use only the number of available channels
        if self.n_chans > 0:
            corr_sheet = corr_sheet.loc[corr_sheet["channel"] <= self.n_chans,:]
            corr_sheet = corr_sheet.reset_index(drop=True)
        
        # Freesurfer stuff
        subject_id = self.freesurfer_subject_id
        subjects_dir = self.freesurfer_subject_dir

        if update:
                
            # Get ielvis data
            elecs_df = read_ielvis(subject=subject_id, subjects_dir=subjects_dir, squeeze=False, parcs=True)
            
            # Remove spec column from elecs_df because correspondence sheet should have
            elecs_df = elecs_df.drop(columns=["spec"])
            
            # Take first 11 columns from corr_sheet and merge with ielvis_df on label column
            n_columns_corr = len(corr_sheet.columns)
            if n_columns_corr > 11:
                corr_sheet = corr_sheet.iloc[:, :11]
            else:
               corr_sheet = corr_sheet.iloc[:, :n_columns_corr] 
    
            # Check for labels in elecs_df that aren't in corr_sheet
            elecs_df_labels = set(elecs_df["label"].str.lower())
            corr_sheet_labels = set(corr_sheet["label"].str.lower())
            missing_from_corr = elecs_df_labels - corr_sheet_labels
            if len(missing_from_corr) > 0:
                warnings.warn(f"Found labels in ielvis that are not in correspondence sheet: {missing_from_corr}")
    
            # Check for labels in corr_sheet that aren't in elecs_df
            missing_from_ielvis = corr_sheet_labels - elecs_df_labels
            if len(missing_from_ielvis) > 0:
                print(f"Found labels in correspondence sheet that are not in ielvis: {missing_from_ielvis}")
    
            # Merge while preserving order of corr_sheet labels
            # Create temporary lowercase columns for case-insensitive merge
            corr_sheet['label_lower'] = corr_sheet['label'].str.lower()
            elecs_df['label_lower'] = elecs_df['label'].str.lower()
            
            elecs_df = pd.merge(corr_sheet, elecs_df,
                               left_on="label_lower",
                               right_on="label_lower", 
                               how="outer",
                               sort=False,
                               suffixes=(None, '_drop')).reindex(corr_sheet.index)
            
            # Drop temporary lowercase columns
            elecs_df = elecs_df.drop(columns=['label_lower'])
            
        else:
            
            elecs_df = corr_sheet
        
        # Drop any duplicate label columns 
        elecs_df = elecs_df.loc[:, ~elecs_df.columns.str.endswith('_drop')]

        # Sort by the "channel" column
        elecs_df = elecs_df.sort_values(by="channel").reset_index(drop=True)
        elecs_df["channel"] = elecs_df["channel"].values -1 
        
        # Store channel and labels in a dict for later
        self.channel_labels = {
            "label": corr_sheet.loc[:,["label"]].values.flatten(),
            "channel": corr_sheet.loc[:,["channel"]].values.flatten()
        }

        # Drop channel column
        #corr_sheet = corr_sheet.drop("channel", axis=1)

        # Variables to use later
        dynamic_columns = [] # Column definitions for ElectrodeTable, list of dictionaries
        cols2rename = {} # Columns to rename for formatting
        cols2keep = [] # Columns to not drop
        missing_cols = [] # Columns that are missing

        # Go through columns
        for c in cols_for_table.keys():

            col_settings = cols_for_table[c]
            is_required = col_settings['required']
            in_ielvis = False
            
            if 'default' in col_settings.keys():
                if col_settings['default'] == 999:
                    col_settings['default'] = np.nan

            # Find the column
            colfound = []
            if 'search' in col_settings.keys():
                r = re.compile(col_settings['search'], re.IGNORECASE)
                colfound = list(filter(r.match, elecs_df.columns))

            # IF column is found then give it the right name and fill blank cells
            if len(colfound) > 0:
                cols2rename[colfound[0]] = col_settings['title']
                if 'default' in col_settings.keys():
                    elecs_df[colfound[0]] = elecs_df[colfound[0]].fillna(col_settings['default'])

            # Column is required, absent and has a default
            elif (len(colfound) == 0) & is_required & ('default' in col_settings.keys()):
                elecs_df[col_settings['title']] = col_settings['default']
                missing_cols.append(col_settings['title'])

            # Column is required, absent
            elif len(colfound) == 0 & is_required:
                elecs_df[col_settings['title']] = 'None'
                missing_cols.append(col_settings['title'])

            # Make into the correct data type
            if ('type' in col_settings.keys()) & (len(colfound) != 0):
                try:
                    elecs_df[colfound[0]] = elecs_df[colfound[0]].astype(col_settings['type'])
                except ValueError:
                    elecs_df[colfound[0]].replace("None", "0").astype(col_settings['type'])

            # Replace values
            if "replace" in col_settings:
                for idx, val in elecs_df[colfound[0]].items():
                    if val in col_settings["replace"]:
                        elecs_df.at[idx, colfound[0]] = col_settings["replace"][val]

            # Append dynamic_columns and mark this column as being kept
            if (len(colfound) > 0) | is_required | in_ielvis:
                cols2keep.append(col_settings['title'])
                dynamic_columns.append({'name': col_settings['title'], 'description': col_settings['description']})

            # Is column still missing and required?
            if (len(colfound) == 0):
                missing_cols.append(col_settings['title'])

        #old_sheet = elecs_df.copy()
        
        # Keep only the needed columns in dataframe
        elecs_df = elecs_df.rename(columns=cols2rename)#.loc[:, cols2keep]

        # Store column definitions for creating ElectrodeTable
        self.electable_columns = dynamic_columns

        # Store the processed correspondence sheet
        self.correspondence_table = elecs_df

    def create_electrode_table(self):
        """Create the ElectrodeTable that is a DynamicTable object."""
        cols_to_keep = [c["name"] for c in self.electable_columns]
        elecs_df = self.correspondence_table.loc[:, cols_to_keep]
        electable = ElectrodeTable().from_dataframe(
            elecs_df,
            self._params["electrode_table"]["name"],
            table_description=self._params["electrode_table"]["description"],
            columns=self.electable_columns
        )
        self.nwbfile.electrodes = electable

    def create_electrode_table_regions(self):
        """Create the regions for the ElectrodeTable."""

        table_regions = {}

        intracranial_specs = self._params["intracranial_specs"]

        df = self.nwbfile.electrodes.to_dataframe()
        spec_indices = {'ieeg': []}
        for idx, row in df.iterrows():
            if row.spec.lower() in intracranial_specs:
                spec_indices['ieeg'].append(idx)
            else:
                if row.spec.lower() not in spec_indices:
                    spec_indices[row.spec.lower()] = []

                spec_indices[row.spec.lower()].append(idx)

        for spec, chans in spec_indices.items():
            table_regions[spec] = self.nwbfile.create_electrode_table_region(
                region=chans,
                description=f"electrodes recording {spec} data",
                name="electrodes"
            )
        self.electable_regions = table_regions

    def read_raw_data(self, raw_data_files, create_device=True, eeg_chans=None):
        """Set the raw data file."""
        self.raw_data_file = raw_data_files

        # Check if file exists
        if not op.exists(raw_data_files):
            raise FileNotFoundError(f"File {raw_data_files} not found")
        
        # Check what type of file it is
        if raw_data_files.endswith('.edf'):
            from mne.io import read_raw_edf
            self.raw_data_type = 'edf'
            self.raw_data = read_raw_edf(raw_data_files, preload=True)
            for annot in self.raw_data.annotations:
                self.annotations.append((annot['onset'], annot['description']))
            amplifier = "xltek"
            start_time = self.raw_data.info["meas_date"]
            self.n_chans = len(self.raw_data.ch_names)
        elif op.isdir(raw_data_files):
            dir_contents = os.listdir(raw_data_files)
            file_extensions = {op.splitext(f)[-1] for f in dir_contents if op.isfile(os.path.join(raw_data_files, f))}
            if ".tev" in file_extensions or ".sev" in file_extensions:
                self.raw_data_type = 'tdt'
                self.raw_data = tdt.read_block(raw_data_files)
                if "info" in self.raw_data.keys():
                    start_time = self.raw_data.info.start_date
                amplifier = "tdt"
                eeg_data, _, stores = self._get_tdt_eeg_data(eeg_chans=eeg_chans, return_store_list=True)
                self.n_chans = eeg_data.shape[0]
                self.eeg_stores = stores
            elif ".erd" in file_extensions:
                from nwreader import read_erd
                self.raw_data_type = 'xltek'
                self.raw_data = read_erd(raw_data_files, use_dask=True, convert=True, pad_discont=True)
                start_time = self.raw_data.attrs["creation_time"]
                amplifier = "xltek"
                self.n_chans = self.raw_data.data.shape[0]
        else:
            raise ValueError("File type not recognized. Must be edf, tdt, or xltek")
            return

        if self.start_time is None:
            self.start_time = start_time

        if self.nwbfile is None:
            self.init_nwbfile()

        if create_device:
            self.amplifier = self.create_device(amplifier)

    def create_analog_acquisitions(self, analog_stores):
        """
        Create the analog acquisition.

        Pass in a list of dictionaries with the following keys:
        * name: name of the acquisition
        * description: description of the acquisition
        * store: name of the store in the raw data, only applicable to TDT data (ex: Wav5)
        * channels: list of channels to use with 0-based indexing (ex: [0, 1])
        * comments: string with additional comments, optional
        * unit: string with units of data, default is "volts"

        ex:
        {
            "name": "audio",
            "description": "audio signal",
            "store": "Wav5",
            "channels": [0, 1],
            "comment": "contains beeps",
            "unit": "volts"
        }

        """
        if self.raw_data_type == "tdt":
            self._tdt_create_analog_acquisition(analog_stores)
        elif self.raw_data_type == "xltek":
            self._xltek_create_analog_acquisition(analog_stores)
        elif self.raw_data_type == "edf":
            self._edf_create_analog_acquisition(analog_stores)

    def create_digital_acquisition(
            self,
            stores,
            name="TTLs",
            description="TTL pulses emitted at specific events"
    ):
        """Create the digital acquisition.
        ex:
        create_digital_acquisition(['PtC2', 'PtC4', 'PtC6'])
        """

        # Get the timestamps and stores they're from
        event_times = read_tdt_ttls(self.raw_data, stores)
        event_times_df = pd.DataFrame(event_times)

        # Get rid of any timestamps occurring at t=0
        invalid_timestamps = event_times_df["time"] == 0
        if len(invalid_timestamps)>0:
            event_times_df = event_times_df[~invalid_timestamps]

        # Make numeric code
        unique_ids = event_times_df['stores'].unique()
        label_vals = list(range(unique_ids.size))
        store_times = event_times_df['stores'].tolist()
        store_codes = dict(zip(unique_ids, label_vals))
        codes = []
        for ii in store_times:
            codes.append(store_codes[ii])

        # Create the TTLs object and add to NWB file
        events = TTLs(
            name=name,
            description=description,
            timestamps=event_times_df['time'].to_numpy(),
            data=codes,
            labels=unique_ids
        )
        self.nwbfile.add_acquisition(events)

    def format_data(self, eeg_chans=None):
        """Process the raw data."""
        if self.raw_data_type == "tdt":
            self._tdt_format_data(eeg_chans)
        elif self.raw_data_type == "xltek":
            self._xltek_format_data()
        elif self.raw_data_type == "edf":
            self._edf_format_data()

    def add_annotations(self):
        """Add annotations from raw data file to the NWB file."""

        if len(self.annotations) == 0:
            raise ValueError("No annotations to add")
            return

        # Filter the annotations first
        passed_annotations = {"timestamp": [], "description": []}
        annot_filter = '(?:% s)' % '|'.join(self._params["annotations_to_ignore"])
        for annot in self.annotations:
            if not re.search(annot_filter, annot[1]):
                passed_annotations["timestamp"].append(annot[0])
                passed_annotations["description"].append(annot[1])

        if len(passed_annotations["timestamp"]) == 0:
            print("---> No annotations valid to add")
            return

        # Create a LabeledEvents object to store
        from ndx_events import LabeledEvents
        annotations = LabeledEvents(
            name='annotations',
            description='annotations directly from recorded file',
            timestamps=passed_annotations['timestamps'],
            labels=passed_annotations['description'],
            data=np.arange(len(passed_annotations['timestamps']))
        )
        self.nwbfile.add_acquisition(annotations)


    def _create_timeseries(self, name, data, fs, description=None, comments=None, unit="volts"):
        """Create the TimeSeries object."""

        if comments is None:
            comments = "no comments"

        if isinstance(data, h5py._hl.dataset.Dataset):
            compressed_data = H5DataIO(data=data, link_data=False)
        else:
            compressed_data = H5DataIO(
                data=data,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                maxshape=(None, data.shape[1]),
                chunks=True
            )
        print(f"---> Adding {name} acquisition to NWBFile")
        ts = TimeSeries(
            name=name,
            data=compressed_data,
            rate=float(fs),
            description=description,
            unit=unit,
            comments=comments,
            starting_time=0.0
        )
        self.nwbfile.add_acquisition(ts)

    def _create_electricalseries(self, name, data=None, fs=None, description=None, electrodes=None):
        """
        Create the ElectricalSeries object.
        Data must already be in correct format (samples x channels)
        """
        if isinstance(data, h5py._hl.dataset.Dataset):
            compressed_data = H5DataIO(data=data, link_data=False)
        else:
            compressed_data = H5DataIO(
                data=data,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                maxshape=(None, len(electrodes.data)),
                chunks=True
            )
        print(f"---> Adding {name} acquisition to NWBFile")
        es = ElectricalSeries(
            name=name,
            data=compressed_data,
            rate=float(fs),
            description=description,
            electrodes=electrodes,
            starting_time=0.0,
        )
        self.nwbfile.add_acquisition(es)

    def _edf_format_data(self):
        """Take EDF raw data and format it to be stored in NWB."""

        # Get data
        eeg_array = self.raw_data.get_data()
        fs = self.raw_data.info["sfreq"]

        eeg_array = eeg_array[self.channel_labels["channel"], :]

        # Create the ElectricalSeries object for each table region
        for acq_name,region in self.electable_regions.items():

            # # Figure out which channels are in the region
            # electable_rows = self.electable_regions[region].data
            # elec_names = self.correspondence_table.loc[electable_rows, "label"].to_list()
            # df = pd.DataFrame(self.channel_labels)
            # elec_indices = df[df["label"].isin(elec_names)].index
            # elec_channels = df.loc[elec_indices,"channel"].to_list()
            elec_channels = region.data

            self._create_electricalseries(
                acq_name,
                data=eeg_array[elec_channels, :].T,
                fs=fs,
                description=f"data recorded from {acq_name} electrodes",
                electrodes=region
            )

    def _edf_create_analog_acquisition(self, analog_stores):
        """For EDF data create extra analog acquisitions."""
        for eac in analog_stores:

            # Get all stores
            # analog_array = self.raw_data.get_data()[eac["channels"], :]
            analog_array = self.raw_data.get_data(np.array(eac["channels"])-1)
            fs = self.raw_data.info["sfreq"]

            if "unit" in eac.keys():
                unit = eac["unit"]
            else:
                unit = "volts"

            if "comments" in eac.keys():
                comments = eac["comments"]
            else:
                comments = None

            # Add as a TimeSeries
            self._create_timeseries(
                eac["name"],
                analog_array.T,
                fs,
                description=eac["description"],
                unit=unit,
                comments=comments
            )

    def _tdt_create_analog_acquisition(self, analog_stores):
        """For TDT data create extra analog acquisitions.
        
        Expects list of dicts
        {"store": <str or list of store containing the data>,
        "channel": <channels in the stores to use>,
        "unit": <unit of measurement (ex: volts)>,
        "comments": <any additional comments>,
        "name": <name of the new TimeSeries>}
        """
        for eac in analog_stores:

            # Get all stores
            analog_array, fs = get_tdt_data(self.raw_data, eac["store"])

            if "channels" not in eac.keys():
                eac["channels"] = None

            # Take only the data from the channels that are in the correspondence table
            if eac["channels"] is not None:
                analog_array = analog_array[np.array(eac["channels"])-1, :]

            if "unit" in eac.keys():
                unit = eac["unit"]
            else:
                unit = "volts"

            if "comments" in eac.keys():
                comments = eac["comments"]
            else:
                comments = None

            # Format data when writing to file
            if eac['write_to_file']:
                
                # Create directory if needed
                file_dir, _ = os.path.split(eac['file'])
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                    
                # Format data
                dim1, dim2 = analog_array.shape
                if dim1 < dim2:
                    analog_array = analog_array.T
                    
                # Write data to a wav file
                if '.wav' in eac['file']:    
                    sf.write(eac['file'], analog_array, round(fs), format='WAV')
                    
                # Write data to a tsv.gz file 
                elif eac['file']:
            
                    with gzip.open(eac['file'], 'wt') as f:    
                        for a in range(len(analog_array)): 
                            f.write('{:1.20f}\t\n'.format(analog_array[a][0]))             
    
                    # Create corresponding json file 
                    ana_descr = {
                        'SamplingFrequency': fs,
                        'StartTime': 0,
                        'Columns': [eac['name']]
                    }
                    
                    json_file = eac['file'].replace('.tsv.gz', '.json')
                    
                    with open(json_file, 'w') as file:
                        json.dump(ana_descr, file, indent=4)
                        
            # Add as a TimeSeries
            else:
            
                self._create_timeseries(
                    eac["name"],
                    analog_array.T,
                    fs,
                    description=eac["description"],
                    unit=unit,
                    comments=comments
                )

    def _get_tdt_eeg_data(self, eeg_chans=None, return_store_list=False):
        # The default is to look for the EEG stores listed here
        tdt_eeg_channels = self._params["tdt_neuro_channels"]

        eeg_array = None
        fs = None

        store_list = []
        if eeg_chans is None:
            for streams in tdt_eeg_channels:
                eeg_array, fs, store_list = get_tdt_data(self.raw_data, streams, ignore_missing=True, return_store_list=True)
                if eeg_array is not None:
                    break

        else:
            if not isinstance(eeg_chans, list):
                eeg_chans = list(eeg_chans)
            eeg_array, fs, available_stores = get_tdt_data(self.raw_data, eeg_chans, ignore_missing=False, return_store_list=True)
            store_list = available_stores

        if eeg_array is None:
            raise RuntimeError(f"EEG data could not be found in the following stores: {tdt_eeg_channels}")
            return None
        
        if return_store_list:
            return eeg_array, fs, store_list
        else:
            return eeg_array, fs

    def _tdt_format_data(self, eeg_chans=None):
        """Take TDT raw data and format it to be stored in NWB."""

        if eeg_chans is None:
            eeg_chans = self.eeg_stores

        eeg_array, fs = get_tdt_data(self.raw_data, self.eeg_stores, ignore_missing=False)
        
        # Create the ElectricalSeries object for each table region
        for acq_name, region in self.electable_regions.items():

            # Figure out which channels are in the region
            # electable_rows = self.electable_regions[region].data
            # elec_names = self.correspondence_table.loc[electable_rows, "label"].to_list()
            # df = pd.DataFrame(self.channel_labels)
            # elec_indices = df[df["label"].isin(elec_names)].index
            # elec_channels = df.loc[elec_indices,"channel"].to_list()

            elec_channels = region.data

            self._create_electricalseries(
                acq_name,
                data=eeg_array[elec_channels, :].T,
                fs=fs,
                description=f"data recorded from {acq_name} electrodes",
                electrodes=region
            )

    def _xltek_format_data(self):
        """Format the xltek data."""
        import dask.array as da

        fs = self.raw_data.attrs["sample_freq"]

        if self.output_file is not None:
            pdir = op.dirname(self.output_file)
        else:
            pdir = op.dirname(self.raw_data_file)

        # Concatenate the data so only channels being used are present
        raw_data_dask = self.raw_data.data[self.channel_labels["channel"], :]

        # Create the ElectricalSeries object for each table region
        for acq_name in self.electable_regions.keys():

            # Figure out which channels are in the region
            # electable_rows = self.electable_regions[region].data
            # elec_names = self.correspondence_table.loc[electable_rows, "label"].to_list()
            # df = pd.DataFrame(self.channel_labels)
            # elec_indices = df[df["label"].isin(elec_names)].index
            # elec_channels = df.loc[elec_indices, "channel"].to_list()

            region = self.electable_regions[acq_name]
            elec_channels = region.data

            # Data for this acquisition
            #this_acq_data = self.raw_data.data[elec_channels, :].T
            this_acq_data = raw_data_dask[elec_channels, :].T

            # Save it to a temporary file
            tmp_filename = op.join(pdir, f"{acq_name}.hdf5")
            print(f"----->Saving {acq_name} data to temporary hdf5 file. This could take a second")
            da.to_hdf5(
                tmp_filename,
                "/data",
                this_acq_data,
                chunks=True,
                compression="gzip"
            )

            # open the temp file
            f = h5py.File(tmp_filename, "r")
            stored_data = f["/data"]  # HDMF5 dataset object

            # Create the ElectricalSeries object
            self._create_electricalseries(
                region,
                data=stored_data,
                fs=fs,
                description=f"data recorded from {acq_name} electrodes",
                electrodes=region,
            )

            # Append file handle to list to close later
            self.hdf5_files.append((tmp_filename, f))

    def _xltek_create_analog_acquisition(self, analog_stores):
        """For XLTEK data create extra analog acquisitions."""
        import dask.array as da

        if self.output_file is not None:
            pdir = op.dirname(self.output_file)
        else:
            pdir = op.dirname(self.raw_data_file)

        fs = self.raw_data.attrs["sample_freq"]

        for eac in analog_stores:

            # Get all stores
            analog_array = self.raw_data.data[eac["channels"], :]

            if "unit" in eac.keys():
                unit = eac["unit"]
            else:
                unit = "volts"

            if "comments" in eac.keys():
                comments = eac["comments"]
            else:
                comments = None

            acq_name = eac["name"]

            # Save it to a temporary file
            tmp_filename = op.join(pdir, f"{acq_name}.hdf5")
            print(f"----->Saving {acq_name} data to temporary hdf5 file. This could take a second")
            da.to_hdf5(
                tmp_filename,
                "/data",
                analog_array,
                chunks=True,
                compression="gzip"
            )

            # open the temp file
            f = h5py.File(tmp_filename, "r")
            stored_data = f["/data"]  # HDMF5 dataset object

            # Add as a TimeSeries
            self._create_timeseries(
                eac["name"],
                stored_data,
                fs,
                description=eac["description"],
                unit=unit,
                comments=comments
            )

            # Append file handle to list to close later
            self.hdf5_files.append((tmp_filename, f))


    def write_json_sidecar(self, filename=None, **kwargs):
        """Write json sidecar file as per BIDS specification

        Input kwargs are keywords used in BIDS sidecar
        """

        # Get number of types of channels
        chan_count = self.correspondence_table["spec"].value_counts().to_dict()

        # Get recording duration
        recording_dur = self.nwbfile.acquisition['ieeg'].data.shape[0] / self.nwbfile.acquisition['ieeg'].rate

        # Add a dataset descriptor
        json_dict = {
            "iEEGReference": "intracranial electrode not included with data",
            "SamplingFrequency": self.nwbfile.acquisition['ieeg'].rate,  
            "PowerLineFrequency": int(60),
            "SoftwareFilters": "n/a",  
            "HardwareFilters": "n/a",
            "ElectrodeManufacturer": "AdTech",
            "ECOGChannelCount": chan_count["ecog"] if "ecog" in chan_count else 0,
            "SEEGChannelCount": chan_count["seeg"] if "seeg" in chan_count else 0,
            "EEGChannelCount": int(0),
            "EOGChannelCount": int(0),
            "ECGChannelCount": int(0),
            "EMGChannelCount": int(0),
            "MiscChannelCount": int(0),
            "TriggerChannelCount": int(0),
            "RecordingDuration": recording_dur,
            "RecordingType": "continuous",
            "Manufacturer": "Tucker Davis Technologies",       
            "TaskName": self.nwbfile.session_id,          
            "TaskDescription": self.nwbfile.session_description,            
            "InstitutionName": self.nwbfile.institution,
        }

        # Loop through kwargs and update json_dict if keys match (case insensitive)
        for key, value in kwargs.items():
            # Check if key exists in json_dict (case insensitive)
            matching_key = next((k for k in json_dict.keys() if k.lower() == key.lower()), None)
            if matching_key:
                json_dict[matching_key] = value

        if filename is None:
            filename = self.output_file
        
        if filename is None:
            raise ValueError("No output filename provided")
            
        # Make sure filename ends with .json
        if not filename.endswith('.json'):
            filename += '.json'
            
        # Write json_dict to file
        with open(filename, 'w') as f:
            json.dump(json_dict, f, indent=4)


    def write_nwb(self, nwb_file=None):
        """Write the NWB file."""
        if nwb_file is None and self.output_file is None:
            raise ValueError("Output file must be provided")
        elif nwb_file is None:
            nwb_file = self.output_file

        # Make sure the filename ends with .nwb
        if not nwb_file.endswith('.nwb'):
            nwb_file += '.nwb'

        # Save the output filename being used
        self.output_file = nwb_file.strip(".nwb")

        # If available annotations haven't been added then add them
        if len(self.annotations["timestamps"]) > 0 and "annotations" not in self.nwbfile.acquisition.keys():
            self.add_annotations()

        # If some data is None or missing, add it in using default settings
        if self.nwbfile.lab is None: self.nwbfile.lab = self._params["lab"]
        if self.nwbfile.institution is None: self.nwbfile.institution = self._params["institution"]
        
        # Write the NWB file
        with NWBHDF5IO(nwb_file, 'w') as io:
            print(f"---> Writing to {nwb_file}")
            io.write(self.nwbfile, link_data=False)

        # Close the h5 files
        if len(self.hdf5_files) > 0:
            for fname, handle in self.hdf5_files:
                handle.close()
                os.remove(fname)

        # Close EDF file
        if self.raw_data_type == "edf":
            self.raw_data.close()

        # Write out the json file
        self.write_json_sidecar(self.output_file)

    def parse_params(self, params):
        """Run the entire converter given params."""

        # Read data
        print('-----> Reading input: %s' % params['block'])
        eeg_chans = params.get('neurodata')
        self.read_raw_data(params['block'], create_device=True, eeg_chans=eeg_chans)

        # Get subject specific info and create subject
        subinfo = ['subject_id', 'sex', 'age', 'subject_description']
        subdict = {}
        for s in subinfo:
            if s in params.keys():
                subdict[s] = str(params[s])

        if len(subdict.keys())==0 and "subject" in params.keys():
            subdict = params["subject"]
        
        if subdict:
            self.create_subject(**subdict)
        
        # Add freesurfer info
        freesurfer_subject_id = params["subject_id"]
        freesurfer_subject_directory = None
        if "freesurfer_subject_id" in params.keys():
            freesurfer_subject_id = params["freesurfer_subject_id"]
        if "freesurfer_subject_directory" in params.keys():
            freesurfer_subject_directory = params["freesurfer_subject_directory"]
        self.set_freesurfer(subject_id=freesurfer_subject_id, subject_dir=freesurfer_subject_directory)

        # If the labelfile key is there then use as much info in that as possible
        if "labelfile" in params.keys() and freesurfer_subject_directory is None:
            labelfile_path = params["labelfile"]
            freesurfer_subject_directory = op.dirname(op.dirname(op.dirname(labelfile_path)))

        # Read correspondence sheet
        if 'labelfile' in params.keys():
            self.read_correspondence_sheet(correspondence_sheet=params['labelfile'])
        else:
            self.read_correspondence_sheet()
            
        self.process_correspondence_sheet(update=params['update_elec_table'])

        # Create electrode table and all necessary variables
        self.create_electrode_groups()
        self.create_electrode_table()

        # Create neural data acquisitions
        self.create_electrode_table_regions()
        self.format_data()

        # Other acquisitions
        if params.get('analog'):
            #self.create_analog_acquisitions(params["analog"])
            for ana in params['analog']:

                if "stores" in ana.keys():
                    store = ana.pop("stores")
                    ana["store"] = store

                if "externalize" in ana.keys():
                    
                    write_to_file = ana.pop("externalize")
                    ana["write_to_file"] = write_to_file
                    
                if "comment" in ana.keys():
                    comments = ana.pop("comment")
                    ana["comments"] = comments

                if "units" in ana.keys():
                    unit = ana.pop("units")
                    ana["unit"] = unit

            self.create_analog_acquisitions(params["analog"])

        # Add TTLs
        if "digital" in params.keys():
            for dig in params["digital"]:
                self.create_digital_acquisition(**dig)

        # Output filename
        if params.get('output'):
            nwbfile_fname = params['output']
        else:
            nwbfile_fname, _ = os.path.splitext(params['block'])
            nwbfile_fname = nwbfile_fname + '.nwb'
            
        # Check if the output directory exists and create if not
        output_dir = os.path.split(nwbfile_fname)[0]        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.write_nwb(nwbfile_fname)


def batch_file_process(batch_excel_file,create_path=False):

    paramsdir,_ = os.path.splitext(batch_excel_file)
    paramsdir += '_params'
    if not os.path.isdir(paramsdir):
        os.mkdir(paramsdir)

    xlsx = pd.ExcelFile(batch_excel_file, engine="openpyxl")
    sheet_names = xlsx.sheet_names

    # Directory to hold all the params
    if "blocks" not in sheet_names:
        raise RuntimeError("No sheet named 'blocks' in excel file")
        return None

    df = pd.read_excel(batch_excel_file,sheet_name='blocks',engine='openpyxl')
    colnames = list(df.columns)

    # If there's a variables sheet then load that in
    df_vars = {}
    if "variables" in sheet_names:
        vars_df = pd.read_excel(batch_excel_file, sheet_name='variables', header=None, engine='openpyxl')
        vars_df = vars_df.astype(str)
        if not vars_df.empty:
            var_names = vars_df[0].to_list()
            var_vals = vars_df[1].to_list()
            df_vars = {var_names[ii]: var_vals[ii] for ii in range(len(var_names))}

    for idx, row in df.iterrows():

        try:

            row_dict = row.dropna().astype(str).to_dict()
            row_dict['create_path'] = create_path

            # Fill in any variables
            for var_key in df_vars.keys():
                for var_key2 in df_vars.keys():
                    var_val = df_vars[var_key2]
                    if isinstance(var_val,str):
                        var_val = var_val.replace('${' + var_key + '}', df_vars[var_key])
                        df_vars[var_key2] = var_val
            for var_key in df_vars.keys():
                for row_key in row_dict.keys():
                    row_val = row_dict[row_key]
                    if isinstance(row_val, str):
                        row_val = row_val.replace('${' + var_key + '}', df_vars[var_key])
                        row_dict[row_key] = row_val

            # replace block_id with block if necessary
            if 'block' not in row_dict.keys():
                try:
                    row_dict['block'] = df_vars['block_path'] + '/' + row_dict['block_id']
                    row_dict.pop('block_id')
                except ValueError:
                    print(
                        'blocks sheet of batch file needs to have either "block" (with the complete path to the block) or "block_id" (with the path relative to "block_path" from the variables sheet).')

            # Most important info
            blockfile = os.path.basename(row_dict['block'])

            if 'experimenter' in row_dict.keys(): row_dict['experimenter'] = row_dict['experimenter'].split(',')

            # Neurodata
            if 'neurodata' in row_dict.keys():
                tmp = row_dict['neurodata'].split(',')
                row_dict['neurodata'] = [x.rstrip().lstrip() for x in tmp]
                #row_dict['neurodata'] = row_dict['neurodata'].split(',')

            # Analog channels
            analog_prefixes = []
            for cname in colnames:
                if cname.startswith("analog"):
                    ana_cname = cname.split("_")[0]
                    if ana_cname not in analog_prefixes:
                        analog_prefixes.append(ana_cname)

            fields2add = ['name', 'store', 'channels', 'description', 'comments', 'externalize', 'file']
            analist = []
            for a in analog_prefixes:
                # If name doesn't exist, then it isn't there
                ana_name = a + '_name'
                if ana_name not in row_dict.keys():
                    continue
                else:
                    new_ana = {}
                    for f in fields2add:
                        field2find = a + '_' + f
                        if field2find in row_dict.keys():
                            if 'externalize' in field2find:
                                new_ana[f] = bool(int(float(row_dict[field2find])))
                            else:
                                new_ana[f] = row_dict[field2find]
                            row_dict.pop(field2find)

                if 'channels' in new_ana.keys():
                    new_ana['channels'] = [int(float(x)) for x in new_ana.get('channels').split(',')]

                analist.append(new_ana)

            # Digital channels
            digital_prefixes = []
            for cname in colnames:
                if cname.startswith("digital"):
                    dig_cname = cname.split("_")[0]
                    if dig_cname not in digital_prefixes:
                        digital_prefixes.append(dig_cname)
            
            fields2add = ['name', 'stores', 'description', 'comments']
            diglist = []
            for d in digital_prefixes:
                dig_name = d + '_name'
                if dig_name not in row_dict.keys():
                    continue

                new_dig = {}
                for f in fields2add:
                    field2find = d + '_' + f
                    if field2find in row_dict.keys():
                        new_dig[f] = row_dict[field2find]
                        row_dict.pop(field2find)

                new_dig['stores'] = new_dig['stores'].split(',')
                diglist.append(new_dig)

            # Define output Path
            if 'output' not in row_dict.keys():
                
                if 'subject_id_bids' in df_vars.keys():
                    outputName = df_vars['subject_id_bids']
                elif'subject' in df_vars.keys():
                    outputName = df_vars['subject_id']
                
                if 'outputName' in locals():
                    if 'session_id' in row_dict.keys(): 
                        outputName = outputName + '_ses-{:02d}'.format(int(row_dict['session_id']))
                    if 'acq' in row_dict.keys():
                        outputName = outputName + '_acq-' + row_dict['acq']
                    if 'task' in row_dict.keys():
                        outputName = outputName + '_task-' + row_dict['task']
                    if 'run' in row_dict.keys():
                        outputName = '{:s}_run-{:02d}'.format(outputName, 
                                                              int(row_dict['run']))
                else:
                    outputName, _ = os.path.splitext(blockfile)
                    outputName = outputName + '.nwb'
                    
                if 'output_path' in df_vars.keys():
                    outputName = df_vars['output_path'] + '/' + outputName
                else:
                    row_dict['output'] = outputName
                    
                if not outputName.endswith('_ieeg.nwb'):
                    outputName += '_ieeg.nwb'
                    
                row_dict['output'] = outputName
                
            # Create output path for external files
            for ana in analist:
           
                if ana['externalize']:
                    
                    if 'file' in ana.keys():             
                        file = ana.pop('file')
                    else:
                        file = 'tsv'
                        
                    if file == 'tsv': 
                        ana['file'] = row_dict['output'].replace('_ieeg.nwb', '_physio.tsv.gz')
                    if file == 'wav':
                        ana['file'] = row_dict['output'].replace('_ieeg.nwb', '_{:s}.wav'.format(ana['name']))

            # add subject level data if necessary
            if 'labelfile' not in row_dict.keys():
                if 'corr_sheet' in df_vars.keys():
                    row_dict['labelfile'] = df_vars['corr_sheet']
                else:
                    print('electrode correspondence sheet could not be found. Either the main sheet or the variables sheet of the batch file should have a field called "corr_sheet"')
            if 'subject_id' not in row_dict.keys():
                if 'subject_id_bids' in df_vars.keys():
                    row_dict['subject_id'] = df_vars['subject_id_bids']
                else:
                    print('subject ID could not be found. Either the main sheet or the variables sheet of the batch file need to have a field called "subject_id"')
            if 'sex' not in row_dict.keys():
                if 'sex' in df_vars.keys():
                    row_dict['sex'] = df_vars['sex']
                else:
                    print('subject ID could not be found. Either the main sheet or the variables sheet of the batch file need to have a field called "sex"')
            if 'age' not in row_dict.keys():
                if 'age' in df_vars.keys():
                    row_dict['age'] = df_vars['age']
                else:
                    print('subject ID could not be found. Either the main sheet or the variables sheet of the batch file need to have a field called "age"')
            if 'subject_description' not in row_dict.keys():
                if 'subject_description' in df_vars.keys():
                    row_dict['subject_description'] = df_vars['subject_description']
                else:
                    print('subject description could not be found. Either the main sheet or the variables sheet of the batch file should have a field called "subject_description"')
            if 'update_elec_table' not in row_dict.keys():
                if 'update_elec_table' in df_vars.keys():
                    row_dict['update_elec_table'] = bool(int(df_vars['update_elec_table']))
                else:
                    row_dict['update_elec_table'] = True
            
            # Add digital analog
            subfields = zip(['analog', 'digital'], [analist, diglist])
            for k,v in subfields:
                if len(v) > 0:
                    row_dict[k] = v
                    
            # Add freesurfer directory and subject directory
            if 'freesurfer_subject_directory' not in row_dict.keys():
                if 'freesurfer_subject_directory' not in df_vars.keys():
                    print('Freesurfer directory has not been set!')
                else:
                    row_dict['freesurfer_subject_directory'] = df_vars['freesurfer_subject_directory']
                    
            if 'freesurfer_subject_id' not in row_dict.keys():
                if 'subject_id' not in df_vars.keys():
                    print('Subject label not defined, freesurfer subject cannot be set!')
                else:
                    row_dict['freesurfer_subject_id'] = df_vars['subject_id']
                    
            # Create a yml file
            outfile = paramsdir + os.sep + '%s.yml' % blockfile
            with open(outfile, 'w') as file:
                yaml.dump(row_dict, file, sort_keys=False)

            print('*' * 100)
            print('Processing %s' % blockfile)
            print('Params written to %s' % outfile)
            print('*' * 100)

            # Parse
            inwb = IEEG2NWB()
            inwb.parse_params(row_dict)

        except Exception as e:
            print('*' * 200)
            print(e)
            print('*' * 10)
            print('Error processing %s. Skipping to next file' % blockfile)
            print('*' * 200)



def cmnd_line_parser():
    # Create parser
    from .messages import example_usage, additional_notes
    parser = argparse.ArgumentParser(description="Convert a file to NWB format",
                                    epilog=example_usage + additional_notes,
                                    formatter_class=argparse.RawDescriptionHelpFormatter
                                    )
    parser.add_argument('--batch', required=False, help='excel file for batch conversion',dest='batch_file',default=None)
    parser.add_argument('--gui', required=False, help='launch the IEEG2NWB gui',dest='gui',action='store_true')
    parser.add_argument('--params','-p', required=False, help='json or yml params file to use instead of command line arguments',dest='params_file',default=None)

    args = parser.parse_args()

    # Setup params
    params = vars(args)

    # Check if params or block path is passed in
    if params['params_file'] is None and params['block'] is None and params['batch_file'] is None and params['gui'] == False:
        print('Error! Have to specify params file OR block with recorded dat')
        parser.print_help()
        sys.exit(2)


    if params['gui']:
        from qtpy.QtWidgets import QApplication
        from .gui import GUI
        app = QApplication([])
        ex = GUI()
        ex.show()
        sys.exit(app.exec_())


    if params['params_file'] is not None and op.isfile(params['params_file']):
        with open(params['params_file']) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)

        converter = IEEG2NWB()
        converter.parse_params(params)

    # elif params['batch_file'] != None and os.path.isfile(params['batch_file']):
    #     batch_file_process(params['batch_file'],create_path=params['create_path'])



