#!/usr/bin/env python

# IEEG2NWB.py
# A python-based script meant to convert edf, TDT and ecog.mat formatted data into an
# NWB file
#
# Noah Markowitz
# Human Brain Mapping Lab
# North Shore University Hospital
# April 2021

# Imports
import numpy as np
import pandas as pd
import os
import os.path as op
import glob
import re
import tdt
from datetime import datetime
from dateutil.tz import tzlocal
from datetime import timedelta
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb.base import TimeSeries
from pynwb.ecephys import ElectricalSeries
from pynwb.epoch import TimeIntervals
import json
import yaml
import argparse
import sys
from colorama import Back, Style
from pymatreader import read_mat
import h5py
#from messages import example_usage, additional_notes
from ieeg2nwb.utils import load_nwb_settings
from ieeg2nwb.tdt import _get_tdt_store, get_tdt_data
from ieeg2nwb.io import read_ielvis

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
        from pynwb.file import Subject
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
        label_col_name = [col for col in corr_columns if col.lower() == 'label'][0]
        corr_sheet.rename(columns={label_col_name: 'label'}, inplace=True)
        corr_sheet = corr_sheet[~corr_sheet["label"].isin(['[]', ''])]

        # Rename column has relevant channel numbering
        r = re.compile(ch_col_name, re.IGNORECASE)
        chn_col_name = list(filter(r.match, corr_columns))[0]
        corr_sheet.rename(columns={chn_col_name: 'channel'}, inplace=True)

        corr_sheet = corr_sheet.reset_index(drop=True)

        self.channel_labels = {
            "label": corr_sheet.loc[:,["label"]].values.flatten(),
            "channel": corr_sheet.loc[:,["channel"]].values.flatten() - 1
        }

        self.correspondence_table = corr_sheet.drop("channel", axis=1)

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

    def process_correspondence_sheet(self, extra_columns=None):
        """Add info to correspondence sheet and prepare it to become ElectrodeTable"""

        # Parameters for the columns of the electrode table
        cols_for_table = self._params["electrode_table"]["columns"]

        corr_sheet = self.correspondence_table

        subject_id = self.freesurfer_subject_id
        subjects_dir = self.freesurfer_subject_dir

        # Get ielvis data
        ielvis_df = read_ielvis(subject=subject_id, subjects_dir=subjects_dir, squeeze=False)

        # Columns that ielvis can contribute to
        ielvis_columns = {
            "label": "label",
            "LEPTO_x": "lepto_x",
            "LEPTO_y": "lepto_y",
            "LEPTO_z": "lepto_z",
            "FSAVERAGE_x": "x",
            "FSAVERAGE_y": "y",
            "FSAVERAGE_z": "z",
            "PTD": "ptd",
            "desikan-killiany_atlas": "desikan_killiany_atlas",
            "destrieux_atlas": "destrieux_atlas",
            "location": "location",
            "hem": "hem",
            "yeo7_atlas": "yeo7_atlas",
            "yeo17_atlas": "yeo17_atlas"
        }

        # If some columns are missing then run the commands to generate them
        missing_ielvis_cols = [col for col in ielvis_columns.keys() if col not in ielvis_df.columns]
        if len(missing_ielvis_cols) > 0:
            if any(c for c in ["PTD", "location"] if c in missing_ielvis_cols):
                from .ptd import get_ptd_index
                get_ptd_index(subject=subject_id, subjects_dir=subjects_dir)
            if any(c for c in ["desikan-killiany_atlas", "destrieux_atlas", "yeo7_atlas", "yeo17_atlas"] if
                   c in missing_ielvis_cols):
                from .surfs import elec_to_parc
                elec_to_parc(subject=subject_id, subjects_dir=subjects_dir)
            ielvis_df = read_ielvis(subject=subject_id, subjects_dir=subjects_dir, squeeze=False)

        # Remove and rename some columns
        ielvis_df = ielvis_df[list(ielvis_columns.keys())].rename(columns=ielvis_columns)

        # Variables to use later
        dynamic_columns = [] # Column definitions for ElectrodeTable, list of dictionaries
        cols2rename = {} # Columns to rename for formatting
        cols2keep = [] # Columns to not drop
        #dfcols = list(corr_sheet.columns)
        missing_cols = [] # Columns that are missing

        # Go through columns
        for c in cols_for_table.keys():
            col_settings = cols_for_table[c]
            is_required = col_settings['required']
            in_ielvis = False

            # Find the column
            colfound = []
            if 'search' in col_settings.keys():
                r = re.compile(col_settings['search'], re.IGNORECASE)
                colfound = list(filter(r.match, corr_sheet.columns))

            # Check first if column can be retrieved from ielvis
            if c in ielvis_df.columns and c != "label":
                corr_sheet = corr_sheet.merge(ielvis_df[["label", c]], on="label", how="left")
                corr_sheet[c] = corr_sheet[c].fillna(col_settings['default'])
                in_ielvis = True

            # IF column is found then give it the right name and fill blank cells
            elif len(colfound) > 0:
                cols2rename[colfound[0]] = col_settings['title']
                if 'default' in col_settings.keys():
                    corr_sheet[colfound[0]] = corr_sheet[colfound[0]].fillna(col_settings['default'])

            # Column is required, absent and has a default
            elif (len(colfound) == 0) & is_required & ('default' in col_settings.keys()):
                corr_sheet[col_settings['title']] = col_settings['default']
                missing_cols.append(col_settings['title'])

            # Column is required, absent
            elif len(colfound) == 0 & is_required:
                corr_sheet[col_settings['title']] = 'None'
                missing_cols.append(col_settings['title'])

            # Make into the correct data type
            if ('type' in col_settings.keys()) & (len(colfound) != 0):
                try:
                    corr_sheet[colfound[0]] = corr_sheet[colfound[0]].astype(col_settings['type'])
                except ValueError:
                    corr_sheet[colfound[0]].replace("None", "0").astype(col_settings['type'])

            # Append dynamic_columns and mark this column as being kept
            if (len(colfound) > 0) | is_required | in_ielvis:
                cols2keep.append(col_settings['title'])
                dynamic_columns.append({'name': col_settings['title'], 'description': col_settings['description']})

            # Is column still missing and required?
            if (len(colfound) == 0):
                missing_cols.append(col_settings['title'])

        # Keep only the needed columns in dataframe
        corr_sheet = corr_sheet.rename(columns=cols2rename).loc[:, cols2keep]

        # Store column definitions for creating ElectrodeTable
        self.electable_columns = dynamic_columns

        # Store the processed correspondence sheet
        self.correspondence_table = corr_sheet

    def create_electrode_table(self):
        """Create the ElectrodeTable that is a DynamicTable object."""
        from pynwb.file import ElectrodeTable
        electable = ElectrodeTable().from_dataframe(
            self.correspondence_table,
            self._params["electrode_table"]["name"],
            table_description=self._params["electrode_table"]["description"],
            columns=self.electable_columns
        )
        self.nwbfile.electrodes = electable

    def create_electrode_table_regions(self):
        """Create the regions for the ElectrodeTable."""

        table_regions = {}

        intracranial_specs = self._params["intracranial_specs"]

        df = self.correspondence_table
        spec_indices = {'ieeg': []}
        for idx, row in df.iterrows():
            if row.spec.lower() in intracranial_specs:
                spec_indices['ieeg'].append(idx)
            else:
                if row.spec.lower() not in spec_indices:
                    spec_indices[row.spec.lower()] = []

                spec_indices[row.spec.lower()].append(idx)

        for spec in spec_indices.keys():
            table_regions[spec] = self.nwbfile.create_electrode_table_region(
                region=spec_indices[spec],
                description=f"electrodes recording {spec} data",
                name="electrodes"
            )
        self.electable_regions = table_regions

    def read_raw_data(self, raw_data_files, create_device=True):
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
        elif op.isdir(raw_data_files):
            dir_contents = os.listdir(raw_data_files)
            file_extensions = {op.splitext(f)[-1] for f in dir_contents if op.isfile(os.path.join(raw_data_files, f))}
            if ".tev" in file_extensions or ".sev" in file_extensions:
                self.raw_data_type = 'tdt'
                self.raw_data = tdt.read_block(raw_data_files)
                if "info" in self.raw_data.keys():
                    start_time = self.raw_data.info.start_date
                amplifier = "tdt"
            elif ".erd" in file_extensions:
                from nwreader import read_erd
                self.raw_data_type = 'xltek'
                self.raw_data = read_erd(raw_data_files, use_dask=True, convert=True, pad_discont=True)
                start_time = self.raw_data.attrs["creation_time"]
                amplifier = "xltek"
        else:
            raise ValueError("File type not recognized. Must be edf, tdt, or xltek")

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
        from .tdt import read_tdt_ttls

        # Get the timestamps and stores they're from
        event_times_df = read_tdt_ttls(self.raw_data, stores)

        # Get rid of any timestamps occurring at t=0
        invalid_timestamps = event_times_df["time"] == 0
        if np.any(invalid_timestamps):
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
        from ndx_events import TTLs
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


    def _create_timeseries(self, name, data, fs, description=None, comments=None, unit="volts", write_to_wav=False):
        """Create the TimeSeries object."""
        if write_to_wav:
            if self.output_file:
                wav_fname = op.splitext(self.output_file)[0] + f"_{name}.wav"
            else:
                wav_fname = op.splitext(self.raw_data_file)[0] + f"_{name}.wav"

            import soundfile as sf
            dim1, dim2 = data.shape
            if dim1 < dim2:
                data = data.T
            print(f"---> Writing {name} to {wav_fname}")
            sf.write(wav_fname, data, round(fs))


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
        for acq_name in self.electable_regions.keys():

            # # Figure out which channels are in the region
            # electable_rows = self.electable_regions[region].data
            # elec_names = self.correspondence_table.loc[electable_rows, "label"].to_list()
            # df = pd.DataFrame(self.channel_labels)
            # elec_indices = df[df["label"].isin(elec_names)].index
            # elec_channels = df.loc[elec_indices,"channel"].to_list()
            region = self.electable_regions[acq_name]
            elec_channels = region.data

            self._create_electricalseries(
                acq_name,
                data=eeg_array[elec_channels, :].T,
                fs=fs,
                description=f"data recorded from {region} electrodes",
                electrodes=self.electable_regions[region]
            )

    def _edf_create_analog_acquisition(self, analog_stores):
        """For EDF data create extra analog acquisitions."""
        for eac in analog_stores:

            # Get all stores
            # analog_array = self.raw_data.get_data()[eac["channels"], :]
            analog_array = self.raw_data.get_data(eac["channels"])
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
        """For TDT data create extra analog acquisitions."""
        for eac in analog_stores:

            # Get all stores
            analog_array, fs = get_tdt_data(self.raw_data, eac["store"])

            if "channel" not in eac.keys():
                eac["channel"] = None

            # Take only the data from the channels that are in the correspondence table
            if eac["channel"] is not None:
                analog_array = analog_array[eac["channel"], :]

            if unit in eac.keys():
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

    def _tdt_format_data(self, eeg_chans=None):
        """Take TDT raw data and format it to be stored in NWB."""

        # The default is to look for the EEG stores listed here
        tdt_eeg_channels = self._params["tdt_neuro_channels"]

        if eeg_chans is None:
            for streams in tdt_eeg_channels:
                eeg_array, fs = get_tdt_data(self.raw_data, streams, ignore_missing=True)
                if eeg_array is not None:
                    break

        else:
            if not isinstance(eeg_chans, list):
                eeg_chans = list(eeg_chans)
            eeg_array, fs = get_tdt_data(self.raw_data, eeg_chans, ignore_missing=False)

        # Create the ElectricalSeries object for each table region
        for acq_name in self.electable_regions.keys():

            # Figure out which channels are in the region
            # electable_rows = self.electable_regions[region].data
            # elec_names = self.correspondence_table.loc[electable_rows, "label"].to_list()
            # df = pd.DataFrame(self.channel_labels)
            # elec_indices = df[df["label"].isin(elec_names)].index
            # elec_channels = df.loc[elec_indices,"channel"].to_list()

            region = self.electable_regions[acq_name]
            elec_channels = region.data

            self._create_electricalseries(
                region,
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
        pass

    def write_nwb(self, nwb_file=None):
        """Write the NWB file."""
        from pynwb import NWBHDF5IO
        if nwb_file is None and self.output_file is None:
            raise ValueError("Output file must be provided")
        elif nwb_file is None:
            nwb_file = self.output_file

        # Make sure the filename ends with .nwb
        if not nwb_file.endswith('.nwb'):
            nwb_file += '.nwb'

        # If available annotations haven't been added then add them
        if len(self.annotations) > 0 and "annotations" not in self.nwbfile.acquisition.keys():
            self.add_annotations()

        # Write the NWB file
        with NWBHDF5IO(nwb_file, 'w') as io:
            io.write(self.nwbfile, link_data=False)

        # Close the h5 files
        if len(self.hdf5_files) > 0:
            for fname, handle in self.hdf5_files:
                handle.close()
                os.remove(fname)

        # Close EDF file
        if self.raw_data_type == "edf":
            self.raw_data.close()

    def parse_params(self, params):
        """Run the entire converter given params."""

        # Read data
        print('-----> Reading input: %s' % params['block'])
        eeg_chans = params.get('neurodata')
        self.read_raw_data(params['block'], make_device=True)

        # Get subject specific info and create subject
        subinfo = ['subject_id', 'sex', 'age', 'subject_description']
        subdict = {}
        for s in subinfo:
            if s in params.keys():
                subdict[s] = str(params[s])

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

        # Read correspondence sheet
        self.read_correspondence_sheet()
        self.process_correspondence_sheet()

        # Create electrode table and all necessary variables
        self.create_electrode_groups()
        self.create_electrode_table()

        # Create neural data acquisitions
        self.create_electrode_table_regions()
        self.format_data()

        # Other acquisitions
        if params.get('analog'):
            self.create_analog_acquisitions(params["analog"])
            for ana in params['analog']:

                if "stores" in ana.keys():
                    store = ana.pop("stores")
                    ana["store"] = store

                if "externalize" in ana.keys():
                    write_to_wav = ana.pop("externalize")
                    ana["write_to_wav"] = write_to_wav

                if "write_to_wav" in ana.keys():
                    tmp = str(ana['externalize']).upper()
                    ana["write_to_wav"] = True if tmp in ['1','TRUE','YES','Y'] else False

                if "comment" in ana.keys():
                    comments = ana.pop("comment")
                    ana["comments"] = comments

                if "units" in ana.keys():
                    unit = ana.pop("units")
                    ana["unit"] = unit

            self.create_analog_acquisitions(params["analog"])

        # Add for TTLs



        # Output filename
        if params.get('output'):
            nwbfile_fname = params['output']
        else:
            nwbfile_fname, _ = os.path.splitext(params['block'])
            nwbfile_fname = nwbfile_fname + '.nwb'

        self.write_nwb(nwbfile_fname)


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



if __name__ == "__main__":
    # Parse command-line arguments
    subjects_dir = "/Applications/freesurfer/7.2.0/subjects"
    subject_id = "NS162_02"
    raw_data_dir = "/Users/noahmarkowitz/Documents/HBML/NWB_conversion/sample_raw_data/NS162_02/B1_VisualLocalizer"
    output_nwb_file = "/Users/noahmarkowitz/Documents/HBML/NWB_conversion/sample_nwb_files/B1_VisualLocalizer.nwb"
    from ieeg2nwb.ieeg2nwb_2 import IEEG2NWB

    converter = IEEG2NWB()
    converter.read_raw_data(raw_data_dir)
    converter.create_subject(subject_id="NS162_02", sex="M", age=25)
    converter.set_freesurfer(subject_id=subject_id, subject_dir=subjects_dir)
    converter.read_correspondence_sheet()
    converter.process_correspondence_sheet()
    converter.create_electrode_groups()
    converter.create_electrode_table()
    converter.create_electrode_table_regions()
    converter.format_data()
    converter.write_nwb(output_nwb_file)


