# TODO:
#   - ieeg2nwb
#       - TDT
#       - EDf
#       - Natus
#   - Write correspondence sheet (correspondence)
#   - surfs
#       - create_indiv_mapping
#       - pial_to_inflated
#       - elec_to_parc
#       - sub_to_fsaverage
#   - ptd (get_ptd_index)
#   - io (read_ielvis)

subid = "NS162_02_copy"
subjects_dir = "/Applications/freesurfer/7.2.0/subjects"
n_jobs = 4

# %% create_indiv_mapping
from ieeg2nwb.surfs import create_indiv_mapping

# Standard
create_indiv_mapping(subid, subjects_dir=subjects_dir, n_jobs=n_jobs)

# New non-listed parcellation
create_indiv_mapping(subid, subjects_dir=subjects_dir, parc="oasis.chubs",n_jobs=n_jobs)


# %% pial_to_inflated
from ieeg2nwb.channels import pial_to_inflated

inf_coords = pial_to_inflated(subid, subjects_dir=subjects_dir, n_jobs=3, write_to_file=True)

# %% elec_to_parc
from ieeg2nwb.channels import elec_to_parc

dk_parc = elec_to_parc(subid, subjects_dir=subjects_dir, parc="DK", write_to_file=True, n_jobs=n_jobs)

oc_parc = elec_to_parc(subid, subjects_dir=subjects_dir, parc=["OC","oasis.chubs"], write_to_file=True, n_jobs=n_jobs)


# %% sub_to_fsaverage

from ieeg2nwb.channels import sub_to_fsaverage

avg_coords = sub_to_fsaverage(subid, subjects_dir=subjects_dir, n_jobs=n_jobs)

# %% get_ptd_index

from ieeg2nwb.ptd import get_ptd_index

ptd = get_ptd_index(subid, subjects_dir=subjects_dir, write_to_file=True)

# %% read_ielvis

from ieeg2nwb.fileio import read_ielvis

ielvis_data = read_ielvis(subid, subjects_dir=subjects_dir, full=True)


# %% Run ieeg2nwb

from converter import IEEG2NWB

subjects_dir = "/Applications/freesurfer/7.2.0/subjects"
subject_id = "NS135"
raw_data_dir = "/Users/noahmarkowitz/Documents/HBML/NWB_conversion/sample_raw_data/NS135/tdt/512ch_reconly_all-180117-130258/B1_rest"
output_nwb_file = "/Users/noahmarkowitz/Documents/HBML/NWB_conversion/sample_nwb_files/NS135_Rest_B1.nwb"

# Instantiates. NWBFile object at converter.nwbfile
converter = IEEG2NWB()

# Load raw data into memory at converter.raw_data and converter.raw_data_type
# Also create device object for amplifier at converter.amplifier
converter.read_raw_data(raw_data_dir, create_device=True)

# Create Subject object at converter.nwbfile.subject
converter.create_subject(subject_id="NS135", sex="M", age=25)

# Have converter obtain variables internally for getting data later
converter.set_freesurfer(subject_id=subject_id, subject_dir=subjects_dir)

# Read raw correspondence sheet, save at converter.correspondence_table
# Also creates converter.channel_labels dict to match neural channel with label
converter.read_correspondence_sheet()

# Merge correspondenc sheet with ielvis data and put back into converter.correspondence_table
# Also create dict for definitions for electable columns
converter.process_correspondence_sheet()

# Create groups based on naming of electrodes, stored directly in nwbfile as well as at converter.electrode_groups
converter.create_electrode_groups()

# Creates the ElectrodeTable object to store in nwbfile, stored in converter.nwbfile.electrodes
converter.create_electrode_table()

# Create table regions to tell which rows of electrodes corresponds to which time-series (ex: ieeg vs cranial)
# Stored in converter.electable_regions 
converter.create_electrode_table_regions()

# Creates neural data ElectricalSeries objects using converter._create_electricalseries()
# by iterating over converter.electable_regions
# Does not create TTL timeseries or Audio etc., only those contained in correspondence sheet
converter.format_data()

# Can create additional analog TimeSeries using  converter._create_timeseries

# Write the NWBFile object to disk
converter.write_nwb(output_nwb_file)



# %% Using params


params = {
'digital': [{
    'description': 'TTL Pulses',
    'name': 'TTL', 
    'stores': ['PtC2', 'PtC4', 'PtC6']
}],
'experiment_description': 'n-back task of black and white faces, text, tools, houses, body parts and patterns',
'output': '/Users/noahmarkowitz/Documents/HBML/NWB_conversion/sample_nwb_files/sub-NS162_ses-implant02_task-visloc_acq-classic1_ieeg.nwb',
'session_id': 'implant01',
'session_description': 'Training, Classic 1, Classic 2',
"block": "/Users/noahmarkowitz/Documents/HBML/NWB_conversion/sample_raw_data/NS162_02/B1_VisualLocalizer",
#"labelfile": "optional now",
"analog": [
    {
        "name": "audio",
        "description": "audio stuff",
        "store": "Wav5",
        "channels": [1],
        "comments": "blah"
    },
    {
        "name": "mic",
        "description": "microphone",
        "store": "Wav5",
        "channels": [2,3]
    }
],
"subject": {
    "age": 21,
    "subject_id": "Noah",
    "sex": "M"
},
"freesurfer_subject_id": "NS162_02",
"freesurfer_subject_directory": "/Applications/freesurfer/7.2.0/subjects"
}

converter = IEEG2NWB()
converter.parse_params(params)
