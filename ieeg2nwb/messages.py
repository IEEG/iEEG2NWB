example_usage = """

##########################
Examples:
##########################

# Import an edf file and create additional timeseries data from specified analog channels. The output file will be called "visloc.nwb"
ieeg2nwb --block_path VisualLocalizer.edf --labelfile NS001/elec_recon/correspondence_sheet.xlsx --analog "Audio:Audio recorded during experiment:DC1" "TTLs:TTL pulses from MMB triggerbox:DC2" --output_path visloc.nwb

# Import a TDT block as the recording. Create additional timeseries data (only channels 1 + 2 of Wav5 store) as well as discrete timepoints from TTL pulses. Neural data is located in the EEG3 and EEG4 stores of TDT recording
ieeg2nwb -b B80_cceps -l NS001/elec_recon/correspondence_sheet.xlsx -a "Audio:Audio recorded during experiment:Wav5:1,2" -d "TTLs:TTL pulses from DB25 port:PtC4" -n EEG3,EEG4

# Import a HBML formatted ecog.mat file. Info file/table generated during preprocessing passed in instead of correspondence sheet. Extracted events also passed in. Additional analog data added as a timeseries
ieeg2nwb -b EntrainSounds.mat -l EntrainSounds_info.xlsx --events EntrainSounds_events.xlsx -a "emg:EMG data recorded simultaneously:emg" -o entrainsounds.nwb
"""


additional_notes = """


##########################
Notes:
##########################
- An 'info' file, generated from preprocessing steps, can be passed in instead of a correspondence sheet. Must be as a spreadsheet
- The "--digital" argument currently only applies to TDT data
- All events file must contain a specific format for their columns.
    - One column MUST be named "event_times" and is the time (in seconds) that the given event occurred
    - All other columns must have a column name and column description separated by a colon. Below is an example
        StimType : Type of stimulus presented to participant


"""
