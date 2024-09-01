#!/usr/bin/env python
import os.path as op
import os
import numpy as np
import pandas as pd
from pynwb.epoch import TimeIntervals
import time
from tqdm import tqdm
import yaml
import json

def create_chunk_indices(nsamples, step=10000):
    chunks = []
    next_start = -1
    next_end = -1
    while next_end != nsamples:
        next_start = next_end + 1
        next_end = next_start + step
        if next_end >= nsamples:
            next_end = nsamples

        chunks.append((next_start, next_end))

    return chunks


# Generator function that loads the data by chunk loading
def iter_by_chan_edf(edf_obj, chunks, chanindices):
    # chn_idx = list(range(256))
    nchans = len(chanindices)
    n = 0
    start = 0
    edf = edf_obj  # read_raw_edf(edf_file)
    # chanindices = tqdm(chanindices,desc='Writing timeseries',ascii=False,ncols=100)
    for chn in chanindices:
        end = time.time()

        if n != 0:
            total = end - start
            print('nLast Read/Write took %0.2f minutes\n' % (total / 60))

        n += 1
        start = time.time()
        # print('Reading and writing channel %03d/%03d' %(n,nchans))
        arr = np.empty((1, 0)).astype('float32')
        # chunks = tqdm(chunks,ncols = 100,)
        for c in tqdm(chunks, ncols=120, desc='Channel %03d/%03d' % (n, nchans), leave=True, position=0):
            # chunks.set_description()
            piece = edf.get_data(chn, c[0], c[1] + 1)  # The last sample is exclusive so need to correct for it
            arr = np.append(arr, piece, axis=1)
            del piece

        arr = arr.astype('float32').T
        print(arr.shape)
        # print('Now writing to file')
        yield arr.flatten()

    # edf.close()
    return


# Import the json NWB file
def load_nwb_settings():
    # pdir = os.path.dirname(
    #     os.path.dirname(
    #         os.path.dirname(
    #             os.path.dirname(
    #                 os.path.dirname(__file__)
    #             )
    #         )
    #     )
    # )
    pdir = os.path.dirname(__file__)
    settings_file = pdir + os.sep + 'nwb_settings.json'
    json_str = ''
    with open(settings_file) as file:
        Lines = file.readlines()
        for line in Lines:
            this_line = line.lstrip()
            if not this_line.startswith('//'):
                json_str += this_line

    return json.loads(json_str)



# Function to easily compress data
def compress_data(data):
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    return H5DataIO(
        data=data,
        compression='gzip',
        compression_opts=4,
        shuffle=True
    )

def inspectNwb(nwbfile) -> dict:
    r"""
      Inspect and return all instances of a TimeSeries class and subclass contained in the file.

      Parameters
      ----------
      nwbfile : NWBFile
          An NWBFile object. Created after opening a .nwb file using pynwb.

      Returns
      -------
      dict
          Series of key-value pairs with useful information about containers within the NWBFile:
              - timeseries: pandas.DataFrame containing information on all TimeSeries classes and subclasses contained in the file including:
                  - Name of container
                  - Class
                  - Description
                  - Sampling rate (if timestamps used instead then the word "timestamps" will be in place)
                  - Size and dimensions of data
                  - Comments
                  - Object ID (to easily load the TimeSeries after)
              - devices: pandas.DataFrame of Device object.
              - subject: dictionary of subject information.
              - elecs: pandas.DataFrame of ElectrodeTable object (if available).

      Examples
      --------
      Run the function to see all instances of a TimeSeries in the file:

      >>> from pynwb import NWBHDF5IO
      >>> io = NWBHDF5IO('myfile.nwb', mode='r', load_namespaces=True)
      >>> nwbfile = io.read()
      >>> nwbinfo = inspectNwb(nwbfile)

      Load a specific TimeSeries named "ieeg":

      >>> ts_df = nwbinfo['timeseries']
      >>> ieeg_id = ts_df.loc[ts_df['name'] == 'ieeg', 'id'].values[0]
      >>> ieeg = nwbfile.objects.get(ieeg_id)
      """

    from pynwb import NWBFile
    from pynwb.base import TimeSeries
    from pynwb.file import Subject
    from pynwb.device import Device
    from hdmf.common.table import DynamicTable

    if not isinstance(nwbfile,NWBFile):
        pass

    output = {}

    # Get some key attributes
    attrs = ['session_id','experiment_description','session_description','data_collection','identifier','lab','institution','notes']
    for a in attrs:
        output[a] = getattr(nwbfile,a)

    # Construct the output pandas DataFrame for TimeSeries objects
    output['timeseries'] = pd.DataFrame({k: [] for k in ['name','type','description','fs','size','comments','id']})

    # Construct the output pandas DataFrame for Device objects
    output['devices'] = pd.DataFrame({k: [] for k in ['name','description','manufacturer','id']})

    # List of object IDs in the nwbfile
    objIds = list(nwbfile.objects.keys())

    # Loop through all object and look for TimeSeries classes and subclasses
    for objName in objIds:
        container = nwbfile.objects.get(objName)
        if isinstance(container,TimeSeries):
            acqDict = {
                'name': container.name,
                'type': container.neurodata_type,
                'description': container.description,
                'fs': container.rate if 'rate' in container.fields.keys() else 'timestamps',
                'size': container.data.shape,
                'comments': container.comments if 'comments' in container.fields.keys() else 'no comments',
                'id': objName
            }
            for k, v in acqDict.items():
                acqDict[k] = [v]
            output['timeseries'] = pd.concat((output['timeseries'], pd.DataFrame(acqDict)))
            #output['timeseries'] = output['timeseries'].append(acqDict,ignore_index=True)
        elif isinstance(container,Device):
            devDict = {
                'name': container.name,
                'description': container.description,
                'manufacturer': container.manufacturer,
                'id': objName
            }
            for k, v in devDict.items():
                devDict[k] = [v]
            output['devices'] = pd.concat((output['devices'], pd.DataFrame(devDict)))
            #output['devices'] = output['devices'].append(devDict,ignore_index=True)
        elif isinstance(container,DynamicTable) and isinstance(container.parent,NWBFile):
            output['elecs'] = nwbfile.electrodes.to_dataframe()
        elif isinstance(container,Subject):
            output['subject'] = nwbfile.subject.fields


    return output

def _get_data_directory():
    return op.join(op.dirname(__file__), 'data')

def read_aseg_csv():
    import pandas as pd
    aseg_csv = op.join(_get_data_directory(), 'aseg.csv')
    return pd.read_csv(aseg_csv)

def copy_fsaverage_data(target_fsaverage):
    import shutil
    import filecmp
    my_fsaverage = op.join(_get_data_directory(), 'fsaverage')
    # Walk through the source directory
    for dirpath, dirnames, filenames in os.walk(my_fsaverage):
        # Determine the destination path
        relative_path = os.path.relpath(dirpath, my_fsaverage)
        target_path = os.path.join(target_fsaverage, relative_path)

        # Ensure the target path exists
        os.makedirs(target_path, exist_ok=True)

        # Compare files
        for filename in filenames:
            src_file = os.path.join(dirpath, filename)
            dst_file = os.path.join(target_path, filename)

            # Copy the file if it doesn't exist in the destination or if it's different
            if not os.path.exists(dst_file) or not filecmp.cmp(src_file, dst_file, shallow=False):
                shutil.copy2(src_file, dst_file)

    fsaverage_dir = op.join(_get_data_directory(), 'fsaverage')
    if not op.isdir(fsaverage_dir):
        shutil.copytree(op.join(_get_data_directory(), 'fsaverage'), fsaverage_dir)
    return fsaverage_dir