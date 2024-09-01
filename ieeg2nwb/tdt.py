import numpy as np
import pandas as pd

# Get a TDT data store
def _get_tdt_store(tdt_data,store):
    storeTypes = ['streams', 'epocs']
    if store in tdt_data.keys():
        return tdt_data[store]
    else:
        for s in storeTypes:
            if store in tdt_data[s].keys():
                return tdt_data[s][store]

        #print('Store not found in TDT block')
        #return KeyError
        return None

raw_data_dir = "/Users/noahmarkowitz/Documents/HBML/NWB_conversion/sample_raw_data/NS162_02/B1_VisualLocalizer"


# Function to collect stores and concatenate them
def get_tdt_data(root, stores, ignore_missing=False):

    stream_list = []
    for s in stores:
        stream = _get_tdt_store(root, s)
        if stream is not None:
            stream_list.append(stream)
        else:
            if not ignore_missing:
                raise ValueError(f"Store {s} not found in TDT block")

    if len(stream_list) == 0:
        return None, None

    # Make sure they all have the same fs
    if not all(s.fs == stream_list[0].fs for s in stream_list):
        raise ValueError("All streams must have the same sampling frequency")

    # Now concatenate the data
    analog_array = stream_list[0].data
    if len(stream_list) > 1:
        for ana_store in stream_list[1:]:
            new_store_data = ana_store.data
            if new_store_data.shape[1] != analog_array.shape[1]:
                print("Stores have different data lengths. Truncating to the shortest length")
                min_samples = min([analog_array.shape[1], new_store_data.shape[1]])
                analog_array = analog_array[:, :min_samples]
                new_store_data = new_store_data[:, :min_samples]

            analog_array = np.concatenate((analog_array, new_store_data), axis=0)

    return analog_array, stream_list[0].fs


    # Now concatenate the streams


# Get TTLs from TDT epocs
def read_tdt_ttls(tdt_data,stores):
    # If stores is a string, convert it to list
    if isinstance(stores, str):
        stores = [stores]

    # Create empty dataframe
    df = pd.DataFrame({'time': [], 'stores': []})

    # Loop through all stores given
    for s in stores:
        thisStore = _get_tdt_store(tdt_data, s)

        # If return type is not None, procceed
        if thisStore != None:

            # Also make sure it has the "onset" field
            if 'onset' in thisStore.keys():
                timestamps = thisStore.onset
                for t in timestamps:
                    idx = df['time'].isin([t])
                    if idx.any():
                        df.loc[idx, 'stores'] = df.loc[idx, 'stores'] + '/' + s
                    else:
                        df = pd.concat( ( df,pd.DataFrame({'time': [t], 'stores': [s]}) ) )
                        #df = df.append({'time': t, 'stores': s}, ignore_index=True)

    if df.empty:
        return None
    else:
        return df.sort_values(by=['time'])

