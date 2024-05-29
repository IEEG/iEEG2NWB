import numpy as np
import pandas as pd

# Get a TDT data store
def getTDTStore(tdt_data,store):
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

# Get TTLs from TDT epocs
def read_tdt_ttls_old(tdt_data,stores):
    # Go through each TDT specified store
    ii = 0
    #print(stores)
    for s in stores:
        ii += 1
        thisStore = getTDTStore(tdt_data, s)
        timestamps = thisStore.onset

        if ii == 1:
            df = pd.DataFrame(timestamps, columns=['time'])
            df['stores'] = s
        else:
            for t in timestamps:
                idx = df['time'].isin([t])
                if idx.any():
                    df.loc[idx, 'stores'] = df.loc[idx, 'stores'] + '/' + s
                else:
                    df = df.append({'time': t, 'stores': s}, ignore_index=True)

    return df.sort_values(by=['time'])

# Get TTLs from TDT epocs
def read_tdt_ttls(tdt_data,stores):
    # If stores is a string, convert it to list
    if isinstance(stores, str):
        stores = [stores]

    # Create empty dataframe
    df = pd.DataFrame({'time': [], 'stores': []})

    # Loop through all stores given
    for s in stores:
        thisStore = getTDTStore(tdt_data, s)

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

