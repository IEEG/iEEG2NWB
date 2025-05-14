#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:27:55 2025

@author: max
"""
import re
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

def read_edf_ttls(raw_data, dig_stores, std_thresh):
    
    if type(std_thresh) is str:
        std_thresh = float(std_thresh)
    
    dig_data = raw_data.get_data(dig_stores)
    event_times_store = [None] * dig_data.shape[0]
    
    for dc in range(dig_data.shape[0]):
        
        # Use Gaussian mixture models to find means 
        gmm = GaussianMixture(n_components=3)
        gmm.fit(dig_data[dc,:].reshape(-1, 1))
        
        means = gmm.means_
        standard_deviations = gmm.covariances_**0.5  
        
        # Look at the fit of the first Gaussian, which is the noise
        # Use this as a threshold
        noise_thresh = means[0] + std_thresh*standard_deviations[0]
        
        idx_trig = dig_data[dc,:]  > noise_thresh
        idx_onset = np.where(np.diff(idx_trig[0,:].astype(float)) > 0)[0] + 1
        
        s_time = raw_data.times[idx_onset]
        s_name = ['{:s}'.format(dig_stores[dc]) for i in range(len(idx_onset))]

        event_times_store[dc] = pd.DataFrame({'time': s_time, 'stores': s_name})
        
        # # For debugging
        # event_vec = np.zeros(raw_data.times.shape)
        # event_vec[idx_onset] = 1
        
        # import matplotlib.pyplot as plt
        # # plt.figure()
        # plt.plot(raw_data.times, dig_data.T)
        # plt.plot(raw_data.times, event_vec)
        # plt.close()
        
    # Loop through all stores given and combine
    event_times_df = pd.DataFrame({'time': [], 'stores': []})
    
    for this_store in event_times_store:
        
        timestamps = this_store.time
        
        for i,t in enumerate(timestamps):
            idx = event_times_df['time'].isin([t])
            if idx.any():
                event_times_df.loc[idx, 'stores'] = event_times_df.loc[idx, 'stores'] + '/' + this_store.stores[i]
            else:
                event_times_df = pd.concat((event_times_df, pd.DataFrame({'time': [t], 'stores': [this_store.stores[i]]})))

    event_times_df = event_times_df.sort_values(by='time')
    event_times_df = event_times_df.reset_index()
    event_times_df = event_times_df.drop('index', axis=1)
    
    idx = np.hstack(([False], np.diff(event_times_df.time) < (3 / raw_data.info['sfreq'])))
    idx_pre = np.hstack((idx[1:], [False]))
    
    idx_num = np.where(idx)[0]
    idx_pre_num = np.where(idx_pre)[0]
    
    store_corr = np.empty(len(idx_pre_num), dtype=object)
    
    for j in range(len(idx_pre_num)):
        
        i_idx = int(re.findall(r'-?\d+\.?\d*', event_times_df.stores.iloc[idx_num[j]])[0])
        i_pre = int(re.findall(r'-?\d+\.?\d*', event_times_df.stores.iloc[idx_pre_num[j]])[0])
        
        if i_idx < i_pre:
            store_corr[j] = '{:s}/{:s}'.format(event_times_df.stores.iloc[idx_num[j]], 
                                               event_times_df.stores.iloc[idx_pre_num[j]])
        elif i_idx > i_pre:
            store_corr[j] = '{:s}/{:s}'.format(event_times_df.stores.iloc[idx_pre_num[j]], 
                                               event_times_df.stores.iloc[idx_num[j]])
            
    event_times_df.loc[idx_pre, 'stores'] = store_corr
    event_times_df = event_times_df.drop(idx_num)
    event_times_df = event_times_df.reset_index()
    event_times_df = event_times_df.drop('index', axis=1)
    
    return event_times_df