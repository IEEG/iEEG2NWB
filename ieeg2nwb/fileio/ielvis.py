import os, re, copy, subprocess
from itertools import compress
import pandas as pd
import numpy as np
import os.path as op
from mne import get_config
from ieeg2nwb.channels import sub_to_fsaverage, pial_to_inflated, elec_to_parc, get_ptd_index
from ieeg2nwb.surfs import find_nearest_vertex, create_indiv_mapping
from ieeg2nwb.fileio.helpers import _read_coordinates, _read_electrodeNames, _read_atlas_labels, _read_ptd
from ieeg2nwb.utils import get_atlases

# subject = "NS162_02"
# subjects_dir=None
# squeeze=False
# write_missing=True
# parcs=True
# extra_coords=True
# full=True
# n_jobs = -1

def read_ielvis(subject, subjects_dir=None, squeeze=False, parcs=False, extra_coords=False, write_to_file=True, full=False, legacy=False, n_jobs=-1):
    """Function to read iELVis output in elec_recon directory

    Parameters
    ----------
    subjects_dir : str
        The freesurfer subject directory containing iELVis files in a elec_recon folder
    squeeze: bool
        If True, the coordinates are returned as a list of lists. If False, all coordinates x,y,z points have their own column
    parcs: bool
        Whether to include parcellations in the output. Includes DK, D, Y7, Y17 and HCP
    extra_coords: bool
        Whether to add extra coordinates such as location of electrode snapped to nearest vertex
    full: bool
        If True will set parcs and extra_coords to True
    write_to_file: bool
        Write any newly generated data to subject elec_recon folder
    legacy: bool
        Name the columns according to the older MATLAB ielvisImport convention
    n_jobs: int
        Number of parallel jobs to have, default -1
        
    Returns : pd.DataFrame
        DataFrame of the iELVis produced information
    -------

    """
    if subjects_dir is None:
        subjects_dir = get_config()['SUBJECTS_DIR']

    if full:
        parcs = True
        extra_coords = True

    elecReconDir = os.path.join(subjects_dir, subject, 'elec_recon')

    # Types of coordinates to import
    coord_types = ['LEPTO','LEPTOVOX','PIAL','PIALVOX','FSAVERAGE','INF']

    # Get electrodeNames files and turn into pandas DataFrame
    elecNamesFile = os.path.join(elecReconDir, subject + '.electrodeNames')
    elecNames = _read_electrodeNames(elecNamesFile)
    elecTable = pd.DataFrame(elecNames)

    # Get types of coordinates and add them to the DataFrame
    for c in coord_types:
        coordFname = os.path.join(elecReconDir, subject + '.' + c)

        if not op.isfile(coordFname):
                if c == "INF":
                    print("---->Getting inflated coordinates")
                    coords = pial_to_inflated(subject, subjects_dir=subjects_dir, write_to_file=True, n_jobs=n_jobs)
                elif c == "FSAVERAGE":
                    print("---->Getting fsaverage coordinates")
                    coords = sub_to_fsaverage(subject, subjects_dir=subjects_dir, write_to_file=True, n_jobs=n_jobs)
        elif not op.isfile(coordFname):
            continue

        coords = _read_coordinates(coordFname)

        if not squeeze:
            coords = np.array(coords)
            for ii, xyz in enumerate(["x", "y", "z"]):
                c_xyz = c + "_" + xyz
                elecTable[c_xyz] = coords[:, ii]
        else:
            elecTable[c] = coords

    # Get PTD values
    ptdFname = os.path.join(elecReconDir, 'GreyWhite_classifications.mat')
    if not op.isfile(ptdFname):
        print("---->Getting PTD")
        ptd= get_ptd_index(subject, subjects_dir=subjects_dir, write_to_file=write_to_file)
    else:
        ptd = _read_ptd(ptdFname)

    # Format the ptd elec labels
    for ii in range( len(ptd['elec']) ):
        ptd['elec'][ii] = ptd['elec'][ii].split('_')[0]
    
    # Convert PTD data to DataFrame to combine
    ptd['label'] = ptd['elec']
    ptd_df = pd.DataFrame(ptd)
    cols2keep = ['label','location','PTD']
    cols2remove = [col for col in list(ptd_df.columns) if col not in cols2keep]
    ptd_df = ptd_df.drop(columns=cols2remove)
    elecTable = pd.merge(elecTable,ptd_df,on='label')
    elecTable = elecTable.rename(columns={"location": "aparc_aseg"})

    
    # If user species "full" then get all other coordinates that are snapped to surface
    if extra_coords:

        # Find nearest vertex for each contact
        if squeeze:
            pial_coords = elecTable["PIAL"].to_list()
            fsavg_coords = elecTable["FSAVERAGE"].to_list()
        else:
            pial_coords = elecTable[["PIAL_x", "PIAL_y", "PIAL_z"]].to_numpy()
            fsavg_coords = elecTable[["FSAVERAGE_x", "FSAVERAGE_y", "FSAVERAGE_z"]].to_numpy()

        # Will be used a lot
        hem = elecTable["hem"].to_list()
        labels = elecTable["label"].to_list()

        new_coords = {}

        # Snap all electrodes to pial surface
        nearest_verts = find_nearest_vertex(subject, subjects_dir=subjects_dir, surf="pial", coords=pial_coords, hem=hem, labels=labels, n_jobs=n_jobs)
        new_coords["pial_snap"] = np.array(nearest_verts.loc[:, "coords"].to_list())

        # Get the inflated coordinated for each electrodes
        new_coords["inf_snap"] = pial_to_inflated(subject, subjects_dir=subjects_dir, coords=new_coords["pial_snap"], hem=hem, labels=labels, write_to_file=False, n_jobs=n_jobs)

        # Coordinates for FSAverage inflated
        new_coords["fsaverage_inf"] = pial_to_inflated("fsaverage", subjects_dir=subjects_dir, coords=fsavg_coords, hem=hem, labels=labels, write_to_file=False, n_jobs=n_jobs)

        # Create fsavg_depth_pial from depth_pial in native space, treat all electrodes as subdural
        is_subdural = [True] * elecTable.shape[0]
        new_coords["fsaverage_pial_snap"] = sub_to_fsaverage(subject, subjects_dir=subjects_dir, coords=new_coords["pial_snap"].tolist(), hem=hem, labels=labels, subdural=is_subdural)

        # coordinates of electrodes snapped to fsaverage inflated surface
        new_coords["fsaverage_inf_snap"]  = pial_to_inflated("fsaverage", subjects_dir=subjects_dir, coords=new_coords["fsaverage_pial_snap"] , hem=hem, labels=labels, write_to_file=False, n_jobs=n_jobs)

        # Add the new coordinates to the table
        for colname, coords in new_coords.items():

            if not squeeze:
                coords = np.array(coords)
                for ii, xyz in enumerate(["x", "y", "z"]):
                    c_xyz = colname + "_" + xyz
                    elecTable[c_xyz] = coords[:, ii]
            else:
                elecTable[colname] = coords

    # Get parcellations (atlases)
    if parcs:
        # Find distance to nearest pial vertex and add as dist_to_surf_mm
        if "PIAL_x" in elecTable.columns:
            coords_for_dist = elecTable.loc[:,["PIAL_x","PIAL_y", "PIAL_z"]].to_numpy().tolist()
        else:
            coords_for_dist = elecTable.loc[:,["PIAL"]].to_numpy().tolist()
            
        dist_to_vert = find_nearest_vertex(subject, subjects_dir=subjects_dir, surf="pial", coords=coords_for_dist, hem=elecTable["hem"].to_list(), labels=elecTable["label"].to_list(), n_jobs=-1)
        dist_df = dist_to_vert[["label", "distance"]].rename(columns={"distance": "dist_to_surf_mm"})
        elecTable = pd.merge(elecTable, dist_df, on="label")

        # Check if atlases need to be made
        atlas_fnames = []
        for a, atlas_info in get_atlases().items():
            atlas_fnames.append(os.path.join(subjects_dir, subject, "label", "lh."+atlas_info["annot_fname"] + ".annot"))
            atlas_fnames.append(os.path.join(subjects_dir, subject, "label", "rh."+atlas_info["annot_fname"] + ".annot"))
        if not all([os.path.isfile(f) for f in atlas_fnames]):
            print("---->Creating parcellations")
            create_indiv_mapping(subject, subjects_dir=subjects_dir, n_jobs=n_jobs)

        # Get atlas labels by snapping electrode to surface
        if squeeze:
            coords_cols = ["PIAL"]
        else:
            coords_cols = ["PIAL_x","PIAL_y", "PIAL_z"]
        for a, atlas_info in get_atlases().items():
            atlas_col_name = a + "_atlas"
            atlas_fname = os.path.join(elecReconDir, subject + '_' + a.upper() + "_AtlasLabels.tsv")
            if os.path.exists(atlas_fname):
                atlas_labels = _read_atlas_labels(atlas_fname).rename(columns={'region': atlas_col_name})
                elecTable = pd.merge(elecTable, atlas_labels, on='label')
            else:
                atlas_labels = elec_to_parc(
                    subject,
                    subjects_dir=subjects_dir,
                    coords=elecTable.loc[:,coords_cols].to_numpy().tolist(),
                    hem=elecTable["hem"].to_list(),
                    labels=elecTable["label"].to_list(),
                    spec=["G"]*elecTable.shape[0],
                    parc=a,
                    write_to_file=write_to_file,
                    n_jobs=n_jobs
                )
                elecTable = pd.merge(elecTable, pd.DataFrame(atlas_labels).rename(columns={"location": atlas_col_name}), on='label')


    # If legacy naming is desired then adjust table
    # """""       
    # {'SubID'}    {'Contact'}    {'ElecType'}    {'Hem'}    {'PIAL'}

    # Columns 6 through 10

    #     {'PIALVOX'}    {'LEPTO'}    {'LEPTOVOX'}    {'INF'}    {'DepthPial'}

    # Columns 11 through 14

    #     {'DepthInf'}    {'Dist'}    {'FSAverage'}    {'FSAverageInf'}

    # Columns 15 through 17

    #     {'FSAverageDPial'}    {'FSAverageDInf'}    {'PTD'}

    # Columns 18 through 21

    #     {'AparcAseg_idx'}    {'AparcAseg_Atlas'}    {'DK_idx'}    {'DK_Atlas'}

    # Columns 22 through 26

    #     {'DK_Lobe'}    {'D_idx'}    {'D_Atlas'}    {'D_Full'}    {'Y7_idx'}

    # Columns 27 through 31

    #     {'Y7_Atlas'}    {'Y17_idx'}    {'Y17_Atlas'}    {'spec'}    {'soz'}

    # Columns 32 through 35

    #     {'spikey'}    {'out'}    {'bad'}    {'fullname'}
    # """

    # elecTable["SubID"] = subject
    # legacy_columns = {
    #     "label": "Contact",
    #     "spec": "ElecType",
    #     "hem": "Hem",
    #     "pial": "PIAL",
    #     "pialvox": "PIALVOX",
    # }

    # Make all column names lower case
    elecTable = elecTable.rename(columns={k: k.lower() for k in elecTable.columns})



    return elecTable


#%% Update correspondence sheet
def update_correspondence_sheet(subject_id, freesurfer_dir, overwrite_file=False, file_copy=None, n_jobs=-1):
    
    #%% Define required columns
    columns_req = ['Good', 'Spec', 'SOZ'	, 'Spikey', 'Out', ['hem', 'LvsR'], 'ptd', 
                   'fsaverage_coords_1', 'fsaverage_coords_2', 'fsaverage_coords_3',
                   'fsaverage_inf_1', 'fsaverage_inf_2', 'fsaverage_inf_3',
                   'lepto_coords_1', 'lepto_coords_2', 'lepto_coords_3', 
                   'aparc_aseg', 'Desikan_Killiany', 'Destrieux', 'Yeo7', 'Yeo17', 'HCP', 
                   'dist']

    # ielvis datasheet is named consistently, but different from correspondence sheet
    columns_req_ielvis = ['N/A', 'spec', 'N/A', 'N/A', 'N/A', 'hem', 'ptd', 
                   'fsaverage_x', 'fsaverage_y', 'fsaverage_z',
                   'fsaverage_inf_x', 'fsaverage_inf_y', 'fsaverage_inf_z',
                   'lepto_x', 'lepto_y', 'lepto_z', 
                   'aparc_aseg', 'dk_atlas', 'd_atlas', 'y7_atlas', 'y17_atlas', 'hcp_atlas', 
                   'dist_to_surf_mm']

    # Colums that should not be changed, but renamed if necessary
    columns_rename = ['Label', 'TDT_chan']

    #%% Compute anatomical data that is not yet available

    # Get missing info 
    ielvis_df = read_ielvis(subject_id, subjects_dir=freesurfer_dir, full=True, n_jobs=n_jobs)

    # Also get coordinates on the inflated brain
    if np.sum(np.isnan(ielvis_df.inf_x)) == ielvis_df.shape[0]:
        
        pial_to_inflated(subject_id, subjects_dir=freesurfer_dir, write_to_file=True, n_jobs=n_jobs)
        
        # Reload the dataframe 
        ielvis_df = read_ielvis(subject_id, subjects_dir=freesurfer_dir, full=True, n_jobs=n_jobs)
        
    # Get updated PTD
    if np.sum(np.isnan(ielvis_df.ptd)) != 0:
        
        get_ptd_index(subject_id, subjects_dir=freesurfer_dir, write_to_file=True)
        
        # Reload the dataframe 
        ielvis_df = read_ielvis(subject_id, subjects_dir=freesurfer_dir, full=True, n_jobs=n_jobs)
      
    #%% Update the correspondence sheet

    #%% Load the correpondence sheet
    file_name = ('{:s}/{:s}/elec_recon/'
                 '{:s}_Electrodes_Natus_TDT_correspondence.xlsx').format(freesurfer_dir, subject_id, subject_id)

    assert os.path.exists(file_name), ('Make sure the correspondence sheet exists and is named '
                                       '{:s}_Electrodes_Natus_TDT_correspondence.xlsx').format(subject_id)

    # Load the correspondence sheet
    correspondence_sheet = pd.read_excel(file_name)

    #%% Reorder ielvis_df
    ielvis_df_sort = pd.DataFrame(columns=ielvis_df.columns)

    empty_row = pd.DataFrame([np.hstack((['LR_label', 'X', 'LR'], 
                                         np.empty(18) * np.nan,
                                         'Out',
                                         np.empty(17) * np.nan,
                                         ['Out'] * 5)).T], 
                             columns=ielvis_df.columns)

    for i in range(correspondence_sheet.shape[0]):
        
        idx = np.where(np.isin(ielvis_df.label, correspondence_sheet.Label[i]))[0]

        if len(idx) == 0:
            
            row = copy.copy(empty_row)
            
            label_i = correspondence_sheet.Label[i]
            
            row.label = label_i
            row.hem = label_i[0]
            
            # For reference channels use 'R' as spec
            if 'Ref' in label_i:
                row.spec = 'R'
                
            # Otherwise get the spec find all channels from the same electrode
            else:         
                idx_elec = [''.join(re.findall(r"\D", label_i)) in l for l in ielvis_df_sort.label]
                
                # Then use the spec of the first channel
                row.spec = ielvis_df_sort[idx_elec].spec.iloc[0]
                
            # Add row to the dataframe
            ielvis_df_sort.loc[i] = row.iloc[0]
           
        else:
            
            ielvis_df_sort.loc[i] = ielvis_df.iloc[idx[0]]

    if ielvis_df_sort.ptd.dtype == 'O':
        ielvis_df_sort.ptd = ielvis_df_sort.ptd.astype(float)
        
    #%% See what data is already there and add the rest
    corr_sheet_cols = correspondence_sheet.columns

    # Go through all required columsn

    for j,col_req in enumerate(columns_req):

        if type(col_req) is str:
            col_req = [col_req]
        
        idx_col = [[re.search(r, c, re.IGNORECASE) is not None for c in corr_sheet_cols] for r in col_req]
        col_in_sheet = [np.sum(ic) == 1 for ic in idx_col]
        
        r_ielvis = columns_req_ielvis[j]
        col_in_ielvis = np.sum([re.search(r_ielvis, c, re.IGNORECASE) is not None 
                                for c in columns_req_ielvis]) == 1
        
        assert np.sum(col_in_sheet) > 0 or col_in_ielvis, ('Column {:s} is not in the correspondence'
                                                           ' sheet for {:s} and cannot be created'
                                                           ' with ielvis, please add!').format(col_req[0], subject_id)
        
        if not np.sum(col_in_sheet) > 0 and col_in_ielvis:
            
            correspondence_sheet[col_req[0]] = ielvis_df_sort[r_ielvis]
            col_name = r_ielvis
            
        else:
            col_name = corr_sheet_cols[list(compress(idx_col, col_in_sheet))[0]][0]
         
        # Rename the column to be more consistent
        correspondence_sheet = correspondence_sheet.rename(columns={col_name: col_req[0]})
        
    #%% Replace columns if there is more data in the ielvis data
    if np.sum(np.isnan(correspondence_sheet.ptd)) > np.sum(np.isnan(ielvis_df_sort.ptd)):
        correspondence_sheet.ptd = ielvis_df_sort.ptd
        
    #%% Cosmetics on HCP labels
    correspondence_sheet.loc[[l == '???' for l in correspondence_sheet.HCP], 'HCP'] = 'Unknown'

    #%% Rename some columns
    for c in columns_rename:
        
        idx_col = [np.sum([re.search(pattern, col, re.IGNORECASE) is not None 
                           for pattern in c.split('_')]) == len(c.split('_')) 
                   for col in correspondence_sheet.columns]
        
        # Handle case when two columns include the pattern
        if np.sum(idx_col) > 1:
            
            col_names = correspondence_sheet.columns[idx_col]
            col_name = col_names[np.argmin([len(ic) for ic in correspondence_sheet.columns[idx_col]])]
            idx_col = correspondence_sheet.columns == col_name
            
        assert np.sum(idx_col) == 1, '{:s} not found in correspondence sheet, double check!'.format(c)
        
        correspondence_sheet = correspondence_sheet.rename(columns={correspondence_sheet.columns[idx_col][0]: c})

    #%% Remove unnamed columns 
    unnamed_cols = correspondence_sheet.columns[['Unnamed' in c 
                                                 for c in correspondence_sheet.columns]]
    
    correspondence_sheet = correspondence_sheet.drop(unnamed_cols, axis=1)
    
    #%% Save the updated correspondence sheet
    if overwrite_file:
        file_name_write = file_name   
    else:
        file_name_write = file_name.replace('.xlsx', '_updated.xlsx')

    correspondence_sheet.to_excel(file_name_write)
        
    #%% Copy the updated correspondence sheet 
    if file_copy is not None:
        
        cmd_cp = ('cp {:s} {:s}').format(file_name_write, file_copy)   
                                                                        
        returned_value = subprocess.call(cmd_cp, shell=True)
        print('Copy file {:s}: {:d}'.format(os.path.split(file_name_write)[-1], returned_value))