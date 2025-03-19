import os
import pandas as pd
import numpy as np
import os.path as op
from tqdm import tqdm
import nibabel as nib
import mne
from ieeg2nwb.surfs import sub_to_fsaverage, pial_to_inflated, find_nearest_vertex, elec_to_parc
from ieeg2nwb.fileio.helpers import _read_coordinates, _read_electrodeNames, _read_atlas_labels, _read_ptd
from ieeg2nwb.ptd import get_ptd_index

def read_ielvis(subject, subjects_dir=None, squeeze=False, write_missing=True, full=False, n_jobs=-1):
    """Function to read iELVis output in elec_recon directory

    Parameters
    ----------
    subdir : str
        The freesurfer subject directory containing iELVis files in a elec_recon folder
    squeeze: bool
        If True, the coordinates are returned as a list of lists. If False, all coordinates x,y,z points have their own column

    Returns : pd.DataFrame
        DataFrame of the iELVis produced information
    -------

    """
    if subjects_dir is None:
        subjects_dir = mne.get_config()['SUBJECTS_DIR']

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

        if not op.isfile(coordFname) and write_missing:
                if c == "INF":
                    pial_to_inflated(subject, subjects_dir=subjects_dir, write_to_file=True, n_jobs=n_jobs)
                elif c == "FSAVERAGE":
                    sub_to_fsaverage(subject, subjects_dir=subjects_dir, write_to_file=True, n_jobs=n_jobs)
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

    # Get PTD if it's there
    try:
        ptdFname = os.path.join(elecReconDir, 'GreyWhite_classifications.mat')

        if not op.isfile(ptdFname) and write_missing:
            _ = get_ptd_index(subject, subjects_dir=subjects_dir)

        ptd = _read_ptd(ptdFname)
        ptd['label'] = ptd['elec']
        ptd_df = pd.DataFrame(ptd)
        cols2keep = ['label','location','PTD']
        cols2remove = [col for col in list(ptd_df.columns) if col not in cols2keep]
        ptd_df = ptd_df.drop(columns=cols2remove)
        elecTable = pd.merge(elecTable,ptd_df,on='label')
    except:
        raise RuntimeError('Could not get PTD values')

    # Get atlas labels if they're there
    from ieeg2nwb.atlases import ATLASES
    for a, atlas_info in ATLASES.items():
        atlas_fname = os.path.join(elecReconDir, subject + '_' + a.upper() + "_AtlasLabels.tsv")
        if os.path.exists(atlas_fname):
            new_col_name = atlas_info["full_name"].lower() + "_atlas"
            atlas_labels = _read_atlas_labels(atlas_fname).rename(columns={'region': new_col_name})
            elecTable = pd.merge(elecTable, atlas_labels, on='label')
        #elif not write_missing:
        #    _ = elec_to_parc(subject, subjects_dir=subjects_dir, write_to_file=True, n_jobs=n_jobs)


    # If user species "full" then get all other coordinates that are snapped to surface
    if full:

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
        new_coords["depth_pial"] = np.array(nearest_verts.loc[:, "coords"].to_list())

        # Get the inflated coordinated for each electrodes
        new_coords["depth_inf"] = pial_to_inflated(subject, subjects_dir=subjects_dir, coords=new_coords["depth_pial"], hem=hem, labels=labels, write_to_file=False, n_jobs=n_jobs)

        # Coordinates for FSAverage inflated
        new_coords["fsaverage_inf"] = pial_to_inflated("fsaverage", subjects_dir=subjects_dir, coords=fsavg_coords, hem=hem, labels=labels, write_to_file=False, n_jobs=n_jobs)

        # Create fsavg_depth_pial from depth_pial in native space, treat all electrodes as subdural
        is_subdural = [True] * elecTable.shape[0]
        new_coords["fsaverage_depth_pial"] = sub_to_fsaverage(subject, subjects_dir=subjects_dir, coords=new_coords["depth_pial"].tolist(), hem=hem, labels=labels, subdural=is_subdural)

        # coordinates of electrodes snapped to fsaverage inflated surface
        new_coords["fsaverage_depth_inf"]  = pial_to_inflated("fsaverage", subjects_dir=subjects_dir, coords=new_coords["fsaverage_depth_pial"] , hem=hem, labels=labels, write_to_file=False, n_jobs=n_jobs)

        # Add the new coordinates to the table
        for colname, coords in new_coords.items():

            if not squeeze:
                coords = np.array(coords)
                for ii, xyz in enumerate(["x", "y", "z"]):
                    c_xyz = colname + "_" + xyz
                    elecTable[c_xyz] = coords[:, ii]
            else:
                elecTable[colname] = coords


    return elecTable
