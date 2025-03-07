import os
import pandas as pd
import numpy as np
import os.path as op
from tqdm import tqdm
import nibabel as nib
from ieeg2nwb.surfs import sub_to_fsaverage, pial_to_inflated
from ieeg2nwb.fileio.helpers import _read_coordinates, _read_electrodeNames, _read_atlas_labels, _read_ptd
from ieeg2nwb.ptd import get_ptd_index

def read_ielvis(subject, subjects_dir=None, squeeze=False, write_missing=True, full=False):
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
        from mne import get_config
        subjects_dir = get_config()['SUBJECTS_DIR']

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
                    pial_to_inflated(subject, subjects_dir=subjects_dir, write_to_file=True)
                elif c == "FSAVERAGE":
                    sub_to_fsaverage(subject, subjects_dir=subjects_dir, write_to_file=True)
        elif not op.isfile(coordFname):
            continue

        coordFname = os.path.join(elecReconDir, subject + '.inf')

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
        print('Could not get PTD values')

    # Get atlas labels if they're there
    from ieeg2nwb.atlases import ATLASES
    for a, atlas_info in ATLASES.items():
        atlas_fname = os.path.join(elecReconDir, subject + '_' + a.upper() + "_AtlasLabels.tsv")
        if os.path.exists(atlas_fname):
            new_col_name = atlas_info["full_name"].lower() + "_atlas"
            atlas_labels = _read_atlas_labels(atlas_fname).rename(columns={'region': new_col_name})
            elecTable = pd.merge(elecTable, atlas_labels, on='label')


    # If user species "full" then get all other coordinates that are snapped to surface
    # if full:

    #     from ieeg2nwb.surfs import sub_to_fsaverage, find_nearest_vertex, pial_to_inflated

    #     # Find nearest vertex for each contact
    #     nearest_verts = find_nearest_vertex()



    return elecTable

def freesurfer_read_xfm(xfm_file):
    """Read a Freesurfer transformation matrix file

    Parameters
    ----------
    xfm_file : str
        The path to the transformation matrix file

    Returns
    -------
    xfm : np.ndarray
        The transformation matrix
    """

    # Initialize variables
    matrix_lines = []
    matrix_start = False

    # Open and read the file
    with open(xfm_file, 'r') as file:
        for line in file:
            if "Linear_Transform" in line:
                matrix_start = True
                continue  # Skip the "Linear Transform" line

            if matrix_start:
                # Split the line into components (assuming space-separated values)
                matrix_lines.append(list(map(float, line.replace(";", "").split())))
                # Stop after reading 4 lines
                if len(matrix_lines) == 4:
                    break

    # Convert the list of lists into a NumPy array
    matrix = np.array(matrix_lines)
    return matrix