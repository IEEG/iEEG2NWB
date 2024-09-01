import os
import pandas as pd
import numpy as np

def _read_electrodeNames(elecNamesFile):
    """Read electrodeNames file and return a list of dictionaries"""
    elecList = []
    # Dictionary for each electrode
    elecDict = {
        'label': None,
        'spec': None,
        'hem': None,
    }
    hdrLine = 'Name, Depth/Strip/Grid, Hem\n'
    with open(elecNamesFile,'r') as f:
        isHdr = True
        for thisLine in f:
            if isHdr:
                isHdrline = thisLine == hdrLine
                if isHdrline:
                    isHdr = False

                continue

            else:
                thisElec = elecDict.copy()
                eInfo = thisLine.replace('\n', '').split(' ')
                thisElec['label'] = eInfo[0]
                thisElec['spec'] = eInfo[1]
                thisElec['hem'] = eInfo[2]
                elecList.append(thisElec)

    return elecList

def _read_coordinates(coordFname):
    """Read electrode coordinates from an ielvis file and return a list of lists"""
    coords = []
    with open(coordFname,'r') as f:
        isHdr = True
        for thisLine in f:
            if isHdr:
                isHdrline = (thisLine == 'R A S\n') or (thisLine == 'X Y Z\n')
                if isHdrline:
                    isHdr = False
                continue

            else:
                thisCoord = [float(ii) for ii in thisLine.replace('\n', '').split(' ')]
                coords.append(thisCoord)

    return coords

def _read_ptd(ptdFname):
    """Read PTD values from a .mat file"""
    from scipy.io import loadmat
    ptd_tmp = loadmat(
        ptdFname,
        variable_names=['PTD_idx'],
        #squeeze_me=True,
        simplify_cells=True
    )

    ptd = ptd_tmp['PTD_idx']
    for ii in range( len(ptd['elec']) ):
        ptd['elec'][ii] = ptd['elec'][ii].split('_')[0]

    return ptd

def _read_atlas_labels(atlasFname):
    """Read labels in a saved tsv file"""
    atlas_labels = pd.read_csv(atlasFname, sep='\t', names=['label','region'])
    return atlas_labels

def read_ielvis(subject, subjects_dir=None, squeeze=False):
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
    from .surfs import ATLASES
    for a in ATLASES.keys():
        atlas_fname = os.path.join(elecReconDir, subject + '_' + a.upper() + "_AtlasLabels.tsv")
        if os.path.exists(atlas_fname):
            new_col_name = ATLASES[a]["full_name"].lower() + "_atlas"
            atlas_labels = _read_atlas_labels(atlas_fname).rename(columns={'region': new_col_name})
            elecTable = pd.merge(elecTable, atlas_labels, on='label')


    return elecTable

