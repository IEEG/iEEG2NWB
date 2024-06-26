import os
import pandas as pd

def _read_electrodeNames(elecNamesFile):
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

def read_ielvis(subject, subjects_dir=None):
    """Simple function to read iELVis elec_recon directory

    Parameters
    ----------
    subdir : str
        The freesurfer subject directory containing iELVis files in a elec_recon folder

    Returns : pd.DataFrame
        DataFrame of the iELVis produced information
    -------

    """
    if subjects_dir is None:
        from mne import get_config
        subjects_dir = get_config()['SUBJECTS_DIR']

    elecReconDir = os.path.join(subjects_dir, subject, 'elec_recon')

    # Types of coordinates to import
    coords2use = ['LEPTO','LEPTOVOX','PIAL','PIALVOX','FSAVERAGE','INF']

    # Get electrodeNames files and turn into pandas DataFrame
    elecNamesFile = os.path.join(elecReconDir, subject + '.electrodeNames')
    elecNames = _read_electrodeNames(elecNamesFile)
    elecTable = pd.DataFrame(elecNames)

    # Get types of coordinates and add them to the DataFrame
    for c in coords2use:
        coordFname = os.path.join(elecReconDir, subject + '.' + c)
        coords = _read_coordinates(coordFname)
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

    return elecTable

