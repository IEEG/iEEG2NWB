import os
import pandas as pd
import numpy as np
import os.path as op
from scipy.io import savemat, loadmat
from ieeg2nwb.utils import read_aseg_csv


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



# Read ielvis files that are automatically produced by the software, no extras
def _read_ielvis_base(subject, subjects_dir):
    """Read in a few basic iELVis file that require no additional processing to be done"""

        elec_recon_dir = os.path.join(subjects_dir, subject, 'elec_recon')
        elecnames_fname = os.path.join(elec_recon_dir, subject + '.electrodeNames')
        elecs_df = pd.DataFrame(_read_electrodeNames(elecnames_fname))
        
        coord_types = ['LEPTO','LEPTOVOX','PIAL','PIALVOX']
        for coord in coord_types:
            coord_fname = os.path.join(elec_recon_dir, subject + '.' + coord)
            coords = _read_coordinates(coord_fname)
            elecs_df[coord] = coords

        return elecs_df