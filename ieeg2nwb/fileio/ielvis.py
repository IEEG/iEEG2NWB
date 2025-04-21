import os
import pandas as pd
import numpy as np
import os.path as op
from mne import get_config
from ieeg2nwb.channel import sub_to_fsaverage, pial_to_inflated, elec_to_parc
from ieeg2nwb.surfs import , find_nearest_vertex, create_indiv_mapping
from ieeg2nwb.fileio.helpers import _read_coordinates, _read_electrodeNames, _read_atlas_labels, _read_ptd
from ieeg2nwb.ptd import get_ptd_index
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
                    coords = pial_to_inflated(subject, subjects_dir=subjects_dir, write_to_file=True, n_jobs=n_jobs)
                elif c == "FSAVERAGE":
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
        ptd= get_ptd_index(subject, subjects_dir=subjects_dir, write_to_file=write_to_file)
    else:
        ptd = _read_ptd(ptdFname)
    
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
        dist_to_vert = find_nearest_vertex(subject, subjects_dir=subjects_dir, surf="pial", coords=elecTable.loc[:,["PIAL_x","PIAL_y", "PIAL_z"]].to_numpy().tolist(), hem=elecTable["hem"].to_list(), labels=elecTable["label"].to_list(), n_jobs=-1)
        dist_df = dist_to_vert[["label", "distance"]].rename(columns={"distance": "dist_to_surf_mm"})
        elecTable = pd.merge(elecTable, dist_df, on="label")

        # Check if atlases need to be made
        atlas_fnames = []
        for a, atlas_info in get_atlases().items():
            atlas_fnames.append(os.path.join(subjects_dir, subject, "label", "lh."+atlas_info["annot_fname"] + ".annot"))
            atlas_fnames.append(os.path.join(subjects_dir, subject, "label", "rh."+atlas_info["annot_fname"] + ".annot"))
        if not all([os.path.isfile(f) for f in atlas_fnames]):
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
