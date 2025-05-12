import numpy as np
from tqdm import tqdm
import os
from nibabel.freesurfer.io import read_geometry, read_annot
from joblib import Parallel, delayed
import os.path as op
import pandas as pd
from typing import Union
import nibabel as nib
from scipy.io import savemat
from nibabel.freesurfer.io import read_geometry
import mne
from mne import get_config, read_freesurfer_lut
from ieeg2nwb.fileio.helpers import _read_electrodeNames, _read_coordinates, _read_ielvis_base
from ieeg2nwb.utils import _get_data_directory, timenow, get_atlases, read_aseg_csv
from ieeg2nwb.fileio.freesurfer import read_xfm
from ieeg2nwb.surfs import find_nearest_vertex
    
def elec_to_parc(
    subject: str,
    subjects_dir: str = None,
    coords: Union[np.array, list] = None,
    hem: Union[list, np.array] = None,
    labels: Union[list, np.array] = None,
    spec: list[str] = None,
    parc: list[str, str] = None,
    write_to_file: bool = True,
    n_jobs: int = -1
) -> dict:
    """
    Map electrode coordinates to parcellations in a FreeSurfer subject's brain.

    Parameters
    ----------
    subject : str
        The FreeSurfer subject identifier.
    subjects_dir : str, optional
        The directory containing the FreeSurfer subjects. If None, the environment variable `SUBJECTS_DIR` will be used.
    coords : np.array, optional
        A NumPy array of electrode coordinates in RAS space (shape: n_electrodes x 3). If None, coordinates should be provided elsewhere.
    hem : Union[list, np.array], optional
        A list or array indicating the hemisphere ('lh' or 'rh') for each electrode. The length should match the number of electrodes in `coords`. If None, the hemispheres should be inferred or provided elsewhere.
    labels : Union[list, np.array], optional
        A list or array of electrode labels.
    spec: list[str], optional
        List of type of sensor for each contact. Options are ["D", "G", "S", "SEEG", "ECOG"] where D=seeg and G/S=ecog
    parc : list[str, str], optional
        A 2-element list with the shorthand for an atlas and the filename piece for it (ex: ["y7", "Yeo2011_7Networks_N1000"])
        default parcellates Desikan-Killiany, Destrieux, Yeo 7 , Yeo 17, HCP
    write_to_file : bool, optional
        Whether to write the parcellation results to a file (default is True).
    n_jobs : int, optional
        The number of parallel jobs to use for computation. Use `-1` to use all available processors (default is -1).

    Returns
    -------
    parcellation_results : dict
        A dictionary containing the parcellation results for each electrode. Keys include electrode names and corresponding parcellation labels.

    Notes
    -----
    This function maps electrode coordinates to the corresponding parcellations in a FreeSurfer subject's brain, allowing for analysis of electrode data within specific brain regions.

    Examples
    --------
    >>> subject = 'subject01'
    >>> coords = np.array([[30.2, -22.5, 50.7], [28.1, -24.3, 48.9]])
    >>> hem = ['l', 'l']
    >>> labels = ['Elec1', 'Elec2']
    >>> spec = ["D", "ecog"]
    >>> parc = "y7"
    >>> results = elec_to_parc(subject, coords=coords, hem=hem, labels=labels, spec=spec, parc=parc)
    """

    if subjects_dir is None:
        subjects_dir = get_config()['SUBJECTS_DIR']

    # Check for incorrect input
    #is_none = [x is None for x in [coords, hem, labels, spec]]
    is_none = []
    for x in [coords, hem, labels, spec]:
        if isinstance(x, np.ndarray) or isinstance(x, list) or isinstance(x, str):
            is_none.append(False)
        elif x is not None:
            is_none.append(False)
        else:
            is_none.append(True)


    if None in is_none and not all(is_none):
        raise ValueError("If one of [hem,label,coords] is None then all must be None")
        return None
    elif not all(is_none):
        elec_df = pd.DataFrame({"label": labels, "spec": spec,"hem": hem, "PIAL": coords, "PIALVOX": coords})
    else:
        elec_df = _read_ielvis_base(subject, subjects_dir)   
        
    # Rename spec to either ecog or seeg
    elec_df["hem"] = elec_df["hem"].str.lower().to_list()
    elec_df["spec"] = elec_df["spec"].str.lower()
    elec_df = elec_df.replace({"spec": ["d"]}, "seeg").replace({"spec": ["g", "s"]}, "ecog")
    
    # Set column for output
    elec_df["location"] = ""

    # Get the filename for the parcellation to make
    parc_fname = None
    parc_shorthand = None
    atlases = get_atlases()
    if isinstance(parc, str):
        parc = parc.lower()
        if parc in atlases.keys():
            parc_fname = atlases[parc]["annot_fname"]
            parc_shorthand = parc.upper()
        else:
            parc_fname = parc
            parc_shorthand = parc.upper()
    elif isinstance(parc, list):
        parc_fname = parc[1]
        parc_shorthand = parc[0]
    else:
        raise ValueError("parc must be a string or a list")
        return None
    
    # Read freesurfer lut
    roi2val, _ = read_freesurfer_lut()
    val2roi = {v: k for k, v in roi2val.items()}

    # Separate into subcortical and depths
    seeg_elecs = elec_df[elec_df["spec"]=="seeg"]
    ecog_elecs = elec_df[elec_df["spec"]=="ecog"]


    # Work on depth electrodes
    if not seeg_elecs.empty:

        # Load volumetric segmentation for depths
        aparc_aseg_file = os.path.join(subjects_dir, subject, 'mri', 'aparc+aseg.mgz')
        aparc_aseg = nib.load(aparc_aseg_file)
        aparc_aseg_data = aparc_aseg.get_fdata()

        # Go through each electrode
        for i, row in seeg_elecs.iterrows():

            # If depth then find voxel it's in
            coords = np.round(row["PIALVOX"]).astype(int)
            xyz = np.array([coords[0], coords[1], aparc_aseg_data.shape[2] - coords[2]])
            aparc_aseg_vox_val = aparc_aseg_data[tuple(xyz)]
            aparc_aseg_roi = val2roi[aparc_aseg_vox_val]
            elec_df.at[i, "location"] = aparc_aseg_roi


    # Work on ecog electrodes
    if not ecog_elecs.empty:

        # Get closest vertices for each electrode
        ecog_coords = ecog_elecs["PIAL"].to_list()
        ecog_hem = ecog_elecs["hem"].to_list()
        ecog_labels = ecog_elecs["label"].to_list()

        # If vertex indices were given then don't need to find nearest vertex again
        if isinstance(ecog_coords[0], list):
            vert_df = find_nearest_vertex(subject, subjects_dir=subjects_dir, surf="pial", coords=ecog_coords, hem=ecog_hem, labels=ecog_labels, n_jobs=n_jobs)
        else:
            vert_df = pd.DataFrame({"label": ecog_labels, "distance": 0, "closest_vert": ecog_coords, "hem": ecog_hem})

        # Read annotation files
        lh_annot_fname = os.path.join(subjects_dir, subject, 'label', 'lh.' + parc_fname + '.annot')
        rh_annot_fname = os.path.join(subjects_dir, subject, 'label', 'rh.' + parc_fname + '.annot')

        if not os.path.isfile(lh_annot_fname) or not os.path.isfile(rh_annot_fname):
            raise RuntimeError("Annotation files do not exist. Create them by running `ieeg2nwb.surfs.create_indiv_mapping()")
            return None
        
        lh_annot_labels, ctab, lh_annot_names = read_annot(lh_annot_fname)
        rh_annot_labels, _, rh_annot_names = read_annot(rh_annot_fname)

        # Go through each electrode
        for i, row in vert_df.iterrows():

            # Find the region value
            label = row["label"]
            if row["hem"] == "l":
                val = lh_annot_labels[row["closest_vert"]]
                elec_df.loc[elec_df["label"]==label, "location"] = lh_annot_names[val].decode()
            else:
                val = rh_annot_labels[row["closest_vert"]]
                elec_df.loc[elec_df["label"]==label, "location"] = rh_annot_names[val].decode()
            if val == -1:
                 elec_df.loc[elec_df["label"]==label, "location"] = "unknown"

    # Create output dataframe
    output_df = elec_df.loc[:,["label","location"]]

    # Write to tsv output
    if write_to_file:
        tsv_fname = os.path.join(subjects_dir, subject, "elec_recon", f"{subject}_{parc_shorthand}_AtlasLabels.tsv")
        output_df.to_csv(tsv_fname, sep="\t", index=False, header=False)

    # Prepare for output
    return output_df.to_dict("list")


def sub_to_fsaverage(subject, subjects_dir=None, coords=None, hem=None, labels=None, subdural=None, n_jobs=-1, write_to_file=True):
    """
    Convert coordinates from subject space to fsaverage space.

    Parameters
    ----------
    subject : str
        The subject ID.
    subjects_dir : str, optional
        The Freesurfer subject directory. If not provided, it will be read from the MNE config file.
    coords : np.array, optional
        Input coordinates. If not provided, it will be read from the .PIAL file in the subject's directory.
    hem : Union[list, np.array], optional
        Hemisphere of each electrode. Must be specified if coords are passed in.
    labels : Union[list, np.array], optional
        Names of electrodes. Must be specified if coords are passed in.
    n_jobs: int, optional
        Number of parallel jobs to run, default is -1

    Returns
    -------
    np.array
        fsaverage coordinates.

    Notes
    -----
    - If `subjects_dir` is not provided, it will be read from the MNE config file.
    - If `coords` is not provided, it will be read from the .PIAL file in the subject's directory.
    - If `labels` is not provided, it will be read from the electrode data in the subject's directory.
    - The function converts the coordinates from the subject space to the fsaverage space using the nearest vertex mapping.
    - The fsaverage coordinates are returned as a numpy array.

    """

    if subjects_dir is None:
        subjects_dir = get_config()['SUBJECTS_DIR']
    
    elecReconDir = op.join(subjects_dir, subject, 'elec_recon')

    # If coordinates not specified then plot subject using iELVis data
    if labels is None:
        elecNamesFile = op.join(elecReconDir, subject + '.electrodeNames')
        elecNames = _read_electrodeNames(elecNamesFile)
        labels = []
        subdural = []
        hem = []
        for elec in elecNames:
            labels.append(elec["label"])
            subdural.append(elec["spec"] == "D")
            hem.append(elec["hem"])
        coords = _read_coordinates(op.join(elecReconDir, subject + '.PIAL'))
    else:
        if coords is None:
            raise ValueError("coords must not be None if labels is specified")
        elif hem is None:
            raise ValueError("hem must not be None if labels is specified")
        elif subdural is None:
            raise ValueError("subdural must not be None if labels is specified")
        elif coords is None:
            raise ValueError("coords must not be None if labels is specified")


    # Store average coordinates here
    avg_coords = np.zeros((len(labels), 3))

    if isinstance(coords, np.ndarray):
        coords = coords.tolist()

    elecs_df = pd.DataFrame({"labels": labels, "hem": hem, "subdural": subdural, "native": coords})

    # For each electrode, find nearest vertex on native brain then on average brain
    if any(not i for i in subdural):

        # Take subset but keep original index for later
        ecog_elecs = elecs_df.loc[~elecs_df["subdural"], :]
        orig_index = ecog_elecs.index
        ecog_elecs = ecog_elecs.reset_index(drop=True)
        ecog_elecs.loc[:, "orig_index"] = orig_index

        # Find nearest vertex on native brain
        nearest_verts_df = find_nearest_vertex(
            subject,
            subjects_dir=subjects_dir,
            coords=ecog_elecs["native"].to_list(),
            hem=ecog_elecs["hem"].to_list(),
            labels=ecog_elecs["labels"].to_list(),
            surf="pial",
            n_jobs=n_jobs
        )

        # Get the closest vertices
        closest_verts = nearest_verts_df["closest_vert"].to_list()

        # Get sub spheres data
        surf_dir = op.join(subjects_dir, subject, 'surf')
        lh_sub_sphere_file = surf_dir + os.sep + 'lh.sphere.reg'
        rh_sub_sphere_file = surf_dir + os.sep + 'rh.sphere.reg'
        verts = {}
        verts['l'], _ = read_geometry(lh_sub_sphere_file)
        verts['r'], _ = read_geometry(rh_sub_sphere_file)

        # Get the coordinates on spheres
        sphere_coords = np.zeros((len(ecog_elecs), 3))
        for i, row in ecog_elecs.iterrows(): #range(len(ecog_elecs)):
            h = row["hem"].lower()
            sphere_coords[i, :] = verts[h][closest_verts[i]]

        # Now find the nearest vertex on fsaverage
        nearest_verts_avg = find_nearest_vertex(
            "fsaverage",
            subjects_dir=subjects_dir,
            coords=sphere_coords,
            hem=ecog_elecs["hem"].to_list(),
            labels=ecog_elecs["labels"].to_list(),
            surf="sphere",
            n_jobs=n_jobs
        )
        del verts

        # Get fsaverage data
        lh_avg_pial_file = op.join(subjects_dir, 'fsaverage', 'surf', 'lh.pial')
        rh_avg_pial_file = op.join(subjects_dir, 'fsaverage', 'surf', 'rh.pial')
        avg_verts = {}
        avg_verts['l'], _ = read_geometry(lh_avg_pial_file)
        avg_verts['r'], _ = read_geometry(rh_avg_pial_file)

        #Get average pial coordinates on average brain
        for i, row in ecog_elecs.iterrows():
            idx = row["orig_index"]
            h = row["hem"].lower()
            avg_coords[idx, :] = avg_verts[h][nearest_verts_avg["closest_vert"][i]]

        del avg_verts

    # Check for subdural electrodes
    if any(i for i in subdural):

        postimp_file = op.join(elecReconDir, subject + '.POSTIMPLANT')
        if not op.isfile(postimp_file):
            postimp_file = op.join(elecReconDir, subject + '.CT')
        postimp_coords = _read_coordinates(postimp_file)

        elecs_df.loc[:, "postimp"] = postimp_coords

        # Take subset but keep original index for later
        subdural_elecs = elecs_df.loc[elecs_df["subdural"], :]
        orig_index = subdural_elecs.index
        subdural_elecs = subdural_elecs.reset_index(drop=True)
        subdural_elecs.loc[:, "orig_index"] = orig_index
        postimp_coords = np.array(subdural_elecs["postimp"].to_list())

        # Read orig.mgz and get transformation info: vox2ras and tkrvox2ras
        mri = nib.load(op.join(subjects_dir, subject, 'mri', 'orig.mgz'))
        Norig = mri.header.get_vox2ras()
        Torig = mri.header.get_vox2ras_tkr()

        # Read talairach.xfm
        tal_xfm = read_xfm(op.join(subjects_dir, subject, "mri", "transforms", 'talairach.xfm'))

        # For readability break the calculation into a few lines
        n_elec = subdural_elecs.shape[0]
        p2 = np.linalg.lstsq(Torig, np.vstack((postimp_coords.T, np.ones((1, n_elec)))), rcond=1)[0]
        mni305_coords = (tal_xfm @ Norig @ p2).T

        #Get average pial coordinates on average brain
        for i, row in subdural_elecs.iterrows():
            idx = row["orig_index"]
            h = row["hem"]
            avg_coords[idx, :] = mni305_coords[i, :]

    # Write out to file
    if write_to_file:
        fname = op.join(subjects_dir, subject, "elec_recon", subject + ".FSAVERAGE")
        with open(fname, 'w') as file:
            file.write(timenow() + '\n')
            file.write("R A S\n")
            np.savetxt(file, avg_coords, fmt='%.6f', delimiter=' ')

    return avg_coords

def pial_to_inflated(subject: str, subjects_dir: str = None, coords: np.array = None,
                     labels: Union[list, np.array] = None, hem: Union[list, np.array] = None,
                     write_to_file: bool = True, n_jobs: int = -1) -> np.array:
    """
    Convert coordinates from pial surface to inflated surface.

    Parameters
    ----------
    subject : str
        Subject ID.
    subjects_dir : str, optional
        Freesurfer subject directory. If not provided, it will be read from the MNE config file.
    coords : np.array, optional
        Input coordinates. If not provided, it will be read from the .PIAL file.
    labels : Union[list,np.array], optional
        Names of electrodes. Must be specified if coords are passed in
    hem : Union[list,np.array], optional
        Hemisphere of each electrode, Must be specified if coords are passed in
    write_to_file : bool, optional
        Whether to write out the information as subject.INF in elec_recon.
    n_jobs: int, optional
        Number of parallel jobs to run, default is -1

    Returns
    -------
    coords : numpy array
        Inflated coordinates.

    Notes
    -----
    - If `subjects_dir` is not provided, it will be read from the MNE config file.
    - If `coords` is not provided, it will be read from the .PIAL file in the subject's directory.
    - If `labels` is not provided, it will be read from the electrode data in the subject's directory.
    - The function converts the coordinates from the pial surface to the inflated surface using the nearest vertex mapping.
    - The inflated coordinates are returned as a numpy array.
    - If `write_to_file` is True, the inflated coordinates will be written out to a file named subject.INF in the elec_recon directory of the subject's directory.

    """

    if subjects_dir is None:
        subjects_dir = get_config()['SUBJECTS_DIR']

    # If coordinates not specified then plot subject using iELVis data
    if coords is None:
        elecs_df = _read_ielvis_base(subject, subjects_dir)
        coords = np.array(elecs_df['PIAL'].to_list())
        labels = elecs_df['label'].to_list()
        hem = [h.lower() for h in elecs_df['hem'].to_list()]
    else:
        if labels is None:
            raise ValueError("labels must not be None, must be the same length as coords")
        elif len(labels) != coords.shape[0]:
            raise ValueError("len(labels) must equal coords.shape[0]")
        elif hem is None:
            raise ValueError("hem must not be none, must be the same length as coords")
        elif len(hem) != coords.shape[0]:
            raise ValueError("len(hem) must equal coords.shape[0]")

    # Convert to inflated coordinates
    df = find_nearest_vertex(subject, subjects_dir=subjects_dir, coords=coords, hem=hem, labels=labels, n_jobs=n_jobs)

    # Load data
    surf_dir = op.join(subjects_dir, subject, 'surf')
    lh_pial_file = surf_dir + os.sep + 'lh.inflated'
    rh_pial_file = surf_dir + os.sep + 'rh.inflated'
    verts = {}
    verts['l'], _ = read_geometry(lh_pial_file)
    verts['r'], _ = read_geometry(rh_pial_file)

    inf_coords = np.zeros((len(labels), 3))

    # Iterate over each electrode to get coordinates
    for i, row in df.iterrows():
        inf_coords[i, :] = verts[row["hem"]][row["closest_vert"]]

    # Write out to file
    if write_to_file:
        fname = op.join(subjects_dir, subject, "elec_recon", subject + ".INF")
        with open(fname, 'w') as file:
            file.write(timenow() + '\n')
            file.write("R A S \n")
            np.savetxt(file, inf_coords, fmt='%.6f', delimiter=' ')

    return inf_coords


def get_ptd_index(subject: str, offset: float = 2, subjects_dir: str = None, write_to_file: bool =True):
    """Find Proximal Tissue Density of each sensor for a subject

    Parameters
    ----------
    subject : str
        Freesurfer subject ID
    offset : float, optional
        Area of tissue for which to calculate PTD by default 2
    subjects_dir : str, optional
        The directory where the subject's data is stored. If not provided, it will be obtained from the MNE configuration
    write_to_file : bool, optional
        Save results to mat file, by default True

    Returns
    -------
    dict
        Contains the following keys: ["elec", "location", "nb_Gpix", "nb_Wpix", "offset", "PTD"]
        Matches the MATLAB style output
    """

    if subjects_dir is None:
        subjects_dir = mne.get_config()['SUBJECTS_DIR']

    # Get LEPTOVOX coordinates
    elecReconDir = op.join(subjects_dir, subject, 'elec_recon')
    elecNamesFile = op.join(elecReconDir, subject + '.electrodeNames')
    elecNames_tmp = _read_electrodeNames(elecNamesFile)
    elecNames = [f"{el['label']}_{el['spec']}_{el['hem']}" for el in elecNames_tmp]    
    coordFname = op.join(elecReconDir, subject + '.LEPTOVOX')
    coordinates = _read_coordinates(coordFname)
    
    #elecs_df = read_ielvis(subject=subject, subjects_dir=subjects_dir, squeeze=True)

    # Read the aparc+aseg.mgz file
    aparc_aseg_file = op.join(subjects_dir, subject, 'mri', 'aparc+aseg.mgz')
    aparc_aseg = nib.load(aparc_aseg_file)
    aparc_aseg_data = aparc_aseg.get_fdata()

    # Read the aseg.mgz file
    aseg_file = op.join(subjects_dir, subject, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_file)
    aseg_data = aseg.get_fdata()

    # read csv with aseg values
    aseg_df = read_aseg_csv()

    # Read freesurfer lut
    roi2val, _ = mne.read_freesurfer_lut()
    val2roi = {v: k for k, v in roi2val.items()}

    # dict for rois for each electrode
    ptd_idx = {
        "elec": [],
        "location": [],
        "nb_Gpix": [],
        "nb_Wpix": [],
        "PTD": [],
        "offset": offset
    }

    # Iterate over electrodes
    for label, coords in tqdm(zip(elecNames, coordinates), desc="Finding PTD", unit="Electrode", position=0, leave=True, total=len(elecNames)):

        coords = np.round(coords).astype(int)

        # Get Voxel coordinates
        xyz = np.array([coords[0], coords[1], aparc_aseg_data.shape[2] - coords[2]])

        # Get the label of the voxel
        aparc_aseg_vox_val = aparc_aseg_data[tuple(xyz)]
        aparc_aseg_roi = val2roi[aparc_aseg_vox_val]

        # Define the range of x, y, z coordinates for the cube around xyz
        x_range = np.arange(max(0, xyz[0] - offset), min(aparc_aseg_data.shape[0], xyz[0] + offset + 1))
        y_range = np.arange(max(0, xyz[1] - offset), min(aparc_aseg_data.shape[1], xyz[1] + offset + 1))
        z_range = np.arange(max(0, xyz[2] - offset), min(aparc_aseg_data.shape[2], xyz[2] + offset + 1))

        # Initialize the distances array with a large value
        distances = np.full(aparc_aseg_data.shape, np.inf)

        # Calculate distances within the cube
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    distances[x, y, z] = np.linalg.norm(np.array([x, y, z]) - xyz)

        # Find close voxels
        close_voxels = np.where(distances <= offset)
        close_vox_vals = aseg_data[close_voxels]

        # For each value in close_vox_vals find whether it's in aseg_df and get the tissue
        close_vox_tissues = [aseg_df[aseg_df['value'] == v]["tissue"].values for v in close_vox_vals]
        close_vox_tissues = np.array(close_vox_tissues).flatten()
        n_gm = np.sum(close_vox_tissues == "gm")
        n_wm = np.sum(close_vox_tissues == "wm")

        # Finally get value of ptd
        ptd_val = (n_gm - n_wm) / (n_gm + n_wm + 1e-6)

        # Collect all info
        ptd_idx["elec"].append(label)
        ptd_idx["location"].append(aparc_aseg_roi)
        ptd_idx["nb_Gpix"].append(n_gm)
        ptd_idx["nb_Wpix"].append(n_wm)
        ptd_idx["PTD"].append(ptd_val)

    # Save
    if write_to_file:
        fname = op.join(subjects_dir, subject, "elec_recon", "GreyWhite_classifications.mat")
        tmp = ptd_idx.copy()
        tmp["elec"] = np.array(tmp["elec"], dtype=object)
        tmp["location"] = np.array(tmp["location"], dtype=object)
        print("--->Saving %s" % fname)
        savemat(fname, {"PTD_idx": tmp})
        
    return ptd_idx

