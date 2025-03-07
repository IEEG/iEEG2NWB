import numpy as np
from tqdm import tqdm
import os
from nibabel.freesurfer.io import read_geometry, read_annot, write_annot
from joblib import Parallel, delayed
import os.path as op
from nibabel.freesurfer.io import read_geometry
import pandas as pd
from typing import Union
import nibabel as nib
from fileio.helpers import _read_electrodeNames, _read_coordinates
from ieeg2nwb.utils import _get_data_directory

"""
Find nearest

* subject vertex to fsaverage vertex (all/subset)
* multiple coordinates distance to nearest vertex (subject space)
"""



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
        from mne import get_config
        subjects_dir = get_config()['SUBJECTS_DIR']

    # If coordinates not specified then plot subject using iELVis data
    if coords is None:
        elecs_df = read_ielvis(op.join(subjects_dir, subject))
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
        from .utils import timenow
        fname = op.join(subjects_dir, subject, "elec_recon", subject + ".INF")
        with open(fname, 'w') as file:
            file.write(timenow() + '\n')
            file.write("R A S \n")
            np.savetxt(file, inf_coords, fmt='%.6f', delimiter=' ')

    return inf_coords


def find_nearest_vertex(subject, subjects_dir=None, surf="pial", coords=None, hem=None, labels=None, n_jobs=-1):
    """
    Find the nearest vertex on the cortical surface to a given set of coordinates.

    Parameters
    ----------
    subject : str
        The name of the subject.
    subjects_dir : str, optional
        The directory where the subject's data is stored. If not provided, it will be obtained from the MNE configuration.
    surf : str
        Type of surface to find coordinates on ["pial", "inflated", "sphere"], default is "pial"
    coords : array-like, shape (n, 3), optional
        The coordinates of the points for which to find the nearest vertex. If not provided, it will be assumed that the coordinates are already defined.
    hem : str or list, optional
        The hemisphere(s) to consider. If 'l' or 'lh', only the left hemisphere will be considered. If 'r' or 'rh', only the right hemisphere will be considered. If not provided, both hemispheres will be considered.
    labels : array-like, shape (n,), optional
        The labels associated with each coordinate. If not provided, the labels will be assigned as consecutive integers starting from 0.
    n_jobs: int, optional
        Number of parallel jobs to run, default is -1

    Returns
    -------
    df : pandas DataFrame
        A DataFrame containing the following columns:
        - 'label': The label associated with each coordinate.
        - 'distance': The distance between each coordinate and its nearest vertex.
        - 'closest_vert': The index of the closest vertex for each coordinate.
        - 'hem': The hemisphere associated with each label.

    """

    # Check for incorrect input
    is_none = [x is None for x in [coords, hem, labels]]
    if None in [coords, hem, labels] and not all(is_none):
        raise ValueError("If one of [hem,label,coords] is None then all must be None")
        return None

    if subjects_dir is None:
        from mne import get_config
        subjects_dir = get_config()['SUBJECTS_DIR']


    if all(is_none):
        coord_type = "PIAL" if surf.lower()=="pial" else "INF"
        elec_recon_dir = os.path.join(subjects_dir, subject, 'elec_recon')
        coord_fname = os.path.join(elec_recon_dir, subject + '.' + coord_type)
        elecnames_fname = os.path.join(elec_recon_dir, subject + '.electrodeNames')
        coords = _read_coordinates(coord_fname)
        coords = np.array(coords)
        elecNames = _read_electrodeNames(elecnames_fname)

        hem = []
        labels = []
        n_elecs = len(elecNames)
        for el in elecNames:
            hem.append(el["hem"].lower())
            labels.append(el["label"])

    else:

        if isinstance(coords, list) or isinstance(coords, tuple):
            coords = np.array(coords)

        n_elecs = coords.shape[0]

        # Interpret hemispheres
        if isinstance(hem, str):
            hem = hem.lower()
            if hem == 'l' or hem == 'lh':
                hem = ['l' for ii in range(n_elecs)]
            elif hem == 'r' or hem == 'rh':
                hem = ['r' for ii in range(n_elecs)]
        elif isinstance(hem, list):
            hem = [h.lower() for h in hem]

    # Load surf data
    surf_dir = os.path.join(subjects_dir, subject, 'surf')
    if isinstance(surf, dict):
        lh_surf_file = surf["l"]
        rh_surf_file = surf["r"]
    else:
        if surf == "sphere":
            surf = "sphere.reg"
        lh_surf_file = surf_dir + os.sep + 'lh.' + surf
        rh_surf_file = surf_dir + os.sep + 'rh.' + surf
    verts = {}
    verts['l'], _ = read_geometry(lh_surf_file)
    verts['r'], _ = read_geometry(rh_surf_file)


    # Function to process each coordinate
    def process_coordinate(ii):
        h = hem[ii]
        dist = np.sqrt(np.sum((verts[h] - coords[ii, :]) ** 2, axis=1))
        closest_vert = dist.argmin()
        return {
            'label': labels[ii], 
            'distance': dist.min(), 
            'closest_vert': closest_vert, 
            'hem': h, 
            "coords": verts[h][closest_vert]
            }

    # Parallel processing of coordinates
    results = Parallel(n_jobs=n_jobs)(delayed(process_coordinate)(ii) for ii in
                                      tqdm(range(n_elecs), desc="Finding nearest vertices on %s" % surf,
                                           unit=' vertices', position=0, leave=True))

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    return df


def elec_to_parc(
    subject: str,
    subjects_dir: str = None,
    coords: np.array = None,
    hem: Union[list, np.array] = None,
    labels: Union[list, np.array] = None,
    parc: list[str, str] = None,
    write_to_file: bool = True,
    n_jobs: int = -1
):
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
    parc : list[str, str], optional
        A 2-element list with the shorthand for an atlas and the filename piece for it (ex: ["y7", "Yeo2011_7Networks_N1000"])
    write_to_file : bool, optional
        Whether to write the parcellation results to a file (default is True).
    n_jobs : int, optional
        The number of parallel jobs to use for computation. Use `-1` to use all available processors (default is -1).

    Returns
    -------
    parcellation_results : pd.DataFrame
        A dictionary containing the parcellation results for each electrode. Keys include electrode names and corresponding parcellation labels.

    Notes
    -----
    This function maps electrode coordinates to the corresponding parcellations in a FreeSurfer subject's brain, allowing for analysis of electrode data within specific brain regions.

    Examples
    --------
    >>> subject = 'subject01'
    >>> coords = np.array([[30.2, -22.5, 50.7], [28.1, -24.3, 48.9]])
    >>> hem = ['lh', 'lh']
    >>> labels = ['caudal-MT', 'precentral']
    >>> results = elec_to_parc(subject, coords=coords, hem=hem, labels=labels)
    """
    if subjects_dir is None:
        from mne import get_config
        subjects_dir = get_config()['SUBJECTS_DIR']

    # Find nearest vertex of each electrode
    elec_df = find_nearest_vertex(subject, subjects_dir=subjects_dir, coords=coords, hem=hem, labels=labels, n_jobs=n_jobs)

    # Combine with ielvis data
    ielvis_df = read_ielvis(subject, subjects_dir, squeeze=True).drop("hem",axis=1)
    elec_df = pd.merge(elec_df, ielvis_df, on="label")

    # Make a dataframe for the final output
    output_df = pd.DataFrame({"label": elec_df["label"]})

    # Load volumetric segmentation for depths
    aparc_aseg_file = os.path.join(subjects_dir, subject, 'mri', 'aparc+aseg.mgz')
    aparc_aseg = nib.load(aparc_aseg_file)
    aparc_aseg_data = aparc_aseg.get_fdata()

    # Read freesurfer lut
    from mne import read_freesurfer_lut
    roi2val, _ = read_freesurfer_lut()
    val2roi = {v: k for k, v in roi2val.items()}

    # Go through each parcellation
    from ieeg2nwb.atlases import ATLASES
    for a in ATLASES.keys():

        # If the atlas doesn't exist then create it
        sample_annot = os.path.join(subjects_dir, 'fsaverage', 'label', 'lh.' + ATLASES[a]["annot_fname"] + '.annot')
        if not os.path.exists(sample_annot):
            create_indiv_mapping(subject, a, subjects_dir=subjects_dir, n_jobs=n_jobs)

        # Create a dataframe to save results
        atlas_labels = elec_df.copy()
        atlas_labels["region"] = ""

        # Load the atlas
        lh_annot_fname = os.path.join(subjects_dir, subject, 'label', 'lh.' + ATLASES[a]["annot_fname"] + '.annot')
        rh_annot_fname = os.path.join(subjects_dir, subject, 'label', 'rh.' + ATLASES[a]["annot_fname"] + '.annot')
        lh_annot_labels, ctab, lh_annot_names = read_annot(lh_annot_fname)
        rh_annot_labels, _, rh_annot_names = read_annot(rh_annot_fname)

        # Go through each electrode
        for i, row in atlas_labels.iterrows():

            # If depth then find voxel it's in
            if row["spec"] == "D":
                coords = np.round(row["PIALVOX"]).astype(int)
                xyz = np.array([coords[0], coords[1], aparc_aseg_data.shape[2] - coords[2]])
                aparc_aseg_vox_val = aparc_aseg_data[tuple(xyz)]
                aparc_aseg_roi = val2roi[aparc_aseg_vox_val]
                atlas_labels.at[i, "region"] = aparc_aseg_roi
                continue

            # Find the region value
            if row["hem"] == "l":
                val = lh_annot_labels[row["closest_vert"]]
                atlas_labels.at[i, "region"] = lh_annot_names[val].decode()
            else:
                val = rh_annot_labels[row["closest_vert"]]
                atlas_labels.at[i, "region"] = rh_annot_names[val].decode()

            if val == -1:
                atlas_labels.at[i, "region"] = "unknown"

        # Save to tsv file
        if write_to_file:
            output_fname = os.path.join(subjects_dir, subject, 'elec_recon', subject + '_' + a.upper() + '_AtlasLabels.tsv')
            atlas_labels.to_csv(output_fname, sep='\t', index=False, columns=["label", "region"], header=False)

        # Add to output dataframe
        output_df[a] = atlas_labels["region"]

    return output_df

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
    from fileio.ielvis import _read_electrodeNames, _read_coordinates
    from nibabel.freesurfer.io import read_geometry

    if subjects_dir is None:
        from mne import get_config

        subjects_dir = get_config()['SUBJECTS_DIR']

    # If coordinates not specified then plot subject using iELVis data
    if labels is None:
        elecReconDir = op.join(subjects_dir, subject, 'elec_recon')
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
            coords=ecog_elecs["pial"].to_list(),
            hem=ecog_elecs["hem"],
            labels=ecog_elecs["labels"],
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
            h = row["hem"]
            sphere_coords[i, :] = verts[h][closest_verts[i]]

        # Now find the nearest vertex on fsaverage
        nearest_verts_avg = find_nearest_vertex(
            "fsaverage",
            subjects_dir=subjects_dir,
            coords=sphere_coords,
            hem=ecog_elecs["hem"],
            labels=ecog_elecs["labels"],
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
        import nibabel as nib
        mri = nib.load(op.join(subjects_dir, subject, 'mri', 'orig.mgz'))
        Norig = mri.header.get_vox2ras()
        Torig = mri.header.get_vox2ras_tkr()

        # Read talairach.xfm
        from .fileio.ielvis import freesurfer_read_xfm
        tal_xfm = freesurfer_read_xfm(op.join(subjects_dir, subject, "mri", "transforms", 'talairach.xfm'))

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
            from .utils import timenow
            fname = op.join(subjects_dir, subject, "elec_recon", subject + ".FSAVERAGE")
            with open(fname, 'w') as file:
                file.write(timenow() + '\n')
                file.write("R A S \n")
                np.savetxt(file, avg_coords, fmt='%.6f', delimiter=' ')

        return avg_coords





