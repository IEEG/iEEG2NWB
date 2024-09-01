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
from .io import read_ielvis
from .utils import _get_data_directory

"""
Find nearest

* subject vertex to fsaverage vertex (all/subset)
* multiple coordinates distance to nearest vertex (subject space)
"""


# Shorthands for the different atlases as part of freesurfer"
ATLASES = {
    'dk': 'aparc',
    'd': 'aparc.a2009s',
    'hcp': 'HCP-MMP1',
    'y7': 'Yeo2011_7Networks_N1000',
    'y17': 'Yeo2011_17Networks_N1000'
}

ATLASES = {
    "dk": {
        "annot_fname": "aparc",
        "description": "Desikan-Killiany atlas",
        "full_name": "Desikan-Killiany"
    },
    "d": {
        "annot_fname": "aparc.a2009s",
        "description": "Destrieux atlas",
        "full_name": "Destrieux"
    },
    "hcp": {
        "annot_fname": "HCP-MMP1",
        "description": "Human Connectome Project Multi-Modal Parcellation",
        "full_name": "HCP-MMP1"
    },
    "y7": {
        "annot_fname": "Yeo2011_7Networks_N1000",
        "description": "Yeo 2011 7 Networks",
        "full_name": "Yeo7",
    },
    "y17": {
        "annot_fname": "Yeo2011_17Networks_N1000",
        "description": "Yeo 2011 17 Networks",
        "full_name": "Yeo17"
    }
}

def create_indiv_mapping(subject, subjects_dir=None, parc=None, n_jobs=-1):
    """Create individual mapping from fsaverage to subject space of parcellations

    Parameters
    ----------
    subject : str
        Freesurfer subject ID
    subjects_dir : str | None
        The Freesurfer subject directory. If None then will take mne.get_config()['SUBJECTS_DIR']

    """

    if parc is not None:
        ATLASES = {parc[0]: parc[1]}

    atlas_list = ATLASES.keys()

    if subjects_dir is None:
        from mne import get_config
        subjects_dir = get_config()['SUBJECTS_DIR']

    if atlas_list is None:
        atlas_list = ATLASES
    else:
        if isinstance(atlas_list, dict):
            atlas_list = [atlas_list]

        if not isinstance(atlas_list, list):
            raise TypeError("atlas_list must be a list of dictionaries or a single dictionary")

    for hem in ["lh", "rh"]:
        global closest_verts
        closest_verts = []
        for a in atlas_list.keys():

            global cverts
            cverts = closest_verts

            annot_fname = hem + '.' + ATLASES[a]["annot_fname"] + '.annot'

            if op.isfile(annot_fname):
                continue

            fsaverage_annot_file = os.path.join(subjects_dir, 'fsaverage', 'label', annot_fname)
            if not op.exists(fsaverage_annot_file):
                from .utils import copy_fsaverage_data
                copy_fsaverage_data(subjects_dir)

            # Get fsaverage data
            fsaverage_sphere_file = os.path.join(subjects_dir,'fsaverage','surf', hem + '.sphere.reg')
            fsaverage_annot_file = os.path.join(subjects_dir, 'fsaverage','label',annot_fname)
            fsavg_vert_coords, _ = read_geometry(fsaverage_sphere_file)
            fsavg_annot_labels, ctab, annot_names = read_annot(fsaverage_annot_file)

            # Get single subject data
            subject_sphere_file = os.path.join(subjects_dir, subject, 'surf', hem + '.sphere.reg')
            subject_annot_file = os.path.join(subjects_dir, subject, 'label', annot_fname)
            sub_vert_coords, _ = read_geometry(subject_sphere_file)

            # Create variables for single subject annot
            n_sub_verts = sub_vert_coords.shape[0]
            subject_vert_labels = np.zeros(n_sub_verts)

            def process_vertex(ii):
                if len(cverts) == 0:
                    dist = np.sum((fsavg_vert_coords - sub_vert_coords[ii, :]) ** 2, axis=1)
                    fsavg_closest_vert = dist.argmin()
                    cverts.append(fsavg_closest_vert)
                    label = fsavg_annot_labels[fsavg_closest_vert]
                else:
                    label = fsavg_annot_labels[cverts[ii]]

                return label

            results = Parallel(n_jobs=n_jobs)(
                delayed(process_vertex)(ii) for ii in
                    tqdm(range(n_sub_verts),
                    desc='Processing %s' % annot_fname,
                    unit=' vertices',
                    position=0,
                    leave=True
                         )
            )

            subject_vert_labels[:] = results

            # Write to file
            write_annot(subject_annot_file, subject_vert_labels.astype('int'), ctab, annot_names)
            print('----> Writing to file: %s' % subject_annot_file)


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
    np.array
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
        from .misc import timenow
        fname = op.join(subjects_dir, subject, "elec_recon", subject + ".INF")
        with open(fname, 'w') as file:
            file.write(timenow() + '\n')
            file.write("R A S \n")
            np.savetxt(file, inf_coords, fmt='%.6f', delimiter=' ')

    return inf_coords


def find_nearest_vertex(subject, subjects_dir=None, surf="pial", coords=None, hem=None, labels=None, n_jobs=-1):
    """
    Find the nearest vertex on the cortical surface to a given set of coordinates.

    Parameters:
    ----------
    subject : str
        The name of the subject.
    subjects_dir : str, optional
        The directory where the subject's data is stored. If not provided, it will be obtained from the MNE configuration.
    coords : array-like, shape (n, 3), optional
        The coordinates of the points for which to find the nearest vertex. If not provided, it will be assumed that the coordinates are already defined.
    hem : str or list, optional
        The hemisphere(s) to consider. If 'l' or 'lh', only the left hemisphere will be considered. If 'r' or 'rh', only the right hemisphere will be considered. If not provided, both hemispheres will be considered.
    labels : array-like, shape (n,), optional
        The labels associated with each coordinate. If not provided, the labels will be assigned as consecutive integers starting from 0.
    n_jobs: int, optional
        Number of parallel jobs to run, default is -1

    Returns:
    -------
    df : pandas DataFrame
        A DataFrame containing the following columns:
        - 'label': The label associated with each coordinate.
        - 'distance': The distance between each coordinate and its nearest vertex.
        - 'closest_vert': The index of the closest vertex for each coordinate.
        - 'hem': The hemisphere associated with each label.

    """
    if subjects_dir is None:
        from mne import get_config
        subjects_dir = get_config()['SUBJECTS_DIR']

    if coords is None:
        coord_type = "PIAL" if surf.lower()=="pial" else "INF"
        elecs_df = read_ielvis(subject, subjects_dir, squeeze=True)
        coords = np.stack(elecs_df[coord_type].values)
        hem = elecs_df["hem"].str.lower().to_list()
        labels = elecs_df["label"].to_list()

    n_elecs = coords.shape[0]

    if labels is None:
        labels = np.arange(0, n_elecs, 1)

    # Interpret hemispheres
    if hem == 'l' or hem == 'lh':
        hem = ['l' for ii in range(n_elecs)]
    elif hem == 'r' or hem == 'rh':
        hem = ['r' for ii in range(n_elecs)]

    # Load data
    surf_dir = os.path.join(subjects_dir, subject, 'surf')
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
        return {'label': labels[ii], 'distance': dist.min(), 'closest_vert': closest_vert, 'hem': h}

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
