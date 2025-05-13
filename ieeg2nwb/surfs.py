import numpy as np
from tqdm import tqdm
import os
from nibabel.freesurfer.io import read_geometry, read_annot, write_annot
from joblib import Parallel, delayed
import os.path as op
from nibabel.freesurfer.io import read_geometry, read_annot, write_annot
import pandas as pd
from typing import Union
import nibabel as nib
from nibabel.freesurfer.io import read_geometry
from mne import get_config, read_freesurfer_lut
from ieeg2nwb.fileio.helpers import _read_electrodeNames, _read_coordinates, _read_ielvis_base
from ieeg2nwb.utils import _get_data_directory, timenow, get_atlases, copy_fsaverage_data
from ieeg2nwb.fileio.freesurfer import read_xfm


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
    if (None in [hem, labels] or coords is None) and not all(is_none):
        raise ValueError("If one of [hem,label,coords] is None then all must be None")
        return None

    if subjects_dir is None:
        subjects_dir = get_config()['SUBJECTS_DIR']


    if all(is_none):
        coord_type = "PIAL" if surf.lower()=="pial" else "INF"
        elecs_df = _read_ielvis_base(subject, subjects_dir)
        coords = np.array(elecs_df[coord_type].to_list())
        hem = elecs_df["hem"].str.lower().to_list()
        labels = elecs_df["label"].to_list()
        n_elecs = elecs_df.shape[0]

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



def create_indiv_mapping(subject, subjects_dir=None, parc=None, n_jobs=-1):
    """Create individual subject parcellations from fsaverage to subject space of parcellations

    Parameters
    ----------
    subject : str
        Freesurfer subject ID
    subjects_dir : str | None
        The Freesurfer subject directory. If None then will take mne.get_config()['SUBJECTS_DIR']
    parc : str | None
        The parcellated atlas to create an individual mapping for. Must use parcellation name as appears as a file
        (ex: to create lh.HCP-MMP1.annot and rh.HCP-MMP1.annot parc="HCP-MMP1"), can work for customm parcellations
    n_jobs : int
        Number of parallels jobs, default is -1
    """

    atlases = get_atlases()
    all_parcs = [k["annot_fname"] for a, k in atlases.items()]
    
    if parc is None:
        parc = all_parcs
    elif isinstance(parc, str):
        if parc in all_parcs:
            all_parcs.pop(all_parcs.index(parc))
        parc = [parc] + all_parcs

    if subjects_dir is None:
        subjects_dir = get_config()['SUBJECTS_DIR']

    # Set paths
    subject_label_dir = os.path.join(subjects_dir, subject, 'label')
    subject_surf_dir = os.path.join(subjects_dir, subject, 'surf')
    fsavg_dir = os.path.join(subjects_dir, "fsaverage")
    fsavg_label_dir = os.path.join(fsavg_dir, "label")
    
    # Check which atlases still need to be made
    parcs_to_make = []
    need_to_copy_fsavg = False
    for annot_fname in parc:
        lh_fname = os.path.join(subject_label_dir, "lh." + annot_fname + ".annot")
        rh_fname = os.path.join(subject_label_dir, "rh." + annot_fname + ".annot")
        fsavg_lh_fname = os.path.join(fsavg_label_dir, "lh." + annot_fname + ".annot")
        fsavg_rh_fname = os.path.join(fsavg_label_dir, "rh." + annot_fname + ".annot")
        if not os.path.isfile(lh_fname) or not os.path.isfile(rh_fname):
            parcs_to_make.append(annot_fname)
        if not os.path.isfile(fsavg_lh_fname) or not os.path.isfile(fsavg_rh_fname):
            need_to_copy_fsavg = True

    # If some parcs are missing then get them from ieeg2nwb fsaverage copy
    if need_to_copy_fsavg:
        print("---->Copying files to fsaverage")
        copy_fsaverage_data(fsavg_dir)
    
    # If nothing to do, exit
    if len(parcs_to_make)==0:
        return
    
    # Loop over hemispheres
    for h in ["lh", "rh"]:

        # Define sphere.reg file
        surf_file = subject_surf_dir + os.sep + h + '.sphere.reg'

        # Load vertices of subject and find their closest vert in fsaverage
        verts, _ = read_geometry(surf_file)
        nearest_verts = find_nearest_vertex("fsaverage", subjects_dir=subjects_dir, coords=verts, hem=h[0], labels=np.arange(len(verts)).tolist(), surf="sphere", n_jobs=n_jobs)
        closest_vert = nearest_verts["closest_vert"].to_list()

        # Find label for each vertex and write out the annot file
        for parc_name in parcs_to_make:
            fsavg_annot_file = os.path.join(fsavg_label_dir, h + "." + parc_name + ".annot")
            subject_annot_fname = os.path.join(subject_label_dir, h + "." + parc_name + ".annot")
            fsavg_labels, ctab, names = read_annot(fsavg_annot_file)
            subject_labels = fsavg_labels[closest_vert]
            print("---->Writing %s" % subject_annot_fname)
            write_annot(subject_annot_fname, subject_labels, ctab, names)

