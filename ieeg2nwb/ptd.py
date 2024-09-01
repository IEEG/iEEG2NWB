import os.path as op
import numpy as np
from tqdm import tqdm
import mne
import nibabel as nib
from scipy.io import savemat
from .io import read_ielvis
from .utils import read_aseg_csv


def get_ptd_index(subject: str, offset: float = 2, subjects_dir: str = None):
    if subjects_dir is None:
        from mne import get_config
        subjects_dir = get_config()['SUBJECTS_DIR']

    elecs_df = read_ielvis(subject=subject, subjects_dir=subjects_dir, squeeze=True)

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
    PTD_idx = {
        "elec": [],
        "location": [],
        "nb_Gpix": [],
        "nb_Wpix": [],
        "PTD": [],
        "offset": offset
    }

    # Iterate over electrodes
    pbar = tqdm(total=elecs_df.shape[0], unit=" Electrode", desc="Finding PTD")
    for i, row in elecs_df.iterrows():
        coords = np.round(row['LEPTOVOX']).astype(int)

        xyz = np.array([coords[0], coords[1], aparc_aseg_data.shape[2] - coords[2]])

        # Get the label of the voxel
        aparc_aseg_vox_val = aparc_aseg_data[tuple(xyz)]
        # aseg_vox_val = aseg_data[tuple(xyz)]
        # aseg_roi = val2roi[aseg_vox_val]
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
        PTD_idx["elec"].append(row['label'])
        PTD_idx["location"].append(aparc_aseg_roi)
        PTD_idx["nb_Gpix"].append(n_gm)
        PTD_idx["nb_Wpix"].append(n_wm)
        PTD_idx["PTD"].append(ptd_val)

        pbar.update(1)

    # Save
    fname = op.join(subjects_dir, subject, "elec_recon", "GreyWhite_classifications.mat")
    savemat(fname, {"PTD_idx": PTD_idx})

    return PTD_idx







