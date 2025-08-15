import nibabel as nib
import numpy as np
import os
from mne import get_config

def annot_to_volume(subject, annot_name, subjects_dir=None):
    """
    Convert FreeSurfer annotation files for both hemispheres to a volumetric file (.mgz or .nii.gz),
    ensuring all gray matter voxels on the cortical ribbon are labeled appropriately.

    Parameters
    ----------
    subject : str
        Subject name.
    annot_name : str
        Name of the annotation file (e.g., 'aparc') without extension.
    subjects_dir : str, optional
        Path to FreeSurfer subjects directory. If None, uses the MNE config SUBJECTS_DIR.

    Returns
    -------
    None
        The function saves the output file to disk and prints the output path.
    """

    if subjects_dir is None:
        subjects_dir = get_config()['SUBJECTS_DIR']
    
    hemis = ['lh', 'rh']
    aseg_path = os.path.join(subjects_dir, subject, 'mri', 'aseg.mgz')
    ribbon_path = os.path.join(subjects_dir, subject, 'mri', 'ribbon.mgz')
    aseg_img = nib.load(aseg_path)
    aseg_data = aseg_img.get_fdata()
    ribbon_img = nib.load(ribbon_path)
    ribbon_data = ribbon_img.get_fdata()
    volume = np.zeros(aseg_data.shape, dtype=np.int32)
    
    for hemi in hemis:
        annot_path = os.path.join(subjects_dir, subject, 'label', f'{hemi}.{annot_name}.annot')
        surf_path = os.path.join(subjects_dir, subject, 'surf', f'{hemi}.white')
        
        if not os.path.exists(annot_path) or not os.path.exists(surf_path):
            print(f"Skipping {hemi} due to missing files.")
            continue
        
        labels, ctab, names = nib.freesurfer.read_annot(annot_path)
        verts, faces = nib.freesurfer.read_geometry(surf_path)
        vox2ras_tkr = aseg_img.header.get_vox2ras_tkr()
        vox_coords = np.linalg.inv(vox2ras_tkr) @ np.hstack((verts, np.ones((verts.shape[0], 1)))).T
        vox_coords = np.round(vox_coords[:3, :].T).astype(int)
        
        for i, (x, y, z) in enumerate(vox_coords):
            if 0 <= x < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= z < volume.shape[2]:
                volume[x, y, z] = labels[i]
        
        # Label all gray matter voxels within the cortical ribbon
        hemi_ribbon_val = 3 if hemi == 'lh' else 42
        gm_voxels = np.where(ribbon_data == hemi_ribbon_val)
        for x, y, z in zip(gm_voxels[0], gm_voxels[1], gm_voxels[2]):
            if volume[x, y, z] == 0:  # Only label previously unassigned voxels
                volume[x, y, z] = labels[np.argmin(np.linalg.norm(verts - np.array([x, y, z]), axis=1))]
    
    if output_file.endswith('.mgz'):
        img = nib.MGHImage(volume, aseg_img.affine)
    elif output_file.endswith('.nii.gz'):
        img = nib.Nifti1Image(volume, aseg_img.affine)
    else:
        raise ValueError("Unsupported output file format. Use .mgz or .nii.gz")
    
    nib.save(img, output_file)
    output_file = f"{subjects_dir}/{subject}/mri/{annot_name}.nii.gz"
    output_file = os.path.join(subjects_dir, subject, "mri", annot_name + ".nii.hz")
    print(f"Saved {output_file}")
