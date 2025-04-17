import numpy as np
import os
from ieeg2nwb.surfs import pial_to_inflated, find_nearest_vertex, elec_to_parc, sub_to_fsaverage, create_indiv_mapping

SUBID = "NS162_02"

def test_coordinate_functions():


    # Make up some coordinates
    n = 10
    sample_coords = np.random.randint(-100, high=100, size=(n,3))
    hems = ["L"] * n
    labels = ["elec" + str(ii+1) for ii in range(n)]
    subdural = [False] * n
    subdural[-2:] = [True, True]

    # Get absolute path of ieeg2nwb directory
    ieeg2nwb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Source and destination paths for fsaverage
    sub_dir = os.path.join(ieeg2nwb_dir, "test_files", "freesurfer")
    fsaverage_src = os.path.join(ieeg2nwb_dir, "data", "fsaverage")
    fsaverage_dst = os.path.join(sub_dir, "fsaverage")
    
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(fsaverage_dst), exist_ok=True)

    # Copy fsaverage directory
    if os.path.exists(fsaverage_src):
        if not os.path.exists(fsaverage_dst):
            import shutil
            shutil.copytree(fsaverage_src, fsaverage_dst)

    try:
        output = find_nearest_vertex(SUBID, subjects_dir=sub_dir, surf="pial", coords=sample_coords, hem=hems, labels=labels)
    except Exception as e:
        raise AssertionError(f"find_nearest_vertex() failed with error: {str(e)}")
    

    try:
        output = pial_to_inflated(SUBID, subjects_dir=sub_dir, coords=sample_coords, hem=hems, labels=labels)
    except Exception as e:
        raise AssertionError(f"pial_to_inflated() failed with error: {str(e)}")
    

    try:
        output = sub_to_fsaverage(SUBID, subjects_dir=sub_dir, )
    except Exception as e:
        raise AssertionError(f"sub_to_fsaverage() failed with error: {str(e)}")
    

    # Clean up
    # Delete fsaverage directory if it exists
    if os.path.exists(fsaverage_dst):
        import shutil
        shutil.rmtree(fsaverage_dst)


def test_create_indiv_mapping():
    
    # Get absolute path of ieeg2nwb directory
    ieeg2nwb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Source and destination paths for fsaverage
    sub_dir = os.path.join(ieeg2nwb_dir, "test_files", "freesurfer")
    fsaverage_src = os.path.join(ieeg2nwb_dir, "data", "fsaverage")
    fsaverage_dst = os.path.join(sub_dir, "fsaverage")
    
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(fsaverage_dst), exist_ok=True)

    # Copy fsaverage directory
    if os.path.exists(fsaverage_src):
        if not os.path.exists(fsaverage_dst):
            import shutil
            shutil.copytree(fsaverage_src, fsaverage_dst)

    try:
        create_indiv_mapping(SUBID, subjects_dir=sub_dir, parc="HCP-MMP1")
    except Exception as e:
        raise AssertionError(f"create_indiv_mapping() failed with error: {str(e)}")
    
    lh_file = os.path.join(sub_dir, SUBID, "label", "lh.HCP-MMP1.annot")
    rh_file = os.path.join(sub_dir, SUBID, "label", "rh.HCP-MMP1.annot")
    assert os.path.isfile(lh_file), "lh annot file does not exist"
    assert os.path.isfile(lh_file), "rh annot file does not exist"

    # Clean up
    # Delete fsaverage directory if it exists
    os.remove(lh_file)
    os.remove(rh_file)
    if os.path.exists(fsaverage_dst):
        import shutil
        shutil.rmtree(fsaverage_dst)


def test_elec_to_parc():

    
    # Get absolute path of ieeg2nwb directory
    ieeg2nwb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Source and destination paths for fsaverage
    sub_dir = os.path.join(ieeg2nwb_dir, "test_files", "freesurfer")

    try:
        output = elec_to_parc(SUBID, subjects_dir=sub_dir, parc="dk", write_to_file=False)
    except Exception as e:
        raise AssertionError(f"elec_to_parc() failed with error: {str(e)}")
    
    assert isinstance(output, dict), "elec_to_parc() output is not dict as expected"
    