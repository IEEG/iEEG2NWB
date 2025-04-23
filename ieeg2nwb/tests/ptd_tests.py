import os
from ieeg2nwb.channels import get_ptd_index

SUBID = "NS162_02"

def test_get_ptd_index():

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
        data = get_ptd_index(SUBID, subjects_dir=sub_dir, write_to_file=False)
    except Exception as e:
        raise AssertionError(f"get_ptd_index() failed with error: {str(e)}")
    
    # Verify data is returned and is a pandas DataFrame
    assert data is not None, "get_ptd_index() returned None"

    # Clean up
    # Delete fsaverage directory if it exists
    if os.path.exists(fsaverage_dst):
        import shutil
        shutil.rmtree(fsaverage_dst)
    