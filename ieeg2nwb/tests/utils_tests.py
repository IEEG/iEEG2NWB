import pandas as pd
import os
import filecmp
import shutil
from ieeg2nwb.utils import copy_fsaverage_data, compress_data, inspectNwb, read_aseg_csv, timenow


def test_nwb_utils():
    



def test_misc_utils():

    # timenow() function
    try:
        t = timenow()
    except Exception as e:
        raise AssertionError(f"timenow() failed with error: {str(e)}")
    assert isinstance(t, str), "timenow() did not return string"

    # read_aseg_csv()
    try:
        df = read_aseg_csv()
    except Exception as e:
        raise AssertionError(f"read_aseg_csv() failed with error: {str(e)}")
    assert isinstance(df, pd.DataFrame), "read_aseg_csv() did not return string"


    # read_aseg_csv()
    try:
        df = read_aseg_csv()
    except Exception as e:
        raise AssertionError(f"read_aseg_csv() failed with error: {str(e)}")
    assert isinstance(df, pd.DataFrame), "read_aseg_csv() did not return string"

    # copy_fsaverage_data()

    # Get absolute path of ieeg2nwb directory
    ieeg2nwb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orig_fsaverage_dir = os.path.join(ieeg2nwb_dir, "data", "fsaverage")    
    fs_test_dir = os.path.join(ieeg2nwb_dir, "test_files", "freesurfer_test")
    test_fsavg_dir = os.path.join(fs_test_dir, "fsaverage")
    os.makedirs(test_fsavg_dir)
    try:
        copy_fsaverage_data(test_fsavg_dir)
    except Exception as e:
        raise AssertionError(f"copy_fsaverage_data() failed with error: {str(e)}")
    
    # Check 
    comparison = filecmp.dircmp(orig_fsaverage_dir, test_fsavg_dir, ignore=['.DS_Store'])
    assert len(comparison.left_only)==0, "Not all files copied in copy_fsaverage_data()"

    # Clean up
    shutil.rmtree(fs_test_dir)


