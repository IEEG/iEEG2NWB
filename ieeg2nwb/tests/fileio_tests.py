import mne
import os
from ieeg2nwb import read_ielvis
from ieeg2nwb.fileio.helpers import _read_atlas_labels, _read_electrodeNames, _read_coordinates, _read_ielvis_base, _read_ptd
from ieeg2nwb.fileio.freesurfer import read_xfm

SUBID = "NS162_02"

def test_read_ielvis(subject_dir=None):
    """
    Test reading ielvis data.
    
    Parameters
    ----------
    subject_dir : str, optional
        Path to the subject directory. If None, will try to use MNE's default
        SUBJECTS_DIR from config.
    """

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

    # Run read_ielvis()
    try:
        data = read_ielvis(SUBID, subjects_dir=sub_dir, extra_coords=True, write_to_file=True, parcs=True)
    except Exception as e:
        raise AssertionError(f"read_ielvis() failed with error: {str(e)}")
    
    # Verify data is returned and is a pandas DataFrame
    assert data is not None, "read_ielvis() returned None"

    # Clean up
    # Delete fsaverage directory if it exists
    if os.path.exists(fsaverage_dst):
        import shutil
        shutil.rmtree(fsaverage_dst)
    
    

def test_helpers():
    
    # Get absolute path of ieeg2nwb directory
    ieeg2nwb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(ieeg2nwb_dir, "test_files", "freesurfer", SUBID)
    elec_recon_dir = os.path.join(sub_dir, "elec_recon")


    # Run _read_electrodeNames()
    try:
        data = _read_electrodeNames(os.path.join(elec_recon_dir, SUBID + ".electrodeNames"))
    except Exception as e:
        raise AssertionError(f"_read_electrodeNames() failed with error: {str(e)}")
    
    # Run _read_coordinates()
    try:
        data = _read_coordinates(os.path.join(elec_recon_dir, SUBID + ".PIAL"))
    except Exception as e:
        raise AssertionError(f"_read_coordinates() failed with error: {str(e)}")
    
    # Verify data is returned and is a list of lists
    assert isinstance(data, list), "_read_coordinates() did not return a list"
    assert all(isinstance(x, list) for x in data), "_read_coordinates() did not return a list of lists"
    assert all(len(x) == 3 for x in data), "_read_coordinates() coordinates do not have 3 values (x,y,z)"
    
    # Run _read_ielvis_base()
    try:
        data = _read_ielvis_base(SUBID, os.path.join(ieeg2nwb_dir, "test_files", "freesurfer"))
    except Exception as e:
        raise AssertionError(f"_read_ielvis_base() failed with error: {str(e)}")
        
    # Verify data is returned and is a pandas DataFrame
    assert isinstance(data, pd.DataFrame), "_read_ielvis_base() did not return a pandas DataFrame"
    
    # Check required columns exist
    required_cols = ['label', 'spec', 'hem', 'LEPTO', 'LEPTOVOX', 'PIAL', 'PIALVOX'] 
    for col in required_cols:
        assert col in data.columns, f"_read_ielvis_base() missing required column: {col}"

    # Run _read_ptd()
    try:
        data = _read_ptd(os.path.join(elec_recon_dir, "GreyWhite_classifications.mat"))
    except Exception as e:
        raise AssertionError(f"_read_ptd() failed with error: {str(e)}")
        
    # Verify data is returned and has expected structure
    assert isinstance(data, dict), "_read_ptd() did not return a dictionary"
    assert 'elec' in data, "_read_ptd() missing 'elec' key in returned dictionary"
    assert isinstance(data['elec'], list), "_read_ptd() 'elec' value is not a list"


    # Run _read_atlas_labels()
    try:
        data = _read_atlas_labels(os.path.join(elec_recon_dir, SUBID + "_DK_AtlasLabels.tsv"))
    except Exception as e:
        raise AssertionError(f"_read_atlas_labels() failed with error: {str(e)}")
        
    # Verify data is returned and is a pandas DataFrame
    assert isinstance(data, pd.DataFrame), "_read_atlas_labels() did not return a pandas DataFrame"
    
    # Check required columns exist
    required_cols = ['label', 'region']
    for col in required_cols:
        assert col in data.columns, f"_read_atlas_labels() missing required column: {col}"

def test_tdt():
    pass

def test_freesurfer():
    pass