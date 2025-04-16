import numpy as np

def read_xfm(xfm_file):
    """Read a Freesurfer transformation matrix file

    Parameters
    ----------
    xfm_file : str
        The path to the transformation matrix file

    Returns
    -------
    xfm : np.ndarray
        The transformation matrix
    """

    # Initialize variables
    matrix_lines = []
    matrix_start = False

    # Open and read the file
    with open(xfm_file, 'r') as file:
        for line in file:
            if "Linear_Transform" in line:
                matrix_start = True
                continue  # Skip the "Linear Transform" line

            if matrix_start:
                # Split the line into components (assuming space-separated values)
                matrix_lines.append(list(map(float, line.replace(";", "").split())))
                # Stop after reading 4 lines
                if len(matrix_lines) == 4:
                    break

    # Convert the list of lists into a NumPy array
    matrix = np.array(matrix_lines)
    return matrix