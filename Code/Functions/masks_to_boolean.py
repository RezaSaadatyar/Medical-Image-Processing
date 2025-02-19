import cv2 as cv
import numpy as np
from skimage import io
from Functions.filepath_extractor import FilePathExtractor

def masks_to_boolean(directory_path, format_type="TIF"):
    """
    Convert mask images from a specified directory into a boolean NumPy array.

    Args:
        directory_path (str): The path to the directory containing the mask images.
        format_type (str, optional): The file format of the mask images (default is "TIF").

    Returns:
        numpy.ndarray: A boolean NumPy array of shape [num_files, height, width, 2], where:
                      - masks[..., 0] represents the background (inverse of the mask).
                      - masks[..., 1] represents the foreground (actual mask).

    Raises:
        ValueError: If no files are found in the specified directory.
    """
    # Create an instance of FilePathExtractor to retrieve file paths
    obj_path = FilePathExtractor(directory_path, format_type)

    # Get a list of all file paths in the specified directory
    files_path = obj_path.all_files_path  # List of full file paths for the files

    # Check if the list of file paths is empty
    if not files_path:
        raise ValueError("No files found in the specified directory.")

    # Get the total number of image files
    num_files = len(files_path)  # Total number of image files

    # Read the first image to determine its dimensions
    mask = io.imread(files_path[0])  # Read the first mask image using scikit-image

    # Initialize a boolean NumPy array to store all masks
    # Shape: [num_files, height, width, 2], where 2 represents background and foreground
    masks = np.zeros((num_files, mask.shape[0], mask.shape[1], 2), dtype=bool)

    # Iterate through all the input files and process each mask
    for ind, val in enumerate(files_path):  # Progressively iterate through all the input files
        mask = np.squeeze(io.imread(val)).astype(bool)  # Load and convert each mask to boolean
        masks[ind, :, :, 0] = ~mask  # Background (inverse of mask)
        masks[ind, :, :, 1] = mask   # Foreground (actual mask)

    # Return the boolean NumPy array containing all masks
    return masks