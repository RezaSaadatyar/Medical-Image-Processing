import cv2 as cv
import numpy as np
from skimage import io  # Import scikit-image library for image I/O operations
from Functions.filepath_extractor import FilePathExtractor

def imgs_to_ndarray(directory_path, format_type):
    """
    Convert images from a specified directory into a NumPy array.

    Args:
        directory_path (str): The path to the directory containing the images.
        format_type (str): The file format of the images (e.g., 'jpg', 'png').

    Returns:
        numpy.ndarray: A NumPy array containing all the images. If the images are grayscale,
                      the array shape will be (num_files, height, width). If the images are
                      colored (e.g., RGB), the array shape will be (num_files, height, width, channels).

    Raises:
        ValueError: If no files are found in the specified directory.
    """
    # Create an instance of FilePathExtractor to retrieve file paths
    obj_path = FilePathExtractor(directory_path, format_type=format_type)
    
    # Get a list of all file paths in the specified directory
    files_path = obj_path.all_files_path  # List of full file paths for the files

    # Check if the list of file paths is empty
    if not files_path:
        raise ValueError("No files found in the specified directory.")

    # Get the total number of image files
    num_files = len(files_path)  # Total number of image files

    # Read the first image to determine its dimensions and type (grayscale or colored)
    img = cv.imread(files_path[0])[..., ::-1]  # Read the first image and reverse color channels (BGR to RGB)

    # Check if the image is grayscale or colored
    if img.ndim == 2:  # Grayscale image (2D array)
        img_height, img_width = img.shape  # Get image dimensions (height, width)
        
        # Initialize an empty NumPy array to store all grayscale images
        img = np.zeros((num_files, img_height, img_width),
                        dtype=np.uint8  # Pixel values are stored as unsigned 8-bit integers
                    )
    else:  # Colored image (e.g., RGB) (3D array)
        img_height, img_width, img_channels = img.shape  # Get image dimensions (height, width, channels)
        
        # Initialize an empty NumPy array to store all colored images
        img = np.zeros((num_files, img_height, img_width, img_channels),
                        dtype=np.uint8
                    )

    # Load all images into the NumPy array
    for idx, file_path in enumerate(files_path):
        img[idx] = cv.imread(file_path)  # Read and store each image in the array

    # Return the NumPy array containing all images
    return img