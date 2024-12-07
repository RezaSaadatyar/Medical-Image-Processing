import numpy as np  # Import numpy for array manipulation
from skimage import io  # Import scikit-image library for image I/O operations
from Functions import directory_reader

class ImageLoader:
    """
    A class to load and process images from a directory into a NumPy array.
    """

    def __init__(self, directory_path: str, format_type: str) -> None:
        """
        Initialize the ImageLoader class.

        :param directory_path: Root directory containing the images.
        :param format_type: File format (extension) of the images (e.g., ".tif").
        """
        self.images = None  # Placeholder for loaded image array
        self.img_width = None  # Width of the images
        self.img_height = None  # Height of the images
        self.img_channels = None  # Number of color channels in the images
        self.format_type: str = format_type  # File format to filter (e.g., ".tif")
        self.directory_path: str = directory_path  # Root directory path to scan

    @property
    def load_dataset(self) -> np.ndarray:
        """
        Load images from the specified directory into a NumPy array.

        :return: NumPy array containing the loaded images.
        """
        # Initialize the DirectoryReader to get file paths
        dir_reader = directory_reader.DirectoryReader(self.directory_path, self.format_type)
        file_paths = dir_reader.all_file_paths

        if not file_paths:
            raise ValueError("No files found in the specified directory.")

        num_files = len(file_paths)  # Total number of image files
        first_image = io.imread(file_paths[0])  # Load the first image to determine its dimensions

        # Check if the image is grayscale or colored
        if first_image.ndim == 2:  # Grayscale image
            self.img_height, self.img_width = first_image.shape  # Get image dimensions
            self.images = np.zeros(  # Initialize NumPy array for grayscale images
                (num_files, self.img_height, self.img_width),
                dtype=np.uint8  # Pixel values are stored as unsigned 8-bit integers
            )
        else:  # Colored image (e.g., RGB)
            self.img_height, self.img_width, self.img_channels = first_image.shape  # Get image dimensions
            self.images = np.zeros(  # Initialize NumPy array for colored images
                (num_files, self.img_height, self.img_width, self.img_channels),
                dtype=np.uint8
            )

        # Load all images into the NumPy array
        for idx, file_path in enumerate(file_paths):
            self.images[idx] = io.imread(file_path)  # Read and store each image

        return self.images  # Return the loaded images as a NumPy array