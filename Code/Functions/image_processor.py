import os
import shutil
import cv2 as cv
import numpy as np
from tensorflow import keras
from skimage import io, transform  # Import scikit-image library
from Functions.filepath_extractor import FilePathExtractor

class ImageProcessor:
    """
    A class for processing image data, including resizing.
    """
    def __init__(self):
        """
        Initialize the ImageProcessor class.

        :param img_data: A numpy array of shape [num_images, height, width, channels].
        
        **Import module:**
        - from Functions.image_processor import ImageProcessor
        """
        self.img_gray = None  # Placeholder for grayscale images after processing
        self.file_names = None  # Placeholder for image file names
        self.file_path = None  # Placeholder for the path to the image files
        self.augmente_path = None  # Placeholder for the path to store augmented images

    # ============================================ Images convert to ndarray ===================================
    def imgs_to_ndarray(self, directory_path:str, format_type:str) -> np.ndarray:
        """
        Convert images from a specified directory into a NumPy array.

        **Args:**
        - directory_path (str): The path to the directory containing the images.
        - format_type (str): The file format of the images (e.g., 'jpg', 'png').

        **Returns:**
        - numpy.ndarray: A NumPy array containing all the images. If the images are grayscale, the array shape will be (num_files, height, width). If the images are colored (e.g., RGB), the array shape will be (num_files, height, width, channels).

        **Example:**
        - obj = ImageProcessor()
        - imgs = obj.imgs_to_ndarray(directory_path, format_type="tif")

        **Raises:**
        - ValueError: If no files are found in the specified directory.
        """
        # Create an instance of FilePathExtractor to retrieve file paths
        obj_path = FilePathExtractor(directory_path, format_type)
        
        # Get a list of all file paths in the specified directory
        files_path = obj_path.all_files_path  # List of full file paths for the files

        # Check if the list of file paths is empty
        if not files_path: raise ValueError("No files found in the specified directory.")

        # Get the total number of image files
        num_files = len(files_path)  # Total number of image files

        # Read the first image to determine its dimensions and type (grayscale or colored)
        img = cv.imread(files_path[0])[..., ::-1]  # Read the first image and reverse color channels (BGR to RGB)

        # Check if the image is grayscale or colored
        if img.ndim == 2:  # Grayscale image (2D array)
            img_height, img_width = img.shape  # Get image dimensions (height, width)
            
            # Initialize an empty NumPy array to store all grayscale images
            imgs = np.zeros((num_files, img_height, img_width),
                            dtype=np.uint8  # Pixel values are stored as unsigned 8-bit integers
                        )
        else:  # Colored image (e.g., RGB) (3D array)
            img_height, img_width, img_channels = img.shape  # Get image dimensions (height, width, channels)
            
            # Initialize an empty NumPy array to store all colored images
            imgs = np.zeros((num_files, img_height, img_width, img_channels),
                            dtype=np.uint8
                        )

        # Load all images into the NumPy array
        for idx, file_path in enumerate(files_path):
            imgs[idx] = cv.imread(file_path)  # Read and store each image in the array

        # Return the NumPy array containing all images
        return imgs
    
    # ============================================ Images convert to ndarray ===================================
    def masks_to_boolean(self, directory_path:str, format_type:str="TIF") -> bool:
        """
        Convert mask images from a specified directory into a boolean NumPy array.

        **Args:**
        - directory_path (str): The path to the directory containing the mask images.
        - format_type (str, optional): The file format of the mask images (default is "TIF").

        **Returns:**
        - numpy.ndarray: A boolean NumPy array of shape [num_files, height, width, 2], where:
            - masks[..., 0] represents the background (inverse of the mask).
            - masks[..., 1] represents the foreground (actual mask).

        **Example:**
        - obj = ImageProcessor()
        - masks = obj.masks_to_boolean(directory_path, format_type="TIF")

        **Raises:**
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
    
    # ================================================ Resizes =================================================
    def resize_images(self, data: np.ndarray, img_height_resized: int, img_width_resized: int) -> np.ndarray:
        """
        Resizes a batch of images to the specified height and width.

        **Args:**
        - data (np.ndarray): A batch of images as a NumPy array. The shape can be:
            - [num_images, height, width] for grayscale images.
            - [num_images, height, width, channels] for colored images.
        - img_height_resized (int): Target height of the images after resizing.
        - img_width_resized (int): Target width of the images after resizing.

        **Returns:**
        - np.ndarray: A NumPy array of resized images with the same number of dimensions as the input.

        **Notes:**
        - Grayscale images are resized to shape [num_images, img_height_resized, img_width_resized].
        - Colored images are resized to shape [num_images, img_height_resized, img_width_resized, channels].
        - The `preserve_range=True` argument ensures that the pixel value range is maintained during resizing.

        **Example:**
        - obj = ImageProcessor()
        - resized_images = obj.resize_images(data, img_height_resized=255, img_width_resized=255) 
        """
        # Check if the input is grayscale (3D) or colored (4D)
        if data.ndim == 3:  # Grayscale images (no color channels)
            img_channels = 1  # Grayscale implies 1 channel
            # Initialize an array to store resized grayscale images
            resized_imgs = np.zeros(
                (data.shape[0], img_height_resized, img_width_resized),
                dtype=np.uint8  # Use unsigned 8-bit integers for pixel values
            )

            # Loop through each grayscale image in the batch
            for i in range(data.shape[0]):
                # Resize the image to the target dimensions and store it
                resized_imgs[i] = transform.resize(
                    data[i], (img_height_resized, img_width_resized),
                    preserve_range=True  # Preserve the range of pixel values
                )

        else:  # Colored images (4D array with channels)
            img_channels = data.shape[-1]  # Get the number of color channels
            # Initialize an array to store resized colored images
            resized_imgs = np.zeros(
                (data.shape[0], img_height_resized, img_width_resized, img_channels),
                dtype=np.uint8  # Use unsigned 8-bit integers for pixel values
            )

            # Loop through each colored image in the batch
            for i in range(data.shape[0]):
                # Resize the image to the target dimensions and store it
                resized_imgs[i] = transform.resize(
                    data[i], (img_height_resized, img_width_resized),
                    preserve_range=True  # Preserve the range of pixel values
                )

        # Return the resized images
        return resized_imgs
    
    # ============================================= Augmentation ===============================================
    def augmentation(self, directory_path: str, augmente_path: str, num_augmented_imag: int, rotation_range: int,
                 format_type: str) -> None:
        """
        Applies image augmentation (rotation) to images in the specified directory and saves them.

        **Args:**
        - directory_path (str): Path to the directory containing the images.
        - augmente_path (str): Path to the directory to save augmented images.
        - num_augmented_imag (int): Number of augmented images to generate.
        - rotation_range (int): Degree range for random image rotation.
        - format_type (str): File format to filter images (e.g., ".jpg", ".png").

        **Notes:**
        - A temporary folder is created inside the input directory for processing.
        - The `ImageDataGenerator` from Keras is used for image augmentation.
        - Augmented images are saved in a subfolder named 'Rotated' within the `augmente_path`.
        
        **Example:**
        - obj = ImageProcessor()
        - img.augmentation(file_path, augmente_path, num_augmented_imag, rotation_range, format_type="tif")
          - rotation_range = 30
          - num_augmented_imag = 3
        """
        # Create a subfolder for rotated images within the augmented images directory
        augmente_path = os.path.join(augmente_path, 'Rotated/')

        # Check if the augmented images folder exists, delete it if so
        if os.path.exists(augmente_path):
            shutil.rmtree(augmente_path)  # Delete the folder and its contents
        os.makedirs(augmente_path, exist_ok=True)  # Recreate the folder

        # Create a temporary folder inside the input directory for processing
        TEMP_DIR = os.path.join(directory_path, 'Temp/')
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)  # Delete the temporary folder if it exists
        os.makedirs(TEMP_DIR, exist_ok=True)  # Recreate the temporary folder

        # Create an instance of FilePathExtractor to retrieve file paths
        obj_path = FilePathExtractor(directory_path, format_type)

        # Get a list of all file paths in the specified directory
        files_path = obj_path.all_files_path  # List of full file paths for the files

        # Check if the list of file paths is empty
        if not files_path:
            raise ValueError("No files found in the specified directory.")

        # Get corresponding filenames
        file_names = obj_path.filesname

        # Read the first image to determine its dimensions
        dat = io.imread(files_path[0])

        # Copy all files from the main folder to the temporary folder
        for ind, val in enumerate(files_path):
            shutil.copy(val, os.path.join(TEMP_DIR, file_names[ind]))

        # Set up the ImageDataGenerator for image augmentation
        Data_Gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=rotation_range)

        # Use flow_from_directory to process the images in the Temp folder
        img_aug = Data_Gen.flow_from_directory(
            directory_path,              # Parent directory of Temp
            classes=['Temp'],            # Specify the subfolder 'Temp' as the target
            batch_size=1,                # Process one image at a time
            save_to_dir=augmente_path,   # Save augmented images to the Rotated folder
            save_prefix='Aug',           # Prefix for augmented images
            target_size=(dat.shape[0], dat.shape[1]),  # Resize images to the specified dimensions
            class_mode=None              # No labels, as we're working with unclassified images
        )

        # Generate augmented images and save them
        for _ in range(num_augmented_imag):
            next(img_aug)  # Process the next image and save it

        # Delete the temporary folder and its contents after processing
        shutil.rmtree(TEMP_DIR)