import os
import cv2
import shutil
from networkx import is_path
import numpy as np
from colorama import Fore
from platformdirs import AppDirs
from tensorflow import keras
from skimage import io, transform, color  # Import scikit-image library
from Functions.filepath_extractor import FilePathExtractor

class ImageProcessor:
    """
    A class for processing image data, including resizing.
    """
    def __init__(self):
        """
        Initialize the ImageProcessor class.

        **Import module:**
        - from Functions.image_processor import ImageProcessor
        
        **Example:**
        
        obj = ImageProcessor()
        1. Images into ndarray
           - data = obj.read_images(directory_path, format_type="tif")
           - print(Fore.GREEN + f"{data.shape = }")
        2. Masks into binary
           - masks = obj.masks_to_binary(directory_path, format_type="TIF")
           - print(Fore.GREEN + f"{masks.shape = }"
        3. Rgb into gray
           - img_gray = obj.rgb_to_gray(directory_path, save_path, format_type="tif", save_img_gray="off")
        4. Resize images
           - resized_images = obj.resize_images(data, img_height_resized=255, img_width_resized=255)  # Resize all images to 255x255
           - print(Fore.GREEN + f"Resizing images from {data.shape} to {resized_images.shape}")
        5. Augmentation
           - obj.augmentation(file_path, augmente_path, num_augmented_imag, rotation_range, format_type="tif")
        6. Crop images & masks
           - cropped_imgs, cropped_masks = obj.crop_images_using_masks(data_path, masks_path, data_format_type, mask_format_type)
        """
    # ============================================ Images convert to ndarray ===================================
    def read_images(self, image_path: str, format_type: str, resize: tuple = None, normalize: bool = False, 
                to_grayscale: bool = False) -> np.ndarray:
        """
        Convert images from a specified directory into a NumPy array with optional processing.

        Args:
        - image_path (str): The path to the directory containing the images.
        - format_type (str): The file format of the images (e.g., 'jpg', 'png').
        - resize (tuple, optional): Target size as (height, width). Default is None.
        - normalize (bool, optional): Normalize pixel values if max > 1. Default is False.
        - to_grayscale (bool, optional): Convert images to grayscale. Default is False.

        Returns:
        - numpy.ndarray: A NumPy array containing all processed images.

        Example:
        - obj = ImageProcessor()
        - imgs = obj.read_images(image_path, format_type="tif", resize=(224, 224), 
                                normalize=False, to_grayscale=False)

        Raises:
        - ValueError: If no files are found in the specified directory.
        """
        # Create an instance of FilePathExtractor to retrieve files path
        obj_path = FilePathExtractor(image_path, format_type)
        
        # Get a list of all files path in the specified directory
        files_path = obj_path.all_files_path

        # Check if the list of files path is empty
        if not files_path: raise ValueError("No files found in the specified directory.")

        # Get the total number of image files
        num_files = len(files_path)

        # Read the first image to determine base dimensions
        img = io.imread(files_path[0])
        # img = cv2.imread(files_path[0])
        
        # Convert to grayscale if specified
        if to_grayscale and img.ndim == 3: img = color.rgb2gray(img)
        # if to_grayscale: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Set target dimensions
        if resize:
            img_height, img_width = resize
        else:
            img_height, img_width = img.shape[:2]

        # Determine channels based on grayscale option
        channels = 1 if to_grayscale or img.ndim == 2 else img.shape[-1] if img.ndim == 3 else 1

        # Initialize array with appropriate shape
        imgs = np.zeros((num_files, img_height, img_width, channels),
                    dtype=np.float32 if normalize else np.uint8)

        # Load and process all images
        for idx, file_path in enumerate(files_path):
            # Read image
            img = io.imread(file_path)
            # img = cv2.imread(file_path)
            
            # Convert to grayscale if specified
            if to_grayscale and img.ndim == 3: img = color.rgb2gray(img)
            # if to_grayscale: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            # Reshape if grayscale image lacks channel dimension
            if img.ndim == 2: img = np.expand_dims(img, axis=-1)
                
            # Resize if specified
            if resize:
                img = transform.resize(img, (img_height, img_width, channels), preserve_range=True)
                # img = cv2.resize(img, (img_width, img_height))
            
            # Normalize if specified and max value exceeds 1
            if normalize and img.max() > 1: img = img / 255.0
                
            imgs[idx] = img
        
        print(Fore.GREEN + f"Image shape: {imgs.shape}")
        
        return imgs
    
    # ============================================ Masks convert to binary =====================================
    def read_masks(self, mask_path: str, format_type: str = "TIF", resize: tuple = None, 
                    normalize: bool = False) -> np.ndarray:
        """
        Convert mask images from a specified directory into a binary or multi-class NumPy array 
        based on the mask type, using OpenCV, and count the number of classes.

        Args:
        - mask_path (str): The path to the directory containing the mask images.
        - format_type (str, optional): The file format of the mask images (default is "TIF").
        - resize (tuple, optional): Target size as (height, width). Default is None.
        - normalize (bool, optional): Normalize pixel values to [0, 1] if max > 1. Default is False.

        Returns:
        - numpy.ndarray: A NumPy array of shape [num_files, height, width, 1], where:
            - Binary: 0 (background), 1 (foreground) if mask has 2 unique values.
            - Multi-class: Integer labels (e.g., 0, 1, 2, ...) if mask has >2 unique values.
            - Normalized to [0, 1] if normalize=True.

        Example:
        - obj = ImageProcessor()
        - masks = obj.masks_to_format(mask_path, format_type="TIF", resize=(224, 224))

        Raises:
        - ValueError: If no files are found in the specified directory.
        """
        # Create an instance of FilePathExtractor to retrieve file paths
        obj_path = FilePathExtractor(mask_path, format_type)
        files_path = obj_path.all_files_path

        if not files_path: raise ValueError("No files found in the specified directory.")

        num_files = len(files_path)
        # Read the first mask to determine dimensions and initial type
        mask = cv2.imread(files_path[0], cv2.IMREAD_GRAYSCALE)

        # Set target dimensions
        img_height, img_width = resize if resize else mask.shape[:2]

        # Determine if the mask is binary or multi-class based on unique values in first mask
        unique_values = np.unique(mask)
        is_binary = len(unique_values) <= 2  # Binary if 2 or fewer unique values

        # Initialize array; use float32 if normalizing, uint8 otherwise
        masks = np.zeros((num_files, img_height, img_width, 1),
                        dtype=np.float32 if normalize else np.uint8)

        # Process all masks
        for ind, file_path in enumerate(files_path):
            # Read mask in grayscale mode
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # Handle binary or multi-class conversion
            if is_binary:
                # Convert to binary (0 and 1)
                mask = (mask > 0).astype(np.uint8)
            else:
                # Preserve multi-class labels (no thresholding)
                mask = mask.astype(np.uint8)  # Ensure uint8 unless normalized

            # Normalize if specified and max value exceeds 1
            if normalize and mask.max() > 1:  mask = mask / mask.max()  # Scale to [0, 1] based on max value

            # Resize if specified
            if resize:
                mask = cv2.resize(mask, (img_width, img_height), 
                                interpolation=cv2.INTER_NEAREST)  # Preserve discrete values
                if is_binary:
                    mask = (mask > 0).astype(np.uint8)  # Re-binarize after resizing

            # Add channel dimension and store
            masks[ind] = np.expand_dims(mask, axis=-1)

        # Analyze the final masks array for class counts
        unique_classes, class_counts = np.unique(masks, return_counts=True)
        num_classes = len(unique_classes)

        # Print mask type and class information
        print(f"Detected mask type: {'Binary' if is_binary else 'Multi-class'} "
            f"based on first mask with {len(unique_values)} unique values")
        print(Fore.GREEN + f"{masks.shape = }")
        print(Fore.YELLOW + f"Total number of classes in masks: {num_classes}")

        # Detailed class counts
        if is_binary and not normalize:
            ones_count = np.sum(masks == 1)
            zeros_count = np.sum(masks == 0)
            print(Fore.YELLOW + f"Number of 1s (foreground): {ones_count}")
            print(Fore.YELLOW + f"Number of 0s (background): {zeros_count}")
        else:
            print(Fore.YELLOW + "Class counts in masks:")
            for cls, count in zip(unique_classes, class_counts):
                if normalize:
                    print(Fore.YELLOW + f"  Class {cls:.2f}: {count}")
                else:
                    print(Fore.YELLOW + f"  Class {int(cls)}: {count}")

        return masks
      
    # ============================================== RGB_to_Gray ===============================================
    def save_grayscale_images(images: np.ndarray, output_path: str, folder_name: str = "Gray image") -> None:
        """
        Save images as grayscale to a specified directory, converting from RGB/BGR if needed.

        Args:
        - images (np.ndarray): A NumPy array of shape (num_files, height, width, channels).
        - output_path (str): The directory path where grayscale images will be saved.
        - folder_name (str, optional): The name of the folder to save the images in. Defaults to "Gray image".

        Returns:
        - None: Saves converted images to disk if RGB/BGR, otherwise prints a message.

        Raises:
        - ValueError: If the input images have an unexpected number of channels (not 1 or 3).
        - OSError: If the output directory cannot be created or accessed, or if saving an image fails.
        """
        # Check if images is a NumPy array with the expected shape
        if not isinstance(images, np.ndarray) or len(images.shape) != 4:
            raise ValueError("Input 'images' must be a 4D NumPy array (num_files, height, width, channels)")

        num_files, height, width, channels = images.shape

        # Validate channel count
        if channels not in [1, 3]:
            raise ValueError(f"Unexpected number of channels: {channels}. Expected 1 (grayscale) or 3 (RGB/BGR)")

        # Case 1: Images are already grayscale (1 channel)
        if channels == 1:
            print("Images are already grayscale; no conversion or saving needed.")
            return

        # Case 2: Images are RGB/BGR (3 channels), convert to grayscale and save
        # Create output directory if it doesn’t exist
        output_folder_path = os.path.join(output_path, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        for i in range(num_files):
            # Extract the RGB/BGR image
            color_image = images[i, :, :, :]  # Shape: (height, width, 3)

            # If normalized (float values in [0, 1]), scale to [0, 255] for processing
            # This block ensures that the color_image is in the correct format (uint8) for OpenCV to process it correctly.
            # If the image's data type is float (either float32 or float64) and its maximum value is less than or equal to 1.0,
            # it's assumed to be a normalized image with pixel values in the range [0, 1].
            # In this case, the image is converted to the range [0, 255] by multiplying each pixel value by 255,
            # and then the data type is converted to uint8.
            if color_image.dtype in [np.float32, np.float64] and color_image.max() <= 1.0:
                color_image = (color_image * 255).astype(np.uint8)
            # If the image's data type is not already uint8, it's converted to uint8.
            # This handles cases where the image might be in a different integer format (e.g., int16, int32).
            elif color_image.dtype != np.uint8:
                color_image = color_image.astype(np.uint8)

            # Convert BGR to grayscale (assuming OpenCV’s BGR format; adjust if RGB)
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Define the output filename
            filename = os.path.join(output_folder_path, f"{i:02d}.png")

            # Save the grayscale image
            success = cv2.imwrite(filename, gray_image)
            if not success:
                raise OSError(f"Failed to save image: {filename}")

        print(f"Converted {num_files} RGB/BGR images to grayscale and saved to {output_folder_path}")
        
    
    
  

    # ============================================= Augmentation ===============================================
    def augmentation(self, directory_path: str, augmente_path: str, num_augmented_imag: int, rotation_range: int,
                 format_type: str) -> None:
        """
        AppDirs image augmentation (rotation) to images in the specified directory and saves them.

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
        - obj.augmentation(file_path, augmente_path, num_augmented_imag, rotation_range, format_type="tif")
          - rotation_range = 30
          - num_augmented_imag = 3
        """
        # Create a subfolder for rotated images within the augmented images directory
        augmente_path = os.path.join(augmente_path, 'Augmented/')

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

        # Get a list of all files path in the specified directory
        files_path = obj_path.all_files_path

        # Check if the list of files path is empty
        if not files_path: raise ValueError("No files found in the specified directory.")

        # Get corresponding filesname
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
            batch_size=1,                # ?Process one image at a time
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
    
    # ======================================= Mask-Based Image Cropping ========================================
    def mask_based_image_cropping(self, data_path: str, masks_path: str, data_format_type: str, mask_format_type: str) -> np.ndarray:
        """
        Crop images and their corresponding masks based on the mask boundaries.

        **Args:**
        - data_path (str): Path to the directory containing the images.
        - masks_path (str): Path to the directory containing the masks.
        - data_format_type (str): File format of the images (e.g., ".jpg", ".png").
        - mask_format_type (str): File format of the masks (e.g., ".png", ".tif").

        **Returns:**
        - tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - cropped_imgs: Array of cropped images with shape [num_images, height, width, channels].
            - cropped_masks: Array of cropped masks with shape [num_images, height, width].

        **Notes:**
        - The masks are expected to be binary (0 for background, 255 for foreground).
        - The images and masks are resized back to their original dimensions after cropping.
        
        **Example:**
        - obj = ImageProcessor()
        - cropped_imgs, cropped_masks = obj.crop_images_using_masks(data_path, masks_path, data_format_type, mask_format_type)
        """
        # Create an instance of FilePathExtractor for images and masks
        obj_data = FilePathExtractor(data_path, format_type=data_format_type)
        obj_masks = FilePathExtractor(masks_path, format_type=mask_format_type)

        # Get filenames and file paths for images and masks
        data_filesname = obj_data.filesname
        data_filespath = obj_data.all_files_path
        masks_filespath = obj_masks.all_files_path

        # Read the first image to determine its shape
        first_img_shape = io.imread(data_filespath[0]).shape

        # Initialize an array to store cropped images
        cropped_imgs = np.zeros((len(data_filesname), *first_img_shape), dtype=np.uint8)

        # Read the first mask to determine its shape
        first_mask_shape = io.imread(masks_filespath[0]).shape

        # Initialize an array to store cropped masks
        cropped_masks = np.zeros((len(masks_filespath), *first_mask_shape), dtype=bool)

        # Loop through each image and its corresponding mask
        for ind, val in enumerate(data_filespath):
            # Read the image and mask
            img = io.imread(val)
            mask = io.imread(masks_filespath[ind])

            # Find the coordinates of the mask's foreground (where mask == 255)
            y_coord, x_coord = np.where(mask == 255)

            # Calculate the bounding box for cropping
            y_min = min(y_coord)
            y_max = max(y_coord)
            x_min = min(x_coord)
            x_max = max(x_coord)

            # Crop and resize the image to the original dimensions
            cropped_imgs[ind] = transform.resize(
                img[y_min:y_max, x_min:x_max],
                first_img_shape,
                mode='constant',
                anti_aliasing=True,
                preserve_range=True
            )

            # Crop and resize the mask to the original dimensions
            cropped_masks[ind] = transform.resize(
                mask[y_min:y_max, x_min:x_max],
                first_mask_shape,
                mode='constant',
                anti_aliasing=True,
                preserve_range=True
            )

        # Return the cropped images and masks
        return cropped_imgs, cropped_masks
        