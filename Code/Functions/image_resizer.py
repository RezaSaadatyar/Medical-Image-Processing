import numpy as np
from skimage import transform  # Import transform module for resizing images

class ImageResizer:  # Define a class for resizing images
    def __init__(self, img_height_resized:int, img_width_resized:int, img_channels:int):
        """
        Initializes the ImageResizer with target dimensions.

        :param img_height_resized: Target height for resizing.
        :param img_width_resized: Target width for resizing.
        :param img_channels: Number of channels in the images (e.g., 3 for RGB).
        """
        self.img_height_resized = img_height_resized  # Store the target height for resizing
        self.img_width_resized = img_width_resized  # Store the target width for resizing
        self.img_channels = img_channels  # Store the number of channels in the images

    def resize_images(self, imgs:int):
        """
        Resizes a batch of images to the specified height, width, and channels.

        :param imgs: A numpy array of shape [num_images, height, width, channels].
        :return: A numpy array of resized images.
        """
        # Initialize an empty array to store resized images
        resized_imgs = np.zeros(
            (imgs.shape[0], self.img_height_resized, self.img_width_resized, self.img_channels),
            dtype=np.uint8
        )
        
        # Loop through each image in the batch
        for i in range(imgs.shape[0]):
            # Resize the image to the target dimensions and store it
            resized_imgs[i] = transform.resize(
                imgs[i],
                (self.img_height_resized, self.img_width_resized, self.img_channels),
                preserve_range=True  # Preserve the range of pixel values
            )
        
        return resized_imgs  # Return the resized images