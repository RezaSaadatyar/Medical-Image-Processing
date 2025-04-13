# ================================ Presented by: Reza Saadatyar (2024-2025) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
class ShowImageTrainingProgress(keras.callbacks.Callback):
    """
    Custom callback to display input image, ground truth mask, and predicted mask for a specific sample
    at the end of the first epoch and the end of the last epoch.

    Example:
    - callbacks = [ShowImageTrainingProgress(train_dataset, sample_idx=0)]
    """

    def __init__(self, dataset, sample_idx=0, figsize=(8, 3.5)):
        """
        Args:
            dataset: tf.data.Dataset object containing (image, mask) pairs
            sample_idx: Index of the sample to display from the dataset (default: 0)
        """
        super().__init__()
        self.dataset = dataset
        self.figsize = figsize
        self.sample_idx = sample_idx
        self.num_epochs = None  # Will be set during training
        self.images = None
        self.masks = None
        self.predictions = {}  # Store predictions for first and last epochs

    def on_train_begin(self, logs=None):
        """Get the total number of epochs from the training configuration."""
        self.num_epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        """Capture prediction at the beginning of the first epoch."""
        if epoch == 0:  # First epoch
            self._store_images_and_predictions(epoch)

    def on_epoch_end(self, epoch, logs=None):
        """Capture prediction at the end of the last epoch and display the figure."""
        if epoch == self.num_epochs - 1:  # Last epoch
            self._store_images_and_predictions(epoch)
            self._display_images()

    def _store_images_and_predictions(self, epoch):
        """Store the input image, ground truth mask, and predicted mask for the specified epoch."""
        # Get a batch from the dataset and select the specified sample
        for batch in self.dataset.take(1):  # Take the first batch
            images, masks = batch
            self.images = images[self.sample_idx].numpy()  # Convert to NumPy
            self.masks = masks[self.sample_idx].numpy()  # Convert to NumPy

            # Predict using the model
            pred = self.model.predict(np.expand_dims(self.images, axis=0), verbose=0)[0]
            self.predictions[epoch] = pred  # Store prediction for this epoch

    def _display_images(self):
        """Display input image, ground truth mask, and predicted masks for the first and last epochs."""
        # Create a figure with 4 subplots
        plt.figure(figsize=self.figsize)

        # Input Image
        plt.subplot(1, 4, 1)
        plt.imshow(self.images)
        plt.title("Input Image", fontsize=10)
        plt.axis('off')

        # Ground Truth Mask
        plt.subplot(1, 4, 2)
        plt.imshow(np.squeeze(self.masks), cmap='gray')
        plt.title("Ground Truth Mask", fontsize=10)
        plt.axis('off')

        # Predicted Mask at First Epoch
        plt.subplot(1, 4, 3)
        plt.imshow(np.squeeze(self.predictions[0]), cmap='gray')
        plt.title(f"Predicted Mask; n_epoch 1", fontsize=10)
        plt.axis('off')

        # Predicted Mask at Last Epoch
        plt.subplot(1, 4, 4)
        plt.imshow(np.squeeze(self.predictions[self.num_epochs - 1]), cmap='gray')
        plt.title(f"Predicted Mask; n_epoch {self.num_epochs}", fontsize=10)
        plt.axis('off')

        plt.tight_layout()
        plt.show()