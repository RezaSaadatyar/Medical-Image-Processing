# ================================ Presented by: Reza Saadatyar (2024-2025) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

class ShowImageTrainingProgress(keras.callbacks.Callback):
    """
    Custom callback to display input image, ground truth mask, predicted mask,
    and training loss plot at the end of training or when training stops early.

    Example:
    - callbacks = [ShowImageTrainingProgress(train_dataset, sample_idx=0)]
    """

    def __init__(self, dataset, sample_idx=0, figsize=(8, 3.5)):
        """
        Args:
            dataset: tf.data.Dataset object containing (image, mask) pairs
            sample_idx: Index of the sample to display from the dataset (default: 0)
            figsize: Figure size for the display (default: (8, 3.5))
        """
        super().__init__()
        self.dataset = dataset
        self.sample_idx = sample_idx
        self.figsize = figsize
        self.images = None
        self.masks = None
        self.predictions = {}  # Store predictions for first and last epochs
        self.epochs_trained = 0

    def on_train_begin(self, logs=None):
        """Initialize at the start of training."""
        self.predictions.clear()
        self.epochs_trained = 0

    def on_epoch_begin(self, epoch, logs=None):
        """Capture prediction at the start of the first epoch."""
        if epoch == 0:
            self._store_images_and_predictions(epoch)

    def on_epoch_end(self, epoch, logs=None):
        """Store loss and predictions, increment epoch count."""
        self.epochs_trained = epoch + 1
        # Store prediction for the current epoch
        self._store_images_and_predictions(epoch)

    def on_train_end(self, logs=None):
        """Display images and loss plot when training ends."""
        self._display_images_and_loss()

    def _store_images_and_predictions(self, epoch):
        """Store the input image, ground truth mask, and predicted mask for the specified epoch."""
        for batch in self.dataset.take(1):
            images, masks = batch
            self.images = images[self.sample_idx].numpy()
            self.masks = masks[self.sample_idx].numpy()
            pred = self.model.predict(np.expand_dims(self.images, axis=0), verbose=0)[0]
            self.predictions[epoch] = pred

    def _display_images_and_loss(self):
        """Display input image, ground truth mask, predicted masks, and loss plot."""
        fig = plt.figure(figsize=self.figsize)

        # Input Image
        plt.subplot(1, 4, 1)
        plt.imshow(self.images)
        plt.title("Input Image", fontsize=8)
        plt.axis('off')

        # Ground Truth Mask
        plt.subplot(1, 4, 2)
        plt.imshow(np.squeeze(self.masks), cmap='gray')
        plt.title("Ground Truth", fontsize=8)
        plt.axis('off')

        # Predicted Mask at First Epoch
        plt.subplot(1, 4, 3)
        plt.imshow(np.squeeze(self.predictions[0]), cmap='gray')
        plt.title("Pred. Epoch 1", fontsize=8)
        plt.axis('off')

        # Predicted Mask at Last Epoch
        last_epoch = min(self.epochs_trained - 1, max(self.predictions.keys()))
        plt.subplot(1, 4, 4)
        plt.imshow(np.squeeze(self.predictions[last_epoch]), cmap='gray')
        plt.title(f"Pred. Epoch {last_epoch + 1}", fontsize=8)
        plt.axis('off')

        plt.tight_layout()
        plt.show()