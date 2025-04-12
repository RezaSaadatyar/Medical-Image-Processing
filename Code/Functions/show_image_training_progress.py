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
        callbacks = [ShowImageTrainingProgress(train_dataset, sample_idx=0)]
    """
    
    def __init__(self, dataset, sample_idx=0):
        """
        Args:
            dataset: tf.data.Dataset object containing (image, mask) pairs
            sample_idx: Index of the sample to display from the dataset (default: 0)
        """
        super().__init__()
        self.dataset = dataset
        self.sample_idx = sample_idx
        self.num_epochs = None
    
    def on_train_begin(self, logs=None):
        """Store the total number of epochs."""
        self.num_epochs = self.params['epochs']
    
    def on_epoch_end(self, epoch, logs=None):
        """Display images at the end of the first and last epochs."""
        if epoch == 0 or epoch == self.num_epochs - 1:
            self._display_images(epoch, "Epoch End")

    def _display_images(self, epoch, title_prefix):
        """Helper function to display input image, ground truth mask, and predicted mask."""
        # Get a batch from the dataset
        for batch in self.dataset.take(1):
            images, masks = batch
            img = images[self.sample_idx].numpy()
            mask = masks[self.sample_idx].numpy()

            # Predict using the model
            pred = self.model.predict(np.expand_dims(img, axis=0), verbose=0)[0]

            # Ensure predicted mask is binarized if needed (e.g., for segmentation)
            if pred.shape[-1] == 1:  # Sigmoid output
                pred = (pred > 0.5).astype(np.float32)
            else:  # Softmax output
                pred = np.argmax(pred, axis=-1).astype(np.float32)

            # Plotting
            plt.figure(figsize=(10, 3))

            # Input Image
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title(f"Input Image\nEpoch {epoch + 1}", fontsize=10)
            plt.axis('off')

            # Ground Truth Mask
            plt.subplot(1, 3, 2)
            plt.imshow(np.squeeze(mask), cmap='gray')
            plt.title(f"Ground Truth\nEpoch {epoch + 1}", fontsize=10)
            plt.axis('off')

            # Predicted Mask
            plt.subplot(1, 3, 3)
            plt.imshow(np.squeeze(pred), cmap='gray')
            plt.title(f"Predicted Mask\nEpoch {epoch + 1}", fontsize=10)
            plt.axis('off')

            plt.tight_layout()
            plt.show()