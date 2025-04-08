# ================================ Presented by: Reza Saadatyar (2024-2025) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

class ShowImageTrainingProgress(keras.callbacks.Callback):
    """
    Custom callback to display input image, ground truth mask, and predicted mask for a specific sample at the 
    beginning of the first epoch and end of the last epoch.
    
    Example:
    - callbacks = [ShowImageTrainingProgress(train_dataset, sample_idx=0)]
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
        self.num_epochs = None  # Will be set during training
 
    def on_train_begin(self, logs=None):
        """Get the total number of epochs from the training configuration."""
        self.num_epochs = self.params['epochs']
    
    #  Display images at the beginning of the first epoch (epoch 0)
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            self._display_images(epoch, "First Epoch Begin")

    # Display images at the end of the last epoch
    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.num_epochs - 1:
            self._display_images(epoch, "Last Epoch End")

    # Helper function to display input image, ground truth mask, and prediction
    def _display_images(self, epoch, title_prefix):
        # Get a batch from the dataset and select the specified sample
        for batch in self.dataset.take(1):  # Take the first batch
            images, masks = batch
            img = images[self.sample_idx].numpy()  # Convert to NumPy
            mask = masks[self.sample_idx].numpy()  # Convert to NumPy

            # Predict using the model
            pred = self.model.predict(np.expand_dims(img, axis=0), verbose=0)[0]

            # Create a figure with 3 subplots
            plt.figure(figsize=(7, 2.45))

            # Input Image
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title(f"Input Image; Epoch {epoch + 1}", fontsize=10)
            plt.axis('off')

            # Ground Truth Mask
            plt.subplot(1, 3, 2)
            plt.imshow(np.squeeze(mask), cmap='gray')
            plt.title(f"Ground Truth Mask; Epoch {epoch + 1}", fontsize=10)
            plt.axis('off')

            # Predicted Mask
            plt.subplot(1, 3, 3)
            plt.imshow(np.squeeze(pred), cmap='gray')
            plt.title(f"Predicted Mask; Epoch {epoch + 1}", fontsize=10)
            plt.axis('off')

            plt.tight_layout()
            plt.autoscale(enable=True, axis='both', tight=True)
            plt.show()