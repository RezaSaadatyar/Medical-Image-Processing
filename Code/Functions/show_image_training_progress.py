# ================================ Presented by: Reza Saadatyar (2024-2025) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def show_image_training_progress(dataset, model, sample_idx=0, figsize=(8, 3.5)):
    """
    Create a callback function to display input image, ground truth mask, predicted mask for Epoch 1,
    and predicted mask when training stops in a single row with four subplots.

    Args:
        dataset: tf.data.Dataset object containing (image, mask) pairs
        model: The Keras model being trained (used for predictions)
        sample_idx: Index of the sample to display from the dataset (default: 0)

    Returns:
        A Keras LambdaCallback to be used during training.
    """
    # Dictionary to store data for Epoch 1
    epoch_1_data = {'img': None, 'mask': None, 'pred': None}
    # Variable to track the last epoch
    last_epoch = 0

    def on_epoch_end(epoch, logs=None):
        """Process images at the end of the first epoch and track epochs."""
        nonlocal last_epoch
        last_epoch = epoch + 1  # Update last_epoch (epoch is 0-based, so +1 for display)
        # Store data only for Epoch 1
        if epoch == 0:
            for batch in dataset.take(1):
                images, masks = batch
                img = images[sample_idx].numpy()
                mask = masks[sample_idx].numpy()
                pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]

                # Handle the predicted mask based on model output
                if pred.shape[-1] == 1:  # Sigmoid output (binary segmentation)
                    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # Normalize to [0, 1]
                    pred = (pred > 0.5).astype(np.float32)  # Threshold to binary
                else:  # Softmax output (multi-class segmentation)
                    pred = np.argmax(pred, axis=-1).astype(np.float32)
                    if pred.max() > 0:
                        pred = pred / pred.max()

                epoch_1_data['img'] = img
                epoch_1_data['mask'] = mask
                epoch_1_data['pred'] = pred

    def on_train_end(logs=None):
        """Display images when training stops."""
        # Get a batch from the dataset for the final prediction
        for batch in dataset.take(1):
            images, masks = batch
            img = images[sample_idx].numpy()
            pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]

            # Handle the predicted mask based on model output
            if pred.shape[-1] == 1:  # Sigmoid output (binary segmentation)
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # Normalize to [0, 1]
                pred = (pred > 0.5).astype(np.float32)  # Threshold to binary
            else:  # Softmax output (multi-class segmentation)
                pred = np.argmax(pred, axis=-1).astype(np.float32)
                if pred.max() > 0:
                    pred = pred / pred.max()

            # Create a figure with 1 row and 4 columns
            plt.figure(figsize=figsize)

            # Epoch 1 - Input Image
            plt.subplot(1, 4, 1)
            plt.imshow(epoch_1_data['img'], cmap='gray')
            plt.title("Input Image", fontsize=10)
            plt.axis('off')

            # Epoch 1 - Ground Truth Mask
            plt.subplot(1, 4, 2)
            plt.imshow(np.squeeze(epoch_1_data['mask']), cmap='gray')
            plt.title("Ground Truth Mask", fontsize=10)
            plt.axis('off')

            # Epoch 1 - Predicted Mask
            plt.subplot(1, 4, 3)
            plt.imshow(np.squeeze(epoch_1_data['pred']), cmap='gray')
            plt.title("Predicted Mask; nEpoch 1", fontsize=10)
            plt.axis('off')

            # Final - Predicted Mask
            plt.subplot(1, 4, 4)
            plt.imshow(np.squeeze(pred), cmap='gray')
            plt.title(f"Predicted Mask; nEpoch {last_epoch}", fontsize=10)  # Use last_epoch
            plt.axis('off')

            plt.tight_layout()
            plt.show()

    # Return a LambdaCallback that handles epoch end and train end
    return keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end, on_train_end=on_train_end)

# class ShowImageTrainingProgress(keras.callbacks.Callback):
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