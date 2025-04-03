# ================================ Presented by: Reza Saadatyar (2024-2025) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

import numpy as np
import tensorflow as tf 

from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_segmentation_predictions(test_dataset, predictions, num_sample=3):
    """
    Evaluate model predictions by calculating IoU and Dice coefficient for a batch of test images.

    Args:
    - test_dataset: Dataset containing test images and masks.
    - predictions: Model predictions (binary masks).
    
    Returns:
    - None: Prints IoU and Dice scores for each sample in the batch.
    
    IoU:
    - Computes Intersection over Union (IoU/Jaccard index) between masks. IoU measures the overlap between two samples
    with values ranging from 0 (no overlap) to 1 (perfect overlap). Defined as:
    IoU = |X ∩ Y| / |X ∪ Y|
    - Args:
        - y_true: Ground truth masks (tensor)
        - y_pred: Predicted masks (tensor)  
    - Returns:
      - Tensor: Scalar IoU value
      
    Dice coefficient:
    - Computes the Dice coefficient (F1 score) between ground truth and predicted masks. The Dice coefficient measures
    the overlap between two samples, with values ranging from 0 (no overlap) to 1 (perfect overlap). Defined as:
    Dice = (2 * |X ∩ Y|) / (|X| + |Y|)  
    - Args:
       - y_true: Ground truth masks (tensor)
       - y_pred: Predicted probability masks (tensor)
    - Returns:
       - Tensor: Scalar Dice coefficient value
    """
    # Convert predictions to binary (threshold at 0.5 for binary segmentation)
    predictions = (predictions > 0.5).astype(np.uint8)

    # Extract a small batch of data from test_dataset for visualization
    for batch in test_dataset.take(1):
        test_images, test_masks = batch
        break

    # Convert to numpy arrays for easier manipulation
    test_images = test_images.numpy()
    test_masks = test_masks.numpy()

    def iou(y_true, y_pred):
        # Ensure tensors are float32 for consistent computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Compute intersection
        intersection = tf.reduce_sum(y_true * y_pred)
        # Compute total area covered by both masks
        total = tf.reduce_sum(y_true + y_pred)
        # Compute union (total area minus intersection)
        union = total - intersection
        # Calculate IoU with smoothing factor
        return (intersection + 1e-4) / (union + 1e-4)

    def dice_coef(y_true, y_pred):
        # Ensure tensors are float32 for consistent computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Compute intersection (element-wise multiplication followed by sum)
        intersection = tf.reduce_sum(y_true * y_pred)
        # Compute union (sum of individual masks)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        # Calculate Dice coefficient with smoothing factor
        return (2. * intersection + 1e-4) / (union + 1e-4)

    # Calculate IoU and Dice for each sample in the batch
    for i in range(num_sample):
        iou_score = iou(test_masks[i], predictions[i])
        acc = accuracy_score(test_masks[i].flatten(), predictions[i].flatten())
        dice_score = dice_coef(test_masks[i], predictions[i])
        rec = recall_score(test_masks[i].flatten(), predictions[i].flatten(), labels=[0, 1], average="binary")
        pre = precision_score(test_masks[i].flatten(), predictions[i].flatten(), labels=[0, 1], average="binary")
        
        print(f"Sample {i+1} Accuracy: {acc:.3f}; IoU: {iou_score:.3f}; Dice coef: {dice_score:.3f}; Recall: {rec:.3f}; Precision: {pre:.3f}")