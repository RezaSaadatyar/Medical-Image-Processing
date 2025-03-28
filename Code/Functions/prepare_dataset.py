import pprint
import numpy as np
import tensorflow as tf
from typing import Tuple, Union
from sklearn.model_selection import train_test_split

def prepare_dataset(
    data: Union[np.ndarray, tf.Tensor], 
    labels: Union[np.ndarray, tf.Tensor], 
    train_size: float = 0.8, 
    valid_size: float = 0.16, 
    batch_size: int = 16, 
    shuffle_train: bool = True, 
    shuffle_buffer_size: int = 1000
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    This function handles the complete pipeline from raw data to optimized TensorFlow Dataset objects,
    including data splitting, shuffling, batching, and prefetching for optimal performance.
    
    Args:
        data: Input features as either numpy array or TensorFlow tensor
               Shape should be (num_samples, ...features_dims)
        labels: Corresponding labels as either numpy array or TensorFlow tensor
                Shape should be (num_samples, ...label_dims)
        train_size: Proportion of total data to use for training (0.0 to 1.0)
        valid_size: Proportion of total data to use for validation (0.0 to 1.0)
        batch_size: Number of samples per training batch (positive integer)
        shuffle_train: Whether to shuffle training data (recommended for training)
        shuffle_buffer_size: Size of buffer used for shuffling (larger = better shuffling but more memory)
    
    Returns:
        A tuple containing three tf.data.Dataset objects in order:
        - train_dataset: Dataset for model training
        - valid_dataset: Dataset for validation during training
        - test_dataset: Dataset for final evaluation
    
    Raises:
        ValueError: If input sizes are invalid or data/labels have mismatched lengths
    """
    
    # ============================================ INPUT VALIDATION ============================================
    # Validate the split proportions make sense
    if train_size + valid_size > 1.0:
        raise ValueError("train_size + valid_size must not exceed 1.0 (test_size would be negative)")
    if train_size < valid_size:
        raise ValueError("Training set should typically be larger than validation set")

    # Convert TensorFlow tensors to numpy arrays for sklearn splitting
    if tf.is_tensor(data):
        data = data.numpy()
    if tf.is_tensor(labels):
        labels = labels.numpy()

    # Verify data and labels have compatible shapes
    if len(data) != len(labels):
        raise ValueError(f"Mismatched lengths: data has {len(data)} samples but labels has {len(labels)}")

    # ======================================== DATA SPLITTING ==================================================
    # First split separates test set from training+validation
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        data,
        labels,
        train_size=train_size + valid_size,  # Combined size for train+val
        test_size=1.0 - (train_size + valid_size),  # Remainder for test
        random_state=24,  # Fixed seed for reproducibility
        # stratify=labels if len(set(labels)) > 1 else None  # Optional stratification
    )

    # Second split divides train+val into separate sets
    # Calculate relative proportion of validation within train+val subset
    valid_proportion = valid_size / (train_size + valid_size)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_val,
        y_train_val,
        train_size=1.0 - valid_proportion,  # Relative train size
        test_size=valid_proportion,          # Relative validation size
        random_state=24,  # Same seed for consistency
        # stratify=y_train_val if len(set(y_train_val)) > 1 else None
    )

    # ======================================= DATASET CREATION =================================================
    # Create TensorFlow Dataset objects with proper type casting
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train, tf.float32),  # Convert features to float32
        tf.cast(y_train, tf.float32)    # Convert labels to float32
    ))
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_valid, tf.float32), 
        tf.cast(y_valid, tf.float32))
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_test, tf.float32), 
        tf.cast(y_test, tf.float32))
    )
    
    # ====================================== DATASET OPTIMIZATION ==============================================
    # Shuffle training data if enabled (recommended for better training)
    if shuffle_train:
        train_dataset = train_dataset.shuffle(
            buffer_size=min(shuffle_buffer_size, len(x_train)),  # Don't exceed dataset size
            reshuffle_each_iteration=True  # Important for proper epoch training
        )
    
    # Batch all datasets for efficient processing
    train_dataset = train_dataset.batch(batch_size)
    valid_dataset = valid_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    
    # Prefetch data to overlap preprocessing and model execution
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # Let TensorFlow optimize buffer size
    valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    # ============================================ VERIFICATION OUTPUT =========================================
    print(f"Training set:   {x_train.shape} features, {y_train.shape} labels")
    print(f"Validation set: {x_valid.shape} features, {y_valid.shape} labels")
    print(f"Test set:       {x_test.shape} features, {y_test.shape} labels")
    print(f"\nBatch size:     {batch_size}")
    print(f"Training shuffle: {'enabled' if shuffle_train else 'disabled'}")
    print("\t")
    pprint.pprint(train_dataset.element_spec, width=80)
    
    return train_dataset, valid_dataset, test_dataset