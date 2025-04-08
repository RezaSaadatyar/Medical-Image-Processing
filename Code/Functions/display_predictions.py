# ================================ Presented by: Reza Saadatyar (2024-2025) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

from matplotlib.pyplot import plt

def display_predictions(test_images, test_masks, predictions, indx=0, figsize=(6, 4)):
    plt.figure(figsize=figsize)

    # Original Image
    plt.subplot(131)
    plt.imshow(test_images[indx], cmap='gray')
    plt.title(f"Test Image {indx + 1}", fontsize=10)
    plt.axis('off')

    # Ground Truth Mask
    plt.subplot(132)
    plt.imshow(test_masks[indx], cmap='gray')
    plt.title(f"Ground Truth Mask {indx + 1}", fontsize=10)
    plt.axis('off')

    # Predicted Mask
    plt.subplot(133)
    plt.imshow(predictions[indx], cmap='gray')
    plt.title(f"Predicted Mask {indx + 1}", fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.show()