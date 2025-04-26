## Medical Image Processing
This repository contains code and resources for processing and analyzing medical images using deep learning techniques. The project focuses on implementing and experimenting with U-Net and ResNet models for tasks such as segmentation, classification, and evaluation of medical images.

### Overview
`Image classification` assigns *a single label to an entire image*, while `segmentation` goes a step further by *labeling each pixel or group of pixels*, effectively dividing the image into meaningful components. This enables more detailed analysis and supports applications like object detection, medical imaging, and autonomous driving, where identifying specific parts of an image is essential.

#### Image Segmentation Types

- **Semantic Segmentation:**
  - Assigns each pixel to a specific class.
  - Doesn't distinguish between instances of the same class.
  - Example: All cars are labeled as "car" class.
- **Instance Segmentation:**
  - Detects and segments individual objects.
  - Distinguishes between instances of the same class.
  - Example: Each car gets a unique label (car1, car2).
- **Panoptic Segmentation:**
  - Combines semantic and instance segmentation.
  - Segments both countable objects and background.
  - Example: Cars as instances + road/sky as semantic.
- **Binary Segmentation:**
  - Segments the image into foreground and background.
  - Only two classes (0 or 1).
  - Example: Object vs background.
- **Real-time Segmentation:**
  - Fast processing for video/streaming.
  - Often trades accuracy for speed.
  - Example: Self-driving car vision systems.

#### Segmentation Methods

- **Manual Annotation:** Involves marking image boundaries or regions of interest. Though reliable, it is time-consuming, labor-intensive, prone to errors, and unsuitable for large datasets or complex tasks due to consistency challenges.
- **Pixel-wise Classification:** Independently classifies each pixel using algorithms like decision trees, SVMs, or random forests. However, it often fails to capture global context, spatial relationships, and accurate object boundaries, leading to segmentation issues.
- **U-Net's Architecture:** Features a contracting path and an expansive path. The contracting path, similar to feedforward layers in other CNNs, uses encoder layers to reduce spatial resolution while capturing contextual information. The expansive path uses decoder layers to upsample feature maps and decode the data, aided by skip connections from the contracting path to preserve spatial details. This combination enables accurate feature localization and segmentation map generation.

#### U-Net Framework

**UNET** was developed to address the inefficiencies and inaccuracies of traditional image segmentation methods.
- **End-to-End Learning:** UNET employs an end-to-end learning approach, enabling it to segment images directly from input-output pairs without requiring user annotations. By training on a large, labeled dataset, UNET automatically extracts essential features and delivers precise segmentation, eliminating the need for time-consuming manual labeling.
- **Fully Convolutional Architecture:** UNET utilizes a fully convolutional architecture, consisting solely of convolutional layers without any fully connected layers. This design allows UNET to process input images of any size, enhancing its versatility and adaptability for diverse segmentation tasks and varying input dimensions.
- **U-shaped Architecture with Skip Connections:** The network's architecture features encoding and decoding paths for local and global context, with skip connections preserving key information for precise segmentation.
- **Contextual Information and Localisation:** Skip connections in UNET merge multi-scale features, enhancing context absorption, detail capture, and improving segmentation accuracy with precise object boundaries.
- **Data Augmentation and Regularization:** UNET enhances resilience and generalization by using data augmentation, like rotations, flips, scaling, and deformations, to diversify training data, and regularization techniques, such as dropout and batch normalization, to prevent overfitting.

#### Key Elements of the U-Net Framework

- **Contracting Path (Encoding Path):** Uses convolution and max pooling techniques to capture high-resolution, low-level features while reducing spatial dimensions. Each downsampling step doubles the number of feature channels, allowing for the extraction of features at various scales from the input image.
- **Expanding Path (Decoding Path):** Transposed convolutions, or deconvolutions, upsample feature maps to reconstruct a detailed segmentation map. This technique restores the features to match the resolution of the input image, ensuring precise alignment.
- **Bottleneck (Bottom):** Bridge between contracting and expansive paths. Has the highest number of feature channels.
- **Skip Connections:** Link encoding and decoding layers, preserving spatial details and enhancing segmentation by merging features from earlier layers. It concatenates the encoder feature map with the decoder, which helps the backward flow of gradients for improved training. After every concatenation, two consecutive regular convolutions are applied to assemble a more precise output.
- **Fully Convolutional Layers:** Uses convolutional layers, avoiding fully connected ones, allowing it to process images of any size while retaining spatial information for versatile segmentation tasks.
- **Final Layer:** 1x1 convolution to map feature vector to the desired number of classes. Output is a pixel-wise segmentation map.
- **Dice Coefficient Loss:** Measures overlap between predicted and true segmentation masks. Dice loss (1 - Dice coefficient) minimizes as alignment improves. It's particularly effective for unbalanced datasets, encouraging accurate separation of foreground and background by penalizing false positives and negatives.
- **Cross-Entropy Loss:** Measures dissimilarity between predicted class probabilities and ground truth labels. It treats each pixel as an independent classification problem, encouraging high probabilities for correct classes and penalizing deviations. This method works well for balanced foreground/background classes or multi-class segmentation tasks.

#### Common Evaluation Metrics for Image Segmentation

- **Intersection over Union (IoU):**
  - Measures overlap between predicted and ground truth masks.
  - Formula: (Area of Intersection) / (Area of Union).
  - Range: 0 (no overlap) to 1 (perfect overlap).
  - Widely used for object detection and segmentation.
- **Dice Coefficient / F1 Score:**
  - Similar to IoU but gives more weight to overlapping regions.
  - Formula: 2 * (Area of Intersection) / (Sum of both areas).
  - Range: 0 to 1.
  - Popular in medical image segmentation.
- **Pixel Accuracy:**
  - Ratio of correctly classified pixels to total pixels.
  - Simple but can be misleading with class imbalance.
  - Formula: (True Positives + True Negatives) / Total Pixels.
- **Mean Pixel Accuracy:**
  - Average per-class pixel accuracy.
  - Better for imbalanced classes.
  - Calculated separately for each class tand hen averaged.
- **Mean IoU (mIoU):**
  - Average IoU across all classes.
  - Standard metric for semantic segmentation.
  - Handles class imbalance well.
- **Boundary F1 Score:**
  - Focuses on segmentation boundaries.
  - Important for precise edge detection.
  - Useful when boundary accuracy is critical.
- **Precision and Recall:**
  - Precision: Accuracy of positive predictions.
  - Recall: Ability to find all positive instances.
  - Important for specific applications.

#### Comparison of U-Net with CNN and FCN

#### Traditional CNN vs U-Net

**Traditional CNN:**
- Primarily designed for classification tasks.
- Uses fully connected layers at the end.
- Loses spatial information through pooling.
- Output is a single class label.
- Not suitable for pixel-wise segmentation.

**U-Net:**
- Specifically designed for segmentation.
- Fully convolutional architecture.
- Preserves spatial information via skip connections.
- Output has the same resolution as input.
- Pixel-wise segmentation prediction.

#### FCN vs U-Net

**Fully Convolutional Network (FCN):**
- First architecture for end-to-end segmentation.
- Uses VGG/ResNet as encoder.
- Simple upsampling in the decoder.
- Limited skip connections.
- Lower resolution output is possible.

**U-Net:**
- Symmetric encoder-decoder structure.
- Custom encoder design.
- Sophisticated decoder with skip connections.
- Multiple skip connections at each level.
- Maintains high-resolution details.
- Better for medical image segmentation.

### Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/RezaSaadatYar/Medical-IMAGE-Processing.git
   cd Medical-IMAGE-Processing
   ```
2. Start with the notebooks in the Code/ directory to understand the workflow:
   - Begin with `01_fundamental.ipynb` for basics.
   - Progress through the U-Net (`02_unet.ipynb`, `03_projects_unet.ipynb`) and ResUNet (`06_resunet.ipynb`, `07_projects_resunet.ipynb`) notebooks for model training and projects.
   - Use `04_unet++.ipynb` and `08_resunet++.ipynb` for advanced implementations.
3. Use the Data/ directory datasets for training and evaluation.
4. Utility scripts in Code/Functions/ can be imported into notebooks for preprocessing, evaluation, and visualization.

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contact

For any questions or support, please contact Reza.Saadatyar@outlook.com
