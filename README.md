**Medical Image Processing**

`Image classification` assigns <font color =#f57207>*a single label to an entire image*</font>, while `segmentation` goes a step further by <font color =#f57207>*labeling each pixel or group of pixels*</font>, effectively dividing the image into meaningful components. This enables more detailed analysis and supports applications like object detection, medical imaging, and autonomous driving, where identifying specific parts of an image is essential.

**Image Segmentation Types:**<br/>
<font color =#38f109><b>Semantic Segmentation:</b></font><br/>
▪ Assigns each pixel to a specific class<br/>
▪ Doesn't distinguish between instances of same class<br/>
▪ Example: All cars are labeled as "car" class<br/>

<font color =#38f109><b>Instance Segmentation:</b></font><br/>
▪ Detects and segments individual objects<br/>
▪ Distinguishes between instances of same class<br/>
▪ Example: Each car gets unique label (car1, car2)<br/>

<font color =#38f109><b>Panoptic Segmentation:</b></font><br/>
▪ Combines semantic and instance segmentation<br/>
▪ Segments both countable objects and background<br/>
▪ Example: Cars as instances + road/sky as semantic<br/>

<font color =#38f109><b>Binary Segmentation:</b></font><br/>
▪ Segments image into foreground and background<br/>
▪ Only two classes (0 or 1)<br/>
▪ Example: Object vs background<br/>

<font color =#38f109><b>Real-time Segmentation:</b></font><br/>
▪ Fast processing for video/streaming<br/>
▪ Often trades accuracy for speed<br/>
▪ Example: Self-driving car vision systems<br/>

**Segmentation methods:**<br/>
▪ <font color =#f17909><b>Manual annotation</b></font> involves marking image boundaries or regions of interest. Though reliable, it is time-consuming, labor-intensive, prone to errors, and unsuitable for large datasets or complex tasks due to consistency challenges.<br/>

▪ <font color =#f17909><b>Pixel-wise classification</b></font> independently classifies each pixel using algorithms like decision trees, SVMs, or random forests. However, it often fails to capture global context, spatial relationships, and accurate object boundaries, leading to segmentation issues.<br/>

▪ <font color =#f17909><b>U-Net's</b></font> architecture features a contracting path and an expansive path. The contracting path, similar to feedforward layers in other CNNs, uses encoder layers to reduce spatial resolution while capturing contextual information. The expansive path uses decoder layers to upsample feature maps and decode the data, aided by skip connections from the contracting path to preserve spatial details. This combination enables accurate feature localization and segmentation map generation.<br/>

**UNET** was developed to address the inefficiencies and inaccuracies of traditional image segmentation methods.<br/>
▪ <font color =#38f109><b>End-to-End Learning:</b></font><br/>
UNET employs an end-to-end learning approach, enabling it to segment images directly from input-output pairs without requiring user annotations. By training on a large, labeled dataset, UNET automatically extracts essential features and delivers precise segmentation, eliminating the need for time-consuming manual labeling.<br/>

▪ <font color =#38f109><b>Fully Convolutional Architecture:</b></font><br/>
UNET utilizes a fully convolutional architecture, consisting solely of convolutional layers without any fully connected layers. This design allows UNET to process input images of any size, enhancing its versatility and adaptability for diverse segmentation tasks and varying input dimensions.<br/>

▪ <font color =#38f109><b>U-shaped Architecture with Skip Connections:</b></font><br/> 
The network's architecture features encoding and decoding paths for local and global context, with skip connections preserving key information for precise segmentation.<br/>

▪ <font color =#38f109><b>Contextual Information and Localisation:</b></font><br/>
Skip connections in UNET merge multi-scale features, enhancing context absorption, detail capture, and improving segmentation accuracy with precise object boundaries.<br/>

▪ <font color =#38f109><b>Data Augmentation and Regularization:</b></font><br/>
UNET enhances resilience and generalization by using data augmentation, like rotations, flips, scaling, and deformations, to diversify training data, and regularization techniques, such as dropout and batch normalization, to prevent overfitting.<br/>

**Key Elements of the UNET Framework:**<br/>
▪ <font color =#f17909><b>Contracting Path (Encoding Path):</b></font><br/>
UNET's contracting path uses convolution and max pooling techniques to capture high-resolution, low-level features while reducing spatial dimensions. Each downsampling step doubles the number of feature channels, allowing for extracting features at various scales from the input image.<br/>

▪ <font color =#f17909><b>Expanding Path (Decoding Path):</b></font><br/> 
Transposed convolutions, or deconvolutions, upsample feature maps in the UNET expansion path to reconstruct a detailed segmentation map. This technique restores the features to match the resolution of the input image, ensuring precise alignment.<br/>

▪ <font color =#f17909><b>Bottleneck (Bottom)</b></font><br/>
Bridge between contracting and expansive paths. Has the highest number of feature channels.<br/>

▪ <font color =#f17909><b>Skip Connections:</b></font><br/>
Skip connections in UNET link encoding and decoding layers, preserving spatial details and enhancing segmentation by merging features from earlier layers. It concatenates the encoder feature map with the decoder, which helps the backward flow of gradients for improved training. After every concatenation we again apply two consecutive regular convolutions so that the model can learn to assemble a more precise output.<br/>

▪ <font color =#f17909><b>Fully Convolutional Layers:</b></font><br/> 
UNET uses convolutional layers, avoiding fully connected ones, allowing it to process images of any size while retaining spatial information for versatile segmentation tasks.<br/>

▪ <font color =#f17909><b>Final Layer:</b></font><br>
1x1 convolution to map feature vector to desired number of classes. Output is a pixel-wise segmentation map.<br/>

▪ <font color =#f17909><b>Dice Coefficient Loss:</b></font><br/>
The Dice coefficient measures overlap between predicted and true segmentation masks, while Dice loss (1 - Dice coefficient) minimizes as alignment improves. It's particularly effective for unbalanced datasets, encouraging accurate separation of foreground and background by penalizing false positives and negatives.<br/>

▪ <font color =#f17909><b>Cross-Entropy Loss:</b></font><br/>
It is used in image segmentation to measure dissimilarity between predicted class probabilities and ground truth labels. It treats each pixel as an independent classification problem, encouraging high probabilities for correct classes and penalizing deviations. This method works well for balanced foreground/background classes or multi-class segmentation tasks.<br/>

**Common Evaluation Metrics for Image Segmentation:**<br/>
<font color =#09f1ab><b>Intersection over Union (IoU):</b></font><br/>
▪ Measures overlap between predicted and ground truth masks<br/>
▪ Formula: (Area of Intersection) / (Area of Union)<br/>
▪ Range: 0 (no overlap) to 1 (perfect overlap)<br/>
▪ Widely used for object detection and segmentation<br/>

<font color =#09f1ab><b>Dice Coefficient / F1 Score:</b></font><br/>
▪ Similar to IoU but gives more weight to overlapping regionsv<br/>
▪ Formula: 2 * (Area of Intersection) / (Sum of both areas)<br/>
▪ Range: 0 to 1<br/>
▪ Popular in medical image segmentation<br/>

<font color =#09f1ab><b>Pixel Accuracy:</b></font><br/>
▪ Ratio of correctly classified pixels to total pixels<br/>
▪ Simple but can be misleading with class imbalance<br/>
▪ Formula: (True Positives + True Negatives) / Total Pixels<br/>

<font color =#09f1ab><b>Mean Pixel Accuracy:</b></font><br/>
▪ Average per-class pixel accuracy<br/>
▪ Better for imbalanced classes<br/>
▪ Calculated separately for each class then averaged<br/>

<font color =#09f1ab><b>Mean IoU (mIoU):</b></font><br/>
▪ Average IoU across all classes<br/>
▪ Standard metric for semantic segmentation<br/>
▪ Handles class imbalance well<br/>

<font color =#09f1ab><b>Boundary F1 Score:</b></font><br/>
▪ Focuses on segmentation boundaries<br/>
▪ Important for precise edge detection<br/>
▪ Useful when boundary accuracy is critical<br/>

<font color =#09f1ab><b>Precision and Recall:</b></font><br/>
▪ Precision: Accuracy of positive predictions<br/>
▪ Recall: Ability to find all positive instances<br/>
▪ Important for specific applications<br/>

**Comparison of U-Net with CNN and FCN:**<br/>
<font color =#f109b7><b>Traditional CNN vs U-Net:</b></font><br/>
**`Traditional CNN:`**<br/>
▪ Primarily designed for classification tasks<br/>
▪ Uses fully connected layers at the end<br/>
▪ Loses spatial information through pooling<br/>
▪ Output is a single class label<br/>
▪ Not suitable for pixel-wise segmentation<br/>

**`U-Net:`**<br/>
▪ Specifically designed for segmentation<br/>
▪ Fully convolutional architecture<br/>
▪ Preserves spatial information via skip connections<br/>
▪ Output has same resolution as input<br/>
▪ Pixel-wise segmentation prediction<br/>

<font color =#f109b7><b>FCN vs U-Net:</b></font><br/>
**`Fully Convolutional Network (FCN):`**<br/>
▪ First architecture for end-to-end segmentation<br/>
▪ Uses VGG/ResNet as encoder<br/>
▪ Simple upsampling in decoder<br/>
▪ Limited skip connections<br/>
▪ Lower resolution output possible<br/>

**`U-Net:`**<br/>
▪ Symmetric encoder-decoder structure<br/>
▪ Custom encoder design<br/>
▪ Sophisticated decoder with skip connections<br/>
▪ Multiple skip connections at each level<br/>
▪ Maintains high resolution details<br/>
▪ Better for medical image segmentation<br/>