# Multispectral Satellite Water Segmentation

##  Project Overview
This project focuses on **water/flood segmentation from multispectral satellite imagery** using deep learning.

Satellite images contain **12 spectral bands**, which provide more environmental information compared to traditional RGB images.

The objective of this project is to build a model capable of **detecting and segmenting water bodies** using semantic segmentation techniques.

Two deep learning approaches were implemented:

1️ **Custom U-Net (Keras / TensorFlow)**  
2️ **Transfer Learning U-Net (PyTorch + ResNet34 Encoder)**  

The models were trained and evaluated to compare their performance on water segmentation tasks.

---

#  Dataset Description

Each sample in the dataset contains:

### Satellite Image
- Format: `.tif`
- Size: `128 × 128`
- Channels: **12 spectral bands**

### Segmentation Mask
- Format: `.png`
- Binary mask

```
1 = Water
0 = Background
```

---

#  Spectral Bands

Each satellite image contains the following **12 channels**:

1. Coastal Aerosol  
2. Blue  
3. Green  
4. Red  
5. NIR  
6. SWIR1  
7. SWIR2  
8. QA Band  
9. Merit DEM  
10. Copernicus DEM  
11. ESA World Cover Map  
12. Water Occurrence Probability  

These bands provide valuable environmental and spectral information useful for water detection.

---

#  Data Preprocessing

The preprocessing pipeline included:

- Convert images to **float32**
- **Band-wise normalization**
- Convert masks to **binary format**
- Preserve original spatial resolution `128×128`

Dataset split:

```
Train: 80%
Validation: 10%
Test: 10%
```

---

#  Data Augmentation

To improve model generalization, several augmentation techniques were applied:

- Horizontal Flip
- Vertical Flip
- Random Rotation
- Shift / Scale / Rotate
- Random Brightness & Contrast

These augmentations increase dataset diversity and reduce overfitting.

---

#  Models

Two segmentation models were implemented and trained.

---

# 1️ Custom U-Net (Keras / TensorFlow)

A fully customized **U-Net architecture** was implemented using Keras.

### Model Characteristics

```
Input Shape: 128 × 128 × 12
Architecture: Encoder–Decoder (U-Net)
Output: Binary segmentation mask
```

### Training Setup

```
Loss Function:
Binary Crossentropy

Optimizer:
Adam

Metric:
Jaccard Coefficient (IoU)
```

### Training Results

The model achieved strong performance during training:

```
Validation Accuracy ≈ 0.978
Validation IoU ≈ 0.90
```

Training stopped using **Early Stopping** to prevent overfitting.

Example training log:

```
Epoch 158 → Best Validation IoU ≈ 0.9017
```

This model achieved the **best performance in this project**.

---

# 2️ Transfer Learning U-Net (PyTorch)

A **U-Net with ResNet34 encoder** was implemented using PyTorch.

### Key Idea

Since most pretrained models expect **3-channel RGB images**, the first convolution layer was modified to accept:

```
Input Channels = 12
```

All layers remained **trainable** to adapt to multispectral satellite data.

### Model Configuration

```
Encoder: ResNet34
Input Channels: 12
Output Classes: 1
Activation: None (logits output)
```

### Training Setup

```
Loss Function:
BCEWithLogitsLoss + Dice Loss

Optimizer:
Adam

Learning Rate:
5e-5
```

### Training Performance

```
Train IoU ≈ 0.77
Validation IoU ≈ 0.83

```
---

#  Model Evaluation

The models were evaluated on the **test dataset**.

### Test Results (Transfer Learning Model)

```
Test IoU: 0.8129
Precision: 0.9389
Recall: 0.8582
F1 Score: 0.8967
```

These results indicate strong performance for water segmentation in multispectral satellite imagery.

---

#  Evaluation Metrics

The following metrics were used to evaluate segmentation performance:

### IoU (Intersection over Union)
Measures overlap between predicted mask and ground truth.

### Precision
Percentage of predicted water pixels that are correct.

### Recall
Percentage of real water pixels detected by the model.

### F1 Score
Balanced measure between precision and recall.

---

#  Visualization

Model predictions were visualized using:

- Satellite Image (RGB band combination)
- Ground Truth Mask
- Predicted Segmentation Mask

This allows visual inspection of segmentation performance.

---

#  Technologies Used

The project was implemented using:

- Python
- PyTorch
- TensorFlow / Keras
- segmentation-models-pytorch
- Rasterio
- OpenCV
- Albumentations
- NumPy
- Matplotlib
- Scikit-learn

---

#  Project Structure
```

project/
│
├── data/
│ ├── images
│ └── labels
│
├── notebooks/
│ └── training_notebook.ipynb
│
├── models/
│ └── best_model.pth
│ └── best_keras_model.h5
│
└── README.md
```

---

#  Future Improvements

Possible improvements include:

- Using larger satellite datasets
- Trying advanced segmentation architectures
- Applying Test-Time Augmentation (TTA)
- Training with multispectral pretrained models

---

#  Author

**Armia Gamal**  
Computer Science & Statistics Student  
AI Engineer | Computer Vision | Deep Learning
