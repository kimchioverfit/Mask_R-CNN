# 📐 Target object Instance Segmentation & Diameter Measurement

This project implements an **instance segmentation–based inference pipeline** for detecting **Specific object** in images and accurately measuring their **diameters** using a trained **Mask R-CNN** model.

The system is designed for **industrial inspection and measurement scenarios**, where pixel-level segmentation results are converted into **physically meaningful measurements (μm)** within a valid **Region of Interest (ROI)**.

---

## 🔁 Overall Pipeline Overview

```
Input Image
   ↓
Preprocessing
   ↓
Model Transform
   ↓
Mask R-CNN Inference
   ↓
Confidence & Mask Filtering
   ↓
ROI Extraction
   ↓
Contour Detection
   ↓
Minimum Enclosing Circle
   ↓
Diameter Calculation (pixel → μm)
   ↓
Range Filtering
   ↓
Visualization & Result Export
```

---

## 📦 Core Components

### 1. Input

- RGB image (`.jpg`, `.png`)
- Trained Mask R-CNN model
- Pixel-to-micron scaling factor (`pixel_per_μm`)
- User-defined diameter range

---

### 2. Image Preprocessing

```python
preprocess_image(img_path)
```

Loads an image, converts it to RGB, normalizes it into a PyTorch tensor, and returns the tensor along with original image dimensions.

---

### 3. Model Transform

```python
transform_image(tensor_image, device)
```

Applies evaluation transforms and moves the tensor to the target device.

---

### 4. Mask Prediction & Filtering

```python
get_filtered_masks(model, x, mask_thresh, box_thresh)
```

Runs Mask R-CNN inference and filters masks using confidence thresholds.

---

### 5. ROI Extraction

```python
image_rgb, roi, gap = getROI(image_rgb)
```

Computes a valid region of interest and grid spacing for spatial indexing.

---

### 6. Circle Fitting & Diameter Measurement

For each valid mask:
- Extract contours
- Fit a minimum enclosing circle
- Convert pixel radius to physical diameter

```
diameter = radius / (pixel_per_μm × 2)
```

---

### 7. Visualization & Export

Annotated images and structured measurement results are saved to disk for review.

---

## 📤 Output Format

### Structured Output
```python
{
  "image_path": [
      [filename, section, center_coordinates, diameter],
      ...
  ]
}
```

---

## ⚙️ Key Parameters

| Parameter        | Description                      | Example |
|------------------|----------------------------------|---------|
| mask_thresh      | Mask confidence threshold        | 0.5     |
| box_thresh       | Detection score threshold        | 0.7     |
| pixel_per_μm     | Pixel-to-micron conversion       | 2.35    |
| diameter_range   | Valid diameter range (μm)        | [5, 50] |

---

## 🎯 Design Goals

- Accurate physical measurement
- ROI-based validation
- Robust filtering
- Operator-friendly visualization

---
