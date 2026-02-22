# Handwritten Digit Recognition using ANN and CNN

## 📌 Project Overview

This project implements Handwritten Digit Recognition using Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) on the MNIST dataset.

The goal is to compare the performance of ANN and CNN models in classifying handwritten digits (0–9).

---

## 📂 Project Structure

```
DIGIT-RECOGNITION/
│
├── notebooks/
    └──.ipynb_checkpoints
    └── digit_recognition.ipynb
├── results/             
    └──accuracy_comparison.png
    └──ann_results.png
    └──cnn_results.png
├── LICENSE
└── README.md
```

---

## 📊 Dataset

Dataset Used: MNIST

- 70,000 grayscale images
- Image size: 28 × 28 pixels
- 10 classes (digits 0–9)
- 60,000 training images
- 10,000 testing images

The dataset is loaded using TensorFlow’s built-in loader.

---

## 🧠 Models Implemented

### 1️⃣ Artificial Neural Network (ANN)

- Input Layer: 784 neurons (flattened image)
- Hidden Layer 1: 128 neurons (ReLU)
- Hidden Layer 2: 64 neurons (ReLU)
- Output Layer: 10 neurons (Softmax)

### 2️⃣ Convolutional Neural Network (CNN)

- Conv2D (32 filters, 3×3)
- MaxPooling
- Conv2D (64 filters, 3×3)
- MaxPooling
- Flatten
- Dense (128 neurons)
- Output Layer (Softmax)

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📈 Results

The following result images are generated and saved inside the `results/` folder:

- ann_results.png  
- cnn_results.png  
- accuracy_comparison.png  

CNN achieves higher accuracy compared to ANN because it preserves spatial information and extracts features using convolutional layers.

---

## 🚀 How to Run the Project

1. Install required libraries:
   pip install tensorflow numpy matplotlib seaborn scikit-learn

2. Open Jupyter Notebook:
   jupyter notebook

3. Open:
   notebooks/digit_recognition.ipynb

4. Run all cells.

The result images will be automatically saved in the `results/` folder.

---

## 🎯 Conclusion

- ANN performs well for digit classification.
- CNN performs better due to spatial feature extraction.
- CNN achieves higher accuracy and better generalization.

---

## 📌 Author

Rajarshi Saha
