# 🧠 Brain Tumor Classification using CNN

This project uses **Deep Learning** techniques, specifically **Convolutional Neural Networks (CNNs)**, to automatically classify **MRI brain scans** into four categories — **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**. The model was implemented using **PyTorch** and achieved an impressive **~99% test accuracy** through effective preprocessing, data augmentation, and model optimization.

---

## 🚀 Key Features

* **Automated Brain Tumor Classification** using deep CNNs
* **Four-Class Prediction**: Glioma, Meningioma, Pituitary, No Tumor
* **Advanced Data Augmentation & Normalization** for improved generalization
* **Model Training and Evaluation** using PyTorch and TorchVision
* **Visualizations** of accuracy, loss curves, and confusion matrix using Matplotlib
* **Efficient Image Processing** using PIL and NumPy

---

## 🧩 Tech Stack

| Category                | Tools / Libraries |
| ----------------------- | ----------------- |
| Programming Language    | Python            |
| Deep Learning Framework | PyTorch           |
| Image Processing        | TorchVision, PIL  |
| Data Handling           | NumPy, Pandas     |
| Model Evaluation        | Scikit-learn      |
| Visualization           | Matplotlib        |

---

## 📁 Dataset

The dataset used consists of MRI brain scan images categorized into:

* **Glioma Tumor**
* **Meningioma Tumor**
* **Pituitary Tumor**
* **No Tumor**

> 📦 Dataset Source: [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)



## 🧠 Model Architecture

The CNN model is composed of:

* **3 Convolutional Layers** with ReLU activation
* **MaxPooling Layers** for spatial reduction
* **Dropout Layers** for regularization
* **Fully Connected Layers** for classification
* **Softmax Output** for 4-class probability prediction

> Loss Function: CrossEntropyLoss
> Optimizer: Adam
> Learning Rate Scheduler for adaptive training

---

## 📊 Training and Evaluation

To train the model:

```bash
python train.py
```

To test or evaluate the model:

```bash
python evaluate.py
```

### 📈 Sample Metrics:

* Training Accuracy: 99.2%
* Validation Accuracy:98.8%
* Test Accuracy:~99%
* Loss: 0.02

## 🧾 Future Improvements

* Integrate Grad-CAM for model interpretability
* Deploy using  Streamlit or Flask for real-time MRI classification
* Fine-tune with Transfer Learning using pretrained models like ResNet or EfficientNet


## 🙌 Acknowledgments

* Dataset by Sartaj Bhuvaji on Kaggle
* Tutorials and references from PyTorch documentation
* Medical imaging research community


