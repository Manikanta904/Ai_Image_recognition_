# 🚀 Vision AI in 5 Days: Image Recognition using CNN

**Author:** Chilakala Manikanta Sai Anurudh  
**College:** KKR & KSR Institute of Technology & Sciences  

## 📌 Project Overview
This project implements an image recognition system using **Convolutional Neural Networks (CNNs)** to classify handwritten digits from the **MNIST dataset**.  
The model was trained and evaluated in **Google Colab** using **TensorFlow/Keras**, achieving high accuracy on both training and validation sets.

---

## 🎯 Goals
- Build and train a custom CNN for digit recognition.
- Apply data preprocessing and normalization.
- Evaluate performance with accuracy, loss curves, and confusion matrix.
- Compare results with transfer learning approaches.

---

## 🗂 Dataset
- **Source:** [MNIST Handwritten Digits](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  
- **Classes:** 10 (Digits 0–9)  
- **Image Size:** 28x28 pixels, grayscale

---

## 🛠 Tech Stack
- **Language:** Python  
- **Libraries:** TensorFlow, Keras, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Platform:** Google Colab (GPU enabled)  

---

## 🔄 Workflow
1. **Environment Setup** – Installed dependencies & enabled GPU.
2. **Data Preparation** – Loaded MNIST dataset, normalized pixel values, visualized samples.
3. **Model Design** – Built a CNN with Conv2D, MaxPooling, Flatten, and Dense layers.
4. **Training Phase** – Trained for 5 epochs with accuracy and loss tracking.
5. **Evaluation** – Accuracy, precision, recall, F1-score, and confusion matrix.
6. **Transfer Learning** – Compared performance with MobileNetV2.

---

## 📊 Results

### **Accuracy vs Epochs**
![Accuracy Curve](accuracy_plot.png)

### **Confusion Matrix**
![Confusion Matrix](confusion_matrix.png)

- **Training Accuracy:** 0.97 → 0.99  
- **Validation Accuracy:** ~0.98  
- **Test Accuracy:** **98%**  
- Strong classification with minimal misclassifications.

---

## 🏆 Key Takeaways
- CNNs excel in digit classification tasks.
- Minimal overfitting due to balanced dataset and regularization.
- Transfer learning improves training speed and sometimes accuracy.

---

## 🔮 Future Work
- Train on larger, more diverse datasets.
- Implement advanced augmentation for robustness.
- Deploy the model as a real-time web app.

---

## 📜 License
This project is licensed for educational purposes under the MIT License.

---
