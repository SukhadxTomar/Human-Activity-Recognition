# 🧠 Human Activity Recognition using Deep Learning

## 📌 Project Overview

This project focuses on **Human Activity Recognition (HAR)** using deep learning models.
The system analyzes images of people and automatically classifies the activity being performed.

Activities include:

* Walking
* Eating
* Drinking
* Sleeping
* Calling
* Clapping
* And more...

---

## 🎯 Objective

* Build a deep learning model to classify human activities from images
* Compare different architectures (EfficientNet, ResNet, VGG)
* Achieve high accuracy and reliable predictions
* Provide prediction confidence for each output

---

## 📂 Project Structure

```
MAJORPROJECT/
│── Data/
│── models/
│── output/
│── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│── requirements.txt
│── test.py
```

---

## 📊 Dataset

* ~12,600 images
* ~15 activity classes
* Split into training and testing sets

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn

---

## 🧹 Data Preprocessing

* Image resizing (224x224)
* Normalization
* Label encoding
* Noise reduction

---

## 🤖 Models Used

* **EfficientNet (Best Performance)**
* ResNet50
* VGG16

---

## 🏋️ Model Training

* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Metrics: Accuracy
* Batch Size: Configurable
* Epochs: Configurable

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 🔍 Prediction

The model predicts activity along with confidence score:

Example:

```
Image: sleeping (0.97 confidence)
Image: eating (0.82 confidence)
```

---

## 🚀 How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model

```
python src/train.py
```

### 3. Run predictions

```
python src/predict.py
```

---

## 📊 Results

* EfficientNet achieved highest accuracy
* Model performs well on clear images
* Some low-confidence predictions on ambiguous data

---

## 🔮 Future Improvements

* Real-time prediction system
* Streamlit web app interface
* Model optimization
* Data augmentation
* Explainable AI (Grad-CAM, LIME)

---

## 🎓 Conclusion

This project demonstrates that deep learning models can effectively recognize human activities from images, with EfficientNet providing the best performance.

---

## 🙌 Author

Sukhad Tomar

## ⭐ If you like this project

Give it a star on GitHub!
