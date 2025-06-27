
# SONAR Object Classification using Logistic Regression

## 📌 Problem Statement

In this project, we aim to build a **binary classification system** that can distinguish between **Rocks** and **Mines** based on SONAR signal data. Each object returns a signal that is processed into 60 numerical features. The target variable is either:

* `R` – Rock
* `M` – Mine

This model is useful in undersea exploration and defense applications where identifying objects via sonar is critical.

---

## 📂 Dataset Overview

* The dataset used is the **"Sonar Dataset"**, where:

  * Each row has **60 numerical features** (attributes of the sonar signal).
  * The 61st column is the **label**: `R` for Rock and `M` for Mine.
* Source: `sonar data.csv`

---

## 🛠️ Technologies Used

* Python
* Pandas & NumPy
* scikit-learn (for model building and evaluation)

---

## ✅ Steps in the Solution

### 1. 📥 Importing Required Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

### 2. 📊 Data Loading & Preprocessing

```python
df = pd.read_csv('./sonar data.csv', header=None)
X = df.drop(columns=60, axis=1)
Y = df[60]
```

* Checked shape, statistical summary, and class distribution
* Labels: `R` for Rock, `M` for Mine

---

### 3. 📚 Splitting Data into Train and Test Sets

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=2)
```

---

### 4. 🧠 Model Training - Logistic Regression

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

---

### 5. 📈 Model Evaluation

```python
# Training data accuracy
X_Train_Predictions = model.predict(X_train)
train_accuracy = accuracy_score(X_Train_Predictions, Y_train)

# Test data accuracy
X_Test_Predictions = model.predict(X_test)
test_accuracy = accuracy_score(X_Test_Predictions, Y_test)

print('Training Accuracy:', train_accuracy)
print('Testing Accuracy:', test_accuracy)
```

---

### 6. 🔮 Predictive System

You can input a new SONAR signal reading (60 values) and predict whether it is a Rock or a Mine.

```python
# Example input
input_data = (0.0293, 0.0644, 0.0390, ..., 0.0011)  # shortened for brevity
input_data_np = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_data_np)

if prediction[0] == 'R':
    print("The object is a Rock")
else:
    print("The object is a Mine")
```

---

## 🎯 Final Output

* The model provides **accurate predictions** for unknown sonar signals.
* It performs reasonably well on both training and test datasets.
* This makes it suitable for use in real-time object detection scenarios undersea.

---

## 📝 Notes

* Consider using **standard scaling** for further improvement.
* You can also explore more complex models like SVM, Random Forest, or Neural Networks for comparison.

---

## 📎 File Structure

```
sonar_classifier/
│
├── sonar_data.csv
├── sonar_model.py      
└── Sonar.md             
```

---
