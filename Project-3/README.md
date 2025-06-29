# 🧠 Diabetes Prediction using Machine Learning

This project builds a system to predict whether an individual is diabetic based on certain health parameters. It uses the **Support Vector Machine (SVM)** algorithm with a **linear kernel** for classification.

---

## 📁 1. Importing Dependencies

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

- `numpy` and `pandas` for data manipulation.
- `StandardScaler` for data standardization.
- `train_test_split` for splitting dataset into training and testing.
- `svm` from `sklearn` for building the classification model.
- `accuracy_score` to evaluate performance.

---

## 📊 2. Loading & Exploring the Dataset

```python
df = pd.read_csv('./diabetes.csv')
df.head()
df.shape
df.describe()
df['Outcome'].value_counts()
df.groupby('Outcome').mean()
```

- Reads the dataset containing health data of individuals.
- `Outcome` column:
  - `0` → Non-diabetic
  - `1` → Diabetic
- Basic EDA shows the distribution and feature means across both classes.

---

## 🧹 3. Data Preprocessing

### Separating Features and Labels

```python
X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']
```

- `X`: all input features (glucose, BMI, etc.).
- `Y`: labels (0 or 1).

---

## 🧪 4. Data Standardization

```python
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
```

- Scales all features to have a mean of 0 and standard deviation of 1.
- Helps SVM perform better, especially with different units in input features.

---

## 🔀 5. Splitting Dataset

```python
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=3)
```

- Splits the dataset into training (70%) and testing (30%) sets.
- `stratify=Y` ensures class balance in both sets.

---

## 🧠 6. Training the Support Vector Machine Model

```python
classifier = svm.SVC(kernel='linear')
classifier.fit(xtrain, ytrain)
```

- Trains an SVM classifier with a **linear** kernel.

---

## 📈 7. Evaluating the Model

```python
train_predict = classifier.predict(xtrain)
train_accuracy = accuracy_score(train_predict, ytrain)

test_predict = classifier.predict(xtest)
test_accuracy = accuracy_score(test_predict, ytest)
```

- Checks accuracy on both training and testing sets.
- Gives insights into potential underfitting or overfitting.

---

## 🤖 8. Making Predictions

```python
input_data = (5,116,74,0,0,25.6,0.201,30)
input_array = np.array(input_data).reshape(1,-1)
std_input = scaler.transform(input_array)
prediction = classifier.predict(std_input)

if prediction == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
```

- Accepts new data for prediction.
- Performs preprocessing just like training data.
- Classifies the individual as diabetic or non-diabetic.

---

## ✅ Summary

This simple ML pipeline demonstrates how healthcare data can be leveraged to build predictive models using Python and scikit-learn. The SVM model performs classification after properly standardizing and splitting the dataset.

---