---

# 🏠 Boston Housing Price Prediction using XGBoost

This notebook explores the Boston Housing dataset using Python and applies machine learning with **XGBoost Regressor** to predict median home values. Despite the initial title mentioning "Diabetes Prediction," this project focuses on house prices (`MEDV` feature).

---

## 🧰 1. Importing Dependencies

We begin by importing essential libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
```

- `numpy` and `pandas`: For numerical computations and data handling.
- `matplotlib.pyplot` and `seaborn`: For data visualization.
- `sklearn` and `xgboost`: For model building and evaluation.

---

## 📥 2. Loading the Dataset

```python
hpd = pd.read_csv('./boston.csv')
hpd.head()
```

This loads the dataset into a DataFrame named `hpd` and shows the first few rows.

---

## 📏 3. Initial Data Exploration

### Shape and Summary

```python
hpd.shape
hpd.describe()
```

- `.shape` shows the number of rows and columns.
- `.describe()` summarizes statistics like mean, standard deviation, etc.

### Missing Values

```python
hpd.isnull().sum()
```

Ensure the dataset has no missing values.

---

## 📈 4. Correlation Analysis

We calculate and visualize the correlation between features to understand what influences `MEDV`.

```python
correlation = hpd.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, fmt='.1f', annot=True, annot_kws={'size':8})
```

A heatmap helps identify the most influential predictors.

---

## 🎯 5. Feature and Target Splitting

```python
X = hpd.drop(['MEDV'], axis=1)
Y = hpd['MEDV']
```

- `X`: All input features except the target.
- `Y`: Target variable – Median Home Value (`MEDV`).

---

## ✂️ 6. Train-Test Split

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

- 80% training, 20% testing split for model validation.

---

## ⚙️ 7. Training XGBoost Regressor

```python
model = XGBRegressor()
model.fit(X_train, Y_train)
```

We train the model using gradient-boosted decision trees—known for excellent regression performance.

---

## 📊 8. Evaluating the Model

### On Training Data

```python
train_prediction = model.predict(X_train)
r2_train = metrics.r2_score(Y_train, train_prediction)
```

### On Test Data

```python
test_prediction = model.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_prediction)
```

- `R² score`: Measures model accuracy. Closer to 1 is better.

---

## 📉 9. Visualizing Predictions

```python
plt.scatter(Y_test, test_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Home Prices")
plt.show()
```

A scatter plot shows how well the model's predictions align with real values.

---

## ✅ Conclusion

built a complete regression pipeline:
- Loaded and explored the dataset.
- Analyzed feature relationships.
- Trained an XGBoost model.
- Evaluated performance using visual and statistical metrics.
