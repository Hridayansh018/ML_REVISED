## 📦 **Importing the Dependencies**

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

* `numpy` & `pandas`: Used for numerical computations and data manipulation.
* `seaborn`: For creating visualizations.
* `train_test_split`: Splits the dataset into training and test subsets.
* `svm`: Module from `sklearn` used for implementing Support Vector Machines.
* `accuracy_score`: Measures the accuracy of predictions.

---

## 📊 **Data Collection and Preprocessing**

### 🔽 Loading the dataset

```python
df = pd.read_csv('./train_u6lujuX_CVtuZ9i (1).csv')
df.head()
```

* Loads the CSV dataset into a pandas DataFrame.
* `df.head()` shows the first five rows of the dataset.

### 📐 Dataset Shape

```python
df.shape
```

* Returns the number of rows and columns in the dataset.

### 📈 Statistical Measures

```python
df.describe()
```

* Provides summary statistics (mean, std, min, etc.) for numerical columns.

### ❓ Number of Missing Values

```python
df.isnull().sum()
```

* Shows the count of missing (null) values for each column.

### 🧹 Dropping Missing Values

```python
df = df.dropna()
df.isnull().sum()
df.shape
```

* Removes any rows that have missing values.
* Checks again to confirm there are no missing values.
* Displays the new shape of the DataFrame.

---

## 🔤 Label Encoding

### ✅ Encode Target Column

```python
df.replace({'Loan_Status':{'N':0, 'Y':1}}, inplace=True)
```

* Converts the `Loan_Status` column from 'Y'/'N' to 1/0.

### 👨‍👩‍👧 Dependents Column

```python
df['Dependents'].value_counts()
df = df.replace(to_replace='3+', value=4)
df['Dependents'].value_counts()
```

* Shows how many people are in the `Dependents` column.
* Replaces '3+' with 4 for numerical consistency.

---

## 📊 **Data Visualization**

### 🧑‍🎓 Education & Loan Status

```python
sns.countplot(x='Education', data=df, hue='Loan_Status')
```

* Creates a bar chart to visualize the count of graduates vs. non-graduates and how many of each got loan approval.

### 💍 Marital Status & Loan Status

```python
sns.countplot(x='Married', data=df, hue='Loan_Status')
```

* Visualizes how marital status relates to loan approval.

---

## 🔄 Convert Categorical to Numerical

```python
df.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0}
}, inplace=True)
```

* Converts multiple categorical columns into numerical values so they can be used for model training.

---

## 📂 Separating Data and Labels

```python
X = df.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = df['Loan_Status']
```

* `X`: Features (input variables).
* `Y`: Target label (Loan\_Status).

---

## ✂️ Splitting Data

```python
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=3)
```

* Splits data into 80% training and 20% testing.
* Uses `stratify=Y` to maintain class distribution.
* `random_state=3` ensures reproducibility.

---

## 🤖 **Training the Model**

```python
classifier = svm.SVC(kernel='linear')
classifier.fit(xtrain, ytrain)
```

* Initializes an SVM classifier with a linear kernel.
* Trains it on the training data.

---

## 📊 **Model Evaluation**

### 🎯 Accuracy on Training Data

```python
train_predictions = classifier.predict(xtrain)
train_score = accuracy_score(train_predictions, ytrain)
print('Accuracy for training data : ', train_score)
```

* Predicts outcomes for training data.
* Calculates and prints accuracy.

### 🧪 Accuracy on Test Data

```python
test_predictions = classifier.predict(xtest)
test_score = accuracy_score(test_predictions, ytest)
print('Accuracy for training data : ', test_score)
```

* Predicts outcomes for test data.
* Calculates and prints accuracy to see how well the model generalizes.

---
