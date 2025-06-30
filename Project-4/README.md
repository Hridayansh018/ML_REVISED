# 📰 Fake News Detection using Logistic Regression

This project demonstrates how to build a fake news classifier using logistic regression on a dataset of real and fake news articles. The pipeline includes loading, cleaning, preprocessing the data, training a model, and evaluating the results.

---

## 📦 1. Importing Dependencies

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import joblib
import string
These libraries provide the tools for data handling, vectorizing text, creating and evaluating the model, and cleaning text data.

📑 2. Loading the Dataset
python
fake = pd.read_csv('./Fake.csv')
true = pd.read_csv('./True.csv')
Two separate CSV files (Fake.csv, True.csv) are loaded—each representing fake and real news respectively.

🏷️ 3. Assigning Labels
python
fake['class'] = 0
true['class'] = 1
Fake news = 0, Real news = 1

🔗 4. Merging & Cleaning
python
data = pd.concat([fake, true], axis=0)
data = data.drop(['title', 'subject', 'date'], axis=1)
data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)
Merges the datasets and removes unnecessary columns to focus on the article text and class.

🧹 5. Text Preprocessing
python
def clean_text(text):
    ...
data['text'] = data['text'].apply(clean_text)
Function removes punctuation, numbers, links, special characters and converts text to lowercase. This makes the model focus on meaningful words only.

✂️ 6. Splitting the Dataset
python
x = data['text']
y = data['class']
xtrain, xtext, ytrain, ytest = train_test_split(x, y, random_state=2, test_size=0.25)
Dataset is split into 75% training and 25% testing.

🧠 7. TF-IDF Vectorization
python
vectorizer = TfidfVectorizer()
xvtrain = vectorizer.fit_transform(xtrain)
xvtest = vectorizer.transform(xtext)
Converts text data to numerical format using TF-IDF so it can be understood by the ML model.

🏋️ 8. Model Training
python
lr = LogisticRegression()
lr.fit(xvtrain, ytrain)
A logistic regression model is trained using the vectorized training data.

📈 9. Evaluation
python
predictions = lr.predict(xvtest)
lr.score(xvtest, ytest)
print(classification_report(ytest, predictions))
✅ Sample Output (depends on your dataset):
              precision    recall  f1-score   support

           0       0.98      0.96      0.97      2356
           1       0.96      0.98      0.97      2334

    accuracy                           0.97      4690
   macro avg       0.97      0.97      0.97      4690
weighted avg       0.97      0.97      0.97      4690
Precision: Out of all predicted positives, how many were actual.

Recall: Out of all actual positives, how many did we catch.

F1-Score: Harmonic mean of precision and recall.

Accuracy: Overall correctness of the model.