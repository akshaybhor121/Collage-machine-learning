
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    "StudyHours": [1, 2, 2.5, 3, 3.5, 4, 5, 6],
    "Result": [0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

print("Dataset:")
print(df)

X = df[["StudyHours"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

new_student = pd.DataFrame({
    "StudyHours": [4.5]
})

prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

print("NEW STUDENT RESULT")

print("Study Hours:", new_student["StudyHours"].iloc[0])
print("Probability of Fail:",
      probability[0][0])

print("Probability of Pass:",
      probability[0][1])

if prediction[0] == 1:
    print("Prediction: PASS ")
else:
    print("Prediction: FAIL ")

print("\nIntercept:", model.intercept_)
print("Coefficient:", model.coef_)