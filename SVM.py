import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[["v1", "v2"]]

df.columns = ["label", "message"]

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print(df.head())

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

model = SVC(kernel="linear")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix :")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_message = ["Congratulations! You have won a free iPhone. Click now!"]

new_message = vectorizer.transform(new_message)

prediction = model.predict(new_message)

if prediction[0] == 1:
    print("\nPrediction: SPAM")
else:
    print("\nPrediction: NOT SPAM")


plt.figure(figsize=(6, 4))

plt.bar(["Ham", "Spam"], [
    (df["label"] == 0).sum(),
    (df["label"] == 1).sum()
])

plt.xlabel("Message Type")
plt.ylabel("Number of Messages")
plt.title("Spam vs Ham Messages")

plt.show()


plt.figure(figsize=(6, 4))

plt.imshow(cm)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["Ham", "Spam"])
plt.yticks([0, 1], ["Ham", "Spam"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()